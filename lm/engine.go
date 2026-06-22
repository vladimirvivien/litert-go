// Package lm is an LLM runtime over the LiteRT CompiledModel C API. An
// Engine loads a .litertlm (or raw .tflite) model — its tokenizer, metadata,
// and prefill/decode/verify signatures — and generates text by driving the
// static-executor protocol: prefill an N-token prompt, then decode one token at
// a time against a fixed-context KV cache. It handles token-input models
// (gemma3, qwen3, …) and embedding-input models (gemma 3n/4, via separate
// embedder stages), with chat templating, temperature/top-k/top-p sampling, and
// MTP speculative decoding.
package lm
import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/vladimirvivien/litert-go/hftok"
	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
	"github.com/vladimirvivien/litert-go/sptok"
)

// maskNeg is the "masked" attention value LiteRT-LM uses for f32 (-0.7 * FLT_MAX).
const maskNeg = float32(-0.7 * math.MaxFloat32)

// tokenizer abstracts the model's tokenizer: SentencePiece or HF byte-level BPE.
// BOS/EOS return -1 when the tokenizer does not define them (the metadata
// start_token / stop_tokens take precedence).
type tokenizer interface {
	Encode(text string) []int32
	Decode(ids []int) string
	BOS() int32
	EOS() int32
}



// hfTokenizer adapts an HF byte-level BPE tokenizer. It defines no BOS/EOS; the
// metadata start_token / stop_tokens drive those.
type hfTokenizer struct{ t *hftok.Tokenizer }

func (h hfTokenizer) Encode(text string) []int32 { return h.t.Encode(text) }
func (h hfTokenizer) Decode(ids []int) string    { return h.t.Decode(ids) }
func (h hfTokenizer) BOS() int32                 { return -1 }
func (h hfTokenizer) EOS() int32                 { return -1 }

// Engine is a loaded, compiled model ready to generate text. It is not safe for
// concurrent use.
type Engine struct {
	env         litert.Environment
	model       litert.Model
	cm          litert.CompiledModel
	opts        litert.Options
	tok         tokenizer
	md          litertlm.Metadata
	fileBytes   []byte
	modelKey    string // per-model GPU program-cache namespace
	pre         prefiller
	decode      sig
	verify      sig
	haveVerify  bool
	accel       litert.HwAccelerator
	gpuCacheDir string
	metrics     func(DecodeStats)
	emb, ple    *embedModel     // text + per-layer embedder stages, compiled once on first use
	embOpts     litert.Options  // compile options backing emb/ple
	vision      *visionPipeline // vision encoder + adapter, compiled once on first use
	audio       *audioPipeline  // audio encoder + adapter, compiled once on first use
	lastText    string          // most recent streamed output (GenerateStream/SendStream)
	lastMetrics PerformanceMetrics
	kvPool      []*kvBanks
	kvMu        sync.Mutex
}

// envOptions builds the LiteRt environment options: the runtime-library
// directory, so accelerator plugins (e.g. libLiteRtWebGpuAccelerator) resolve
// from libDir without having to be on the OS search path.
func envOptions(libDir string) []litert.EnvOption {
	if libDir == "" {
		return nil
	}
	if fi, err := os.Stat(libDir); err == nil && !fi.IsDir() {
		libDir = filepath.Dir(libDir)
	}
	return []litert.EnvOption{{Tag: litert.EnvRuntimeLibraryDir, Str: libDir}}
}

// modelCacheKey is a namespace for a model's serialized GPU programs, unique to
// the file's identity (path, size, mtime) so a changed model does not reuse a
// stale cache.
func modelCacheKey(modelPath string) string {
	key := filepath.Base(modelPath)
	if fi, err := os.Stat(modelPath); err == nil {
		key = fmt.Sprintf("%s_%d_%d", key, fi.Size(), fi.ModTime().UnixNano())
	}
	return key
}

// defaultGPUCacheDir is the default directory for persisting compiled GPU
// programs across runs.
func defaultGPUCacheDir() string {
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "litert-go", "gpu-cache")
}

// gpuCompileOptions builds compilation options for accel. On GPU it attaches the
// "gpu_options" payload that persists compiled WebGPU programs to gpuCacheDir
// under modelKey+tag, so a warm run skips kernel recompilation. tag distinguishes
// the graphs compiled from one model (main decode, embedders, vision, audio) so
// their caches do not collide.
//
// mainKV marks a KV-bearing graph (main model or MTP drafter), whose KV cache
// is driven double-buffered. Those graphs keep the kv_cache_ tensors in the
// delegate's native layout — skipping a per-step layout conversion of every KV
// tensor, whose dispatch exceeds the WebGPU 65535-workgroup cap on large
// models (gemma-4) — and prepare command buffers two steps ahead, matching the
// C++ engine's configuration (llm_executor_settings_utils.cc). The native KV
// layout requires the two-bank KV scheme: the WebGPU delegate rejects an
// external buffer serving as both a run's input and its output.
//
// paramTensor marks models with an int32 param_tensor input (MTP models —
// gemma 4); the C++ engine additionally puts param_tensor in the external set
// and both patterns in buffer storage.
func gpuCompileOptions(accel litert.HwAccelerator, modelKey, cacheDir, tag string, mainKV, paramTensor bool) (litert.Options, error) {
	opts, err := litert.NewOptions(accel)
	if err != nil {
		return 0, err
	}
	if accel != litert.AccelGPU {
		return opts, nil
	}
	dir := cacheDir
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return opts, nil
	}
	// These match the C++ engine's GPU cache settings and produce its compact
	// split cache (a small program cache plus a separate weight cache) instead of a
	// single multi-GB blob. enable_constant_tensors_sharing is load-bearing: it
	// shares each constant (weight) tensor across the prefill-bucket and decode
	// signatures, so the cache stores the weights once instead of per signature.
	// serialize_external_tensors writes the deduplicated weights to the weight
	// cache; serialize_program_cache writes the compiled programs. Warm starts then
	// read ~0.5 GB instead of ~11 GB. num_threads_to_upload parallelizes the upload.
	// enable_infinite_float_capping clamps infinities under fp16 activations
	// (the C++ engine sets it for every GPU compile); without it large models
	// (gemma-4) propagate NaNs into garbage output.
	toml := fmt.Sprintf(
		"model_cache_key = \"%s_%s\"\n"+
			"serialization_dir = \"%s\"\n"+
			"serialize_program_cache = true\n"+
			"serialize_external_tensors = true\n"+
			"enable_constant_tensors_sharing = true\n"+
			"enable_infinite_float_capping = true\n"+
			"num_threads_to_upload = 2\n",
		modelKey, tag, filepath.ToSlash(dir))
	if mainKV {
		ext := "\"kv_cache_\""
		if paramTensor {
			ext += ", \"param_tensor\""
			toml += "buffer_storage_tensor_patterns = [\"kv_cache_\", \"param_tensor\"]\n"
		}
		toml += "external_tensor_patterns = [" + ext + "]\n" +
			"num_steps_of_command_buffer_preparations = 2\n"
	}
	if err := opts.AddOpaqueOption("gpu_options", toml); err != nil {
		opts.Close()
		return 0, err
	}
	return opts, nil
}

func (e *Engine) compileOptions(tag string) (litert.Options, error) {
	return gpuCompileOptions(e.accel, e.modelKey, e.gpuCacheDir, tag,
		tag == "main", sigHasInput(e.decode, "param_tensor"))
}

// singleKV reports whether the KV cache must run in single-buffer mode (GPU +
// param_tensor models; see kvBanks).
func (e *Engine) singleKV() bool {
	return e.accel == litert.AccelGPU && sigHasInput(e.decode, "param_tensor")
}

// Open reads the .litertlm or .tflite model at modelPath, extracts its main
// generation graph, tokenizer, and metadata, and compiles it for the
// configured accelerator (CPU by default). The runtime libraries resolve from
// WithLibDir, then the LITERT_LIB environment variable, then libfetch's
// default download location; WithFetch opts in to downloading them. Close
// releases everything. ctx covers Open itself (including an opted-in
// download), not later generation calls.
func Open(ctx context.Context, modelPath string, options ...Option) (*Engine, error) {
	var c openConfig
	c.accel = litert.AccelCPU
	for _, o := range options {
		o(&c)
	}
	libDir, err := resolveLibDir(ctx, &c)
	if err != nil {
		return nil, err
	}
	if err := litert.Load(libDir); err != nil {
		return nil, err
	}
	if c.gpuCacheDir == "" {
		c.gpuCacheDir = defaultGPUCacheDir()
	}
	envOpts := envOptions(libDir)
	if c.gpuCacheDir != "" {
		_ = os.MkdirAll(c.gpuCacheDir, 0o755)
		envOpts = append(envOpts, litert.EnvOption{
			Tag: litert.EnvCompilerCacheDir,
			Str: c.gpuCacheDir,
		})
	}
	if c.minLogLevel != nil {
		envOpts = append(envOpts, litert.EnvOption{
			Tag:    litert.EnvMinLoggerSeverity,
			IsInt:  true,
			IntVal: *c.minLogLevel,
		})
	}
	env, err := litert.NewEnvironment(envOpts...)
	if err != nil {
		return nil, err
	}
	e := &Engine{env: env, accel: c.accel, gpuCacheDir: c.gpuCacheDir, metrics: c.metrics}
	ok := false
	defer func() {
		if !ok {
			e.Close()
		}
	}()

	if e.fileBytes, err = os.ReadFile(modelPath); err != nil {
		return nil, err
	}
	e.modelKey = modelCacheKey(modelPath)
	tflite := e.fileBytes
	if litertlm.IsContainer(e.fileBytes) {
		if tflite, err = litertlm.SectionTFLite(e.fileBytes); err != nil {
			return nil, err
		}
		e.md, _ = litertlm.ReadMetadata(e.fileBytes)
		if sp, serr := litertlm.SectionBytes(e.fileBytes, litertlm.SectionSPTokenizer); serr == nil {
			tok, err := sptok.New(sp)
			if err != nil {
				return nil, fmt.Errorf("load tokenizer: %w", err)
			}
			e.tok = tok
		} else if hf, herr := litertlm.SectionBytes(e.fileBytes, litertlm.SectionHFTokenizerZlib); herr == nil {
			h, herr := hftok.LoadSection(hf)
			if herr != nil {
				return nil, fmt.Errorf("load HF tokenizer: %w", herr)
			}
			e.tok = hfTokenizer{h}
		}
	}
	if e.model, err = litert.OpenModelFromBuffer(env, tflite); err != nil {
		return nil, err
	}

	nsig, _ := e.model.NumSignatures()
	haveDecode := false
	for i := 0; i < nsig; i++ {
		s, _ := e.model.Signature(i)
		key, _ := s.Key()
		switch {
		case key == "decode":
			if e.decode, err = loadSig(e.model, i); err != nil {
				return nil, err
			}
			haveDecode = true
		case key == "verify":
			if e.verify, err = loadSig(e.model, i); err != nil {
				return nil, err
			}
			e.haveVerify = true
		case strings.HasPrefix(key, "prefill"):
			g, err := loadSig(e.model, i)
			if err != nil {
				return nil, err
			}
			if err := e.pre.add(g); err != nil {
				return nil, err
			}
		}
	}
	if e.pre.empty() || !haveDecode {
		return nil, fmt.Errorf("lm: model lacks prefill/decode signatures")
	}
	e.pre.sortSizes()

	if e.opts, err = e.compileOptions("main"); err != nil {
		return nil, err
	}
	if e.cm, err = litert.Compile(env, e.model, e.opts); err != nil {
		return nil, err
	}
	ok = true
	return e, nil
}

// Close releases the model, compiled model, and environment.
func (e *Engine) Close() {
	e.kvMu.Lock()
	for _, kv := range e.kvPool {
		kv.close()
	}
	e.kvPool = nil
	e.kvMu.Unlock()

	if e.emb != nil {
		e.emb.close()
	}
	if e.ple != nil {
		e.ple.close()
	}
	if e.embOpts != 0 {
		e.embOpts.Close()
	}
	if e.vision != nil {
		e.vision.close()
	}
	if e.audio != nil {
		e.audio.close()
	}
	if e.cm != 0 {
		e.cm.Close()
	}
	if e.opts != 0 {
		e.opts.Close()
	}
	if e.model != 0 {
		e.model.Close()
	}
	if e.env != 0 {
		e.env.Close()
	}
	runtime.KeepAlive(e.fileBytes)
}

// ensureEmbedders compiles the text embedder and, when the decode graph asks for
// it, the per-layer embedder, caching both on the Engine. Embedding-input models
// run every token through these stages, so they are compiled once and reused
// across Generate, speculative decoding, and sessions. The Engine is not safe
// for concurrent use, so the lazy initialization needs no lock.
func (e *Engine) ensureEmbedders() (*embedModel, *embedModel, error) {
	if e.emb != nil {
		return e.emb, e.ple, nil
	}
	// Embedders compile on CPU regardless of the engine accelerator, matching
	// the C++ engine (EmbeddingLookupText::Initialize). An embedding lookup is a
	// gather; running it on GPU buys nothing and forces the embedding table into
	// GPU memory and the GPU weight cache.
	opts, err := litert.NewOptions(litert.AccelCPU)
	if err != nil {
		return nil, nil, err
	}
	embSec, err := litertlm.SectionTFLiteModelType(e.fileBytes, litertlm.TFLiteEmbedder)
	if err != nil {
		opts.Close()
		return nil, nil, fmt.Errorf("embedder section: %w", err)
	}
	emb, err := newEmbedModel(e.env, opts, embSec)
	if err != nil {
		opts.Close()
		return nil, nil, fmt.Errorf("embedder: %w", err)
	}
	var ple *embedModel
	if sigHasInput(e.decode, "per_layer_embeddings") {
		pleSec, err := litertlm.SectionTFLiteModelType(e.fileBytes, litertlm.TFLitePerLayerEmbedder)
		if err != nil {
			emb.close()
			opts.Close()
			return nil, nil, fmt.Errorf("per-layer embedder section: %w", err)
		}
		if ple, err = newEmbedModel(e.env, opts, pleSec); err != nil {
			emb.close()
			opts.Close()
			return nil, nil, fmt.Errorf("per-layer embedder: %w", err)
		}
	}
	e.emb, e.ple, e.embOpts = emb, ple, opts
	return emb, ple, nil
}

// Metadata returns the model's parsed LlmMetadata.
func (e *Engine) Metadata() litertlm.Metadata { return e.md }

// HasTokenizer reports whether the model shipped a SentencePiece tokenizer.
func (e *Engine) HasTokenizer() bool { return e.tok != nil }

// Tokenize converts text to model token IDs: the raw tokenizer
// encoding, with no BOS token or chat-template affixes.
func (e *Engine) Tokenize(text string) ([]int32, error) {
	if e.tok == nil {
		return nil, ErrNoTokenizer
	}
	return e.tok.Encode(text), nil
}

// Detokenize converts model token IDs back to text: the raw tokenizer
// decoding.
func (e *Engine) Detokenize(ids []int32) (string, error) {
	if e.tok == nil {
		return "", ErrNoTokenizer
	}
	intIDs := make([]int, len(ids))
	for i, id := range ids {
		intIDs[i] = int(id)
	}
	return e.tok.Decode(intIDs), nil
}

// HasChatTemplate reports whether the model has chat affixes (for Generate with
// chat=true or NewChat).
func (e *Engine) HasChatTemplate() bool {
	_, ok := e.templates()
	return ok
}

// templates resolves the chat affixes. Containers without llm_model_type are
// family-inferred the way the C++ engine infers them — token id 105 decoding
// to <start_of_turn> identifies the gemma family — and inferred-gemma3 models
// take the canonical gemma3 affixes instead of the container's: published
// gemma3 containers carry malformed prompt_templates (gemma3-270m embeds
// literal backslash-n character pairs), and the canonical form is what the
// C++ engine renders for the family.
func (e *Engine) templates() (litertlm.PromptTemplates, bool) {
	if e.md.ModelType == litertlm.ModelUnknown && e.tok != nil &&
		e.tok.Decode([]int{105}) == "<start_of_turn>" {
		return litertlm.FallbackTemplates(litertlm.ModelGemma3)
	}
	return e.md.Templates()
}

// SupportsSpec reports whether MTP speculative decoding is available (a verify
// signature plus an embedding-input main graph).
func (e *Engine) SupportsSpec() bool {
	return e.haveVerify && sigHasInput(e.decode, "embeddings")
}

// Message is one message in a conversation history.
type Message struct {
	Role string // "user", "model", "system"
	Text string
}

// GenOptions configures one generation.
type GenOptions struct {
	MaxTokens         int              // cap on generated tokens
	Temp              float32          // sampling temperature; 0 = greedy
	TopK              int              // top-k filter; 0 = off
	TopP              float32          // top-p / nucleus filter; 0 = off
	Seed              int64            // sampling RNG seed
	Spec              bool             // use MTP speculative decoding when available (greedy only)
	System            string           // optional system prompt, rendered at the conversation start (chat only)
	ToolsJSON         string           // OpenAI-style tool specs, rendered into the conversation start (tool-capable families only)
	Tools             []ToolDefinition // high-level tool definitions for auto-dispatching
	MaxToolHops       int              // max hops in tool execution loop
	History           []Message        // optional pre-populated conversation history
	JSONSchema          string           // optional schema constraint
	SchemaInstruction   string           // optional custom instruction for structured fallback path
	Retries             int              // number of retries for parsing structured data
	ConstrainedDecoding bool             // toggle engine's constrained decoding mode
}

// Generate completes prompt and returns the generated text. When chat is true
// the prompt is wrapped in the model's chat template.
func (e *Engine) Generate(ctx context.Context, prompt string, chat bool, o GenOptions) (string, error) {
	if e.tok == nil {
		return "", fmt.Errorf("%w; use GenerateIDs", ErrNoTokenizer)
	}
	ids, err := e.buildPrompt(o.System, prompt, chat)
	if err != nil {
		return "", err
	}
	gen, err := e.generate(ctx, ids, o, nil)
	if err != nil {
		return "", err
	}
	return e.tok.Decode(gen), nil
}

// GenerateIDs runs the model on explicit prompt token IDs and returns the
// generated token IDs. For models without a tokenizer.
func (e *Engine) GenerateIDs(ctx context.Context, prompt []int32, o GenOptions) ([]int, error) {
	return e.generate(ctx, prompt, o, nil)
}

// GenerateStream is Generate with incremental output: onPiece is called with
// each newly decoded text fragment as it is produced. It returns the full text.
func (e *Engine) GenerateStream(ctx context.Context, prompt string, chat bool, o GenOptions, onPiece func(string)) (string, error) {
	if e.tok == nil {
		return "", fmt.Errorf("%w; use GenerateIDs", ErrNoTokenizer)
	}
	ids, err := e.buildPrompt(o.System, prompt, chat)
	if err != nil {
		return "", err
	}
	_, err = e.generate(ctx, ids, o, e.streamer(onPiece))
	if err != nil {
		return "", err
	}
	return e.lastText, nil
}

// LastMetrics returns the performance metrics from the most recent generation run or session turn.
func (e *Engine) LastMetrics() PerformanceMetrics {
	return e.lastMetrics
}

// GenerateWithMetrics runs synchronous generation and returns the generated text and the PerformanceMetrics.
func (e *Engine) GenerateWithMetrics(ctx context.Context, prompt string, chat bool, o GenOptions) (string, PerformanceMetrics, error) {
	text, err := e.Generate(ctx, prompt, chat, o)
	if err != nil {
		return "", PerformanceMetrics{}, err
	}
	return text, e.lastMetrics, nil
}

// GenerateStreamWithMetrics runs streaming generation and returns the generated text and the PerformanceMetrics.
func (e *Engine) GenerateStreamWithMetrics(ctx context.Context, prompt string, chat bool, o GenOptions, onPiece func(string)) (string, PerformanceMetrics, error) {
	text, err := e.GenerateStream(ctx, prompt, chat, o, onPiece)
	if err != nil {
		return "", PerformanceMetrics{}, err
	}
	return text, e.lastMetrics, nil
}

// streamer returns a per-token callback that decodes the running output and
// reports each new fragment. It records the full text in e.lastText.
func (e *Engine) streamer(onPiece func(string)) func(int32) {
	var ids []int
	e.lastText = ""
	return func(id int32) {
		ids = append(ids, int(id))
		full := e.tok.Decode(ids)
		if onPiece != nil && len(full) >= len(e.lastText) {
			onPiece(full[len(e.lastText):])
		}
		e.lastText = full
	}
}

func (e *Engine) generate(ctx context.Context, prompt []int32, o GenOptions, onToken func(int32)) ([]int, error) {
	smp := newSampler(o.Temp, o.TopK, o.TopP, o.Seed)
	stop := stopSet(e.tok, e.md)
	switch {
	case o.Spec && e.SupportsSpec():
		emb, ple, err := e.ensureEmbedders()
		if err != nil {
			return nil, err
		}
		return decodeSpeculative(ctx, e.env, e.cm, e.fileBytes, e.pre, e.decode, e.verify, emb, ple, prompt, o.MaxTokens, stop, e.accel, e.modelKey, e.gpuCacheDir, onToken)
	case sigHasInput(e.decode, "embeddings"):
		_, _, err := e.ensureEmbedders()
		if err != nil {
			return nil, err
		}
		return e.decodeEmbeddingInput(ctx, prompt, o.MaxTokens, stop, smp, onToken)
	default:
		return e.decodeTokenInput(ctx, prompt, o.MaxTokens, stop, smp, onToken)
	}
}

// Chat is a multi-turn conversation over an Engine. Each Send re-renders the
// whole history through the chat template and decodes the reply.
type Chat struct {
	e    *Engine
	o    GenOptions
	tpl  litertlm.PromptTemplates
	hist []turn
}

// NewChat starts a multi-turn chat. It fails if the model has no tokenizer or
// chat template.
func (e *Engine) NewChat(o GenOptions) (*Chat, error) {
	if e.tok == nil {
		return nil, ErrNoTokenizer
	}
	tpl, ok := e.templates()
	if !ok {
		return nil, fmt.Errorf("%w (model type %q)", ErrNoChatTemplate, e.md.ModelType)
	}
	c := &Chat{e: e, o: o, tpl: tpl}
	if o.System != "" {
		c.hist = append(c.hist, turn{role: "system", text: o.System})
	}
	for _, msg := range o.History {
		c.hist = append(c.hist, turn{role: msg.Role, text: msg.Text})
	}
	return c, nil
}

// Close releases the Chat. Chat holds no resources of its own (the Engine owns
// the model); it exists to satisfy Conversation.
func (c *Chat) Close() {}

// TokenCount returns the number of tokens in the conversation history.
func (c *Chat) TokenCount() int {
	prompt := buildConversation(c.e.tok, c.e.md, c.tpl, c.hist)
	return len(prompt)
}

// Send adds a user message and returns the model's reply, retaining both for
// context on subsequent calls.
func (c *Chat) Send(ctx context.Context, parts ...Part) (string, error) {
	return c.send(ctx, parts, nil)
}

// SendStream is Send with incremental output: onPiece is called with each newly
// decoded fragment of the reply as it is produced.
func (c *Chat) SendStream(ctx context.Context, parts []Part, onPiece func(string)) (string, error) {
	return c.send(ctx, parts, onPiece)
}

func (c *Chat) send(ctx context.Context, parts []Part, onPiece func(string)) (string, error) {
	userText, err := textPartsOnly(parts)
	if err != nil {
		return "", err
	}
	c.hist = append(c.hist, turn{role: "user", text: userText})
	prompt := buildConversation(c.e.tok, c.e.md, c.tpl, c.hist)
	var reply string
	if onPiece != nil {
		_, err := c.e.generate(ctx, prompt, c.o, c.e.streamer(onPiece))
		if err != nil {
			return "", err
		}
		reply = c.e.lastText
	} else {
		gen, err := c.e.generate(ctx, prompt, c.o, nil)
		if err != nil {
			return "", err
		}
		reply = c.e.tok.Decode(gen)
	}
	c.hist = append(c.hist, turn{role: "model", text: reply})
	return reply, nil
}

// Part is one piece of multimodal input to Conversation.Send/SendStream.
type Part struct {
	Kind   string // "text", "image", "audio"
	Text   string
	Data   []byte
	Mime   string
	Budget int // visual token budget for image parts
}

func textPartsOnly(parts []Part) (string, error) {
	var sb strings.Builder
	for _, p := range parts {
		if p.Kind != "text" && p.Kind != "" {
			return "", fmt.Errorf("lm: only text parts are supported on this conversation type")
		}
		sb.WriteString(p.Text)
	}
	return sb.String(), nil
}

// Conversation is a multi-turn chat session. Both Chat (re-prefills the history
// each turn) and Session (reuses the KV cache across turns) satisfy it.
// Implementations are not safe for concurrent use.
type Conversation interface {
	Send(ctx context.Context, parts ...Part) (string, error)
	SendStream(ctx context.Context, parts []Part, onPiece func(string)) (string, error)
	TokenCount() int
	Close()
}

// NewConversation starts a multi-turn chat with a KV-reuse session: a Session
// for token-input models, an embedSession (turns ingested via prefill-at-offset)
// for embedding-input models. Models without a tokenizer or chat template fall
// back to a re-prefill Chat. Always Close the result.
func (e *Engine) NewConversation(o GenOptions) (Conversation, error) {
	if e.tok == nil || !e.HasChatTemplate() {
		return e.NewChat(o)
	}
	if sigHasInput(e.decode, "embeddings") {
		return e.newEmbedSession(o)
	}
	return e.NewSession(o)
}

// Session is a multi-turn chat that keeps the KV cache across turns: each turn
// ingests only its new tokens at the cached position instead of re-prefilling
// the whole history. Token-input models only; embedding-input models use the
// equivalent embedSession via NewConversation. Close releases the cache.
type Session struct {
	e       *Engine
	o       GenOptions
	tpl     litertlm.PromptTemplates
	start   string // conversation-start render: system prompt + tool declarations
	stop    map[int32]bool
	kv      *kvBanks
	dec     *decoder
	pos     int
	started bool
}

// NewSession starts a KV-reuse chat session for a token-input model. It fails on
// an embedding-input model: those ingest each turn with a batched
// prefill-at-offset rather than a sequential decode (the i8 KV cache requires
// it), handled by the embedSession that NewConversation returns.
func (e *Engine) NewSession(o GenOptions) (*Session, error) {
	if e.tok == nil {
		return nil, ErrNoTokenizer
	}
	if sigHasInput(e.decode, "embeddings") {
		return nil, fmt.Errorf("lm: NewSession supports token-input models only")
	}
	tpl, ok := e.templates()
	if !ok {
		return nil, fmt.Errorf("%w (model type %q)", ErrNoChatTemplate, e.md.ModelType)
	}
	start, err := conversationStart(e, tpl, o)
	if err != nil {
		return nil, err
	}
	kv, err := e.getKVBanks(false)
	if err != nil {
		return nil, err
	}
	s := &Session{e: e, o: o, tpl: tpl, start: start, stop: stopSet(e.tok, e.md), kv: kv}
	dec, err := newDecoder(e.env, e.cm, e.decode, s.kv, newSampler(o.Temp, o.TopK, o.TopP, o.Seed))
	if err != nil {
		s.Close()
		return nil, err
	}
	s.dec = dec
	if len(o.History) > 0 {
		if err := s.ingestHistory(context.Background(), o.History); err != nil {
			s.Close()
			return nil, err
		}
	}
	return s, nil
}

// Close releases the session's KV cache and decode buffers.
func (s *Session) Close() {
	if s.dec != nil {
		s.dec.close()
	}
	if s.kv != nil {
		s.e.putKVBanks(s.kv)
	}
}

// TokenCount returns the number of tokens currently stored in the session's KV cache.
func (s *Session) TokenCount() int {
	return s.pos
}

// Send adds a user message and returns the reply.
func (s *Session) Send(ctx context.Context, parts ...Part) (string, error) {
	return s.send(ctx, parts, nil)
}

// SendStream is Send with incremental output.
func (s *Session) SendStream(ctx context.Context, parts []Part, onPiece func(string)) (string, error) {
	return s.send(ctx, parts, onPiece)
}

func (s *Session) send(ctx context.Context, parts []Part, onPiece func(string)) (string, error) {
	userText, err := textPartsOnly(parts)
	if err != nil {
		return "", err
	}
	if len(s.o.Tools) == 0 {
		return s.sendTurn(ctx, s.tpl.User.Prefix+userText+s.tpl.User.Suffix, onPiece)
	}
	return s.sendWithDispatch(ctx, s.tpl.User.Prefix+userText+s.tpl.User.Suffix, onPiece)
}

// SendToolResults delivers function results to the model and decodes
// its follow-up turn. The conversation must target a tool-capable
// family.
func (s *Session) SendToolResults(ctx context.Context, results []ToolResult) (string, error) {
	return s.SendToolResultsStream(ctx, results, nil)
}

// SendToolResultsStream is SendToolResults with incremental output.
func (s *Session) SendToolResultsStream(ctx context.Context, results []ToolResult, onPiece func(string)) (string, error) {
	turn, err := toolResultsTurn(s.e, results)
	if err != nil {
		return "", err
	}
	return s.sendTurn(ctx, turn, onPiece)
}

// sendTurn ingests one rendered turn and decodes the model's reply.
// The first turn carries the start token and the conversation-start
// render; later turns first close the previous model turn with its
// suffix.
func (s *Session) sendTurn(ctx context.Context, turn string, onPiece func(string)) (string, error) {
	render := turn + s.tpl.Model.Prefix
	var ids []int32
	if !s.started {
		ids = startIDs(s.e.tok, s.e.md)
		render = s.start + render
		s.started = true
	} else {
		render = modelBoundary(s.e.md, s.tpl) + render
	}
	ids = append(ids, s.e.tok.Encode(render)...)
	if len(ids) == 0 {
		return "", fmt.Errorf("empty turn")
	}

	// Ingest all but the turn's last token through the prefill graphs at the
	// session position — the prefill-then-decode sequencing Generate and the
	// C++ executor use. Do not ingest through consecutive asynchronous decode
	// runs: the WebGPU delegate corrupts the KV cache under that pattern. The
	// held-back token's decode logits start the reply.
	cacheHits := s.pos
	n := len(ids)
	tPrefillStart := time.Now()
	if n > 1 {
		if err := prefillTokenRun(ctx, s.e.env, s.e.cm, s.e.pre, s.kv, ids[:n-1], s.pos); err != nil {
			return "", err
		}
	}
	prefillDuration := time.Since(tPrefillStart)

	tFirstTokenStart := time.Now()
	pos := s.pos + n - 1
	id, err := s.dec.step(ids[n-1], pos)
	if err != nil {
		return "", err
	}
	timeToFirstToken := prefillDuration + time.Since(tFirstTokenStart)
	pos++

	var stream func(int32)
	if onPiece != nil {
		stream = s.e.streamer(onPiece)
	}
	var gen []int
	t0 := time.Now()
	for g := 0; g < s.o.MaxTokens; g++ {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		if s.stop[id] {
			break
		}
		gen = append(gen, int(id))
		if stream != nil {
			stream(id)
		}
		if err := s.dec.feed(id, pos); err != nil {
			return "", err
		}
		pos++
		if id, err = s.dec.sample(); err != nil {
			return "", err
		}
	}
	decodeDuration := time.Since(t0)
	s.pos = pos

	s.e.lastMetrics = PerformanceMetrics{
		PrefillDuration:  prefillDuration,
		DecodeDuration:   decodeDuration,
		TimeToFirstToken: timeToFirstToken,
		PrefillTokens:    n,
		DecodeTokens:     len(gen),
		CacheHits:        cacheHits,
	}
	if decodeDuration > 0 && len(gen) > 0 {
		s.e.lastMetrics.TokensPerSecond = float64(len(gen)) / decodeDuration.Seconds()
	}

	return s.e.tok.Decode(gen), nil
}

// --- signatures ---

// sig bundles a signature with its index and input/output name ordering.
type sig struct {
	idx       int
	s         litert.Signature
	inNames   []string
	outNames  []string
	inByName  map[string]int
	outByName map[string]int
}

func loadSig(m litert.Model, idx int) (sig, error) {
	s, err := m.Signature(idx)
	if err != nil {
		return sig{}, err
	}
	g := sig{idx: idx, s: s, inByName: map[string]int{}, outByName: map[string]int{}}
	nin, _ := s.NumInputs()
	for i := 0; i < nin; i++ {
		name, err := s.InputName(i)
		if err != nil {
			return sig{}, err
		}
		g.inNames = append(g.inNames, name)
		g.inByName[name] = i
	}
	nout, _ := s.NumOutputs()
	for i := 0; i < nout; i++ {
		name, err := s.OutputName(i)
		if err != nil {
			return sig{}, err
		}
		g.outNames = append(g.outNames, name)
		g.outByName[name] = i
	}
	return g, nil
}

func isKV(name string) bool { return strings.HasPrefix(name, "kv_cache_") }

// tokenInput returns the name of a token-input signature's token IDs tensor;
// models vary between "tokens" and "token_ids".
func tokenInput(g sig) string {
	if sigHasInput(g, "tokens") {
		return "tokens"
	}
	return "token_ids"
}

// sigHasInput reports whether the signature has an input tensor of the given name.
func sigHasInput(g sig, name string) bool {
	_, ok := g.inByName[name]
	return ok
}

func inputShape(g sig, name string) ([]int32, error) {
	tt, err := g.s.InputType(name)
	if err != nil {
		return nil, err
	}
	return tt.Shape, nil
}

// --- prefill bucketing ---

// prefiller is a model's set of prefill signatures, keyed by sequence-length
// bucket (the size of the tokens/embeddings input). A prompt longer than a
// single bucket is covered by several prefill calls at increasing positions; a
// short prompt uses the smallest bucket that fits.
type prefiller struct {
	sizes []int // ascending
	sig   map[int]sig
}

// prefillChunk is one prefill call in a plan: a bucket signature of size bucket
// fed take real tokens (take <= bucket; the rest of the bucket is padding).
type prefillChunk struct {
	bucket int
	take   int
}

func (p *prefiller) add(g sig) error {
	size, err := bucketSize(g)
	if err != nil {
		return err
	}
	if p.sig == nil {
		p.sig = map[int]sig{}
	}
	if _, dup := p.sig[size]; !dup {
		p.sizes = append(p.sizes, size)
		p.sig[size] = g
	}
	return nil
}

func (p *prefiller) sortSizes() { sort.Ints(p.sizes) }
func (p prefiller) empty() bool { return len(p.sizes) == 0 }

// max returns the largest-bucket signature. Its KV-cache inputs are shared by
// every bucket, so it is used to allocate the cache.
func (p prefiller) max() sig { return p.sig[p.sizes[len(p.sizes)-1]] }

// plan splits n tokens into chunks: repeated max-bucket chunks for the bulk,
// then the smallest bucket that fits the remainder.
func (p prefiller) plan(n int) []prefillChunk {
	maxSize := p.sizes[len(p.sizes)-1]
	var plan []prefillChunk
	for n > 0 {
		if n >= maxSize {
			plan = append(plan, prefillChunk{maxSize, maxSize})
			n -= maxSize
			continue
		}
		bucket := maxSize
		for _, s := range p.sizes {
			if s >= n {
				bucket = s
				break
			}
		}
		plan = append(plan, prefillChunk{bucket, n})
		n = 0
	}
	return plan
}

// bucketSize reads a prefill signature's sequence-length bucket from its token
// or embedding input shape ([1, bucket] or [1, bucket, H]).
func bucketSize(g sig) (int, error) {
	for _, name := range []string{"tokens", "token_ids", "embeddings"} {
		if sh, err := inputShape(g, name); err == nil && len(sh) >= 2 {
			return int(sh[1]), nil
		}
	}
	return 0, fmt.Errorf("prefill signature has no tokens/token_ids/embeddings input")
}

// allocKV allocates a zeroed buffer for every KV-cache input of g, keyed by
// name. The cache is shared across prefill, decode, and verify.
func allocKV(env litert.Environment, cm litert.CompiledModel, g sig) (map[string]litert.TensorBuffer, error) {
	kv := map[string]litert.TensorBuffer{}
	for _, name := range g.inNames {
		if !isKV(name) {
			continue
		}
		buf, err := allocReqInput(env, cm, g, name)
		if err != nil {
			for _, b := range kv {
				b.Close()
			}
			return nil, fmt.Errorf("alloc %s: %w", name, err)
		}
		kv[name] = buf
	}
	return kv, nil
}

// kvBanks is a double-buffered KV cache. Each Run reads one bank and writes the
// full updated cache to the other, then the banks swap. Alternating banks lets
// the GPU delegate reuse pre-prepared command buffers across steps (resource
// bindings repeat with period two) instead of re-encoding every dispatch, and
// keeps each Run's KV reads and writes on distinct buffers.
//
// In single-buffer mode (GPU + param_tensor models — gemma 4) both banks are
// the same buffer set and swap is a no-op: the delegate's add_values_to_cache
// kernel scatter-writes only the rows param_tensor names, in place — the graph
// never emits a full cache copy, so a second bank would lose all prior KV. The
// aliased in/out binding is legal under the buffer_storage_tensor_patterns the
// compile sets for these models. This is the C++ executor's
// gpu_optimized_single_buffer_cache mode.
type kvBanks struct {
	a, b   map[string]litert.TensorBuffer
	bIn    bool // false: a is the current (valid) input bank; true: b is
	single bool
}

// allocKVBanks allocates the KV cache from g's buffer requirements: one buffer
// set in single-buffer mode, two banks otherwise. Single-buffer KV is created
// from the signature's *output* requirements — the input-side requirements for
// these delegate-managed (external + buffer storage) tensors are unavailable
// and would fall back to host buffers the delegate's in-place scatter never
// writes. Matches the C++ executor's CreateOutputBuffer in
// gpu_optimized_single_buffer_cache mode.
func allocKVBanks(env litert.Environment, cm litert.CompiledModel, g sig, single bool) (*kvBanks, error) {
	if single {
		a := map[string]litert.TensorBuffer{}
		for _, name := range g.outNames {
			if !isKV(name) {
				continue
			}
			buf, err := allocReqOutput(env, cm, g, name)
			if err != nil {
				closeBufs(a)
				return nil, fmt.Errorf("alloc %s: %w", name, err)
			}
			a[name] = buf
		}
		return &kvBanks{a: a, b: a, single: true}, nil
	}
	a, err := allocKV(env, cm, g)
	if err != nil {
		return nil, err
	}
	b, err := allocKV(env, cm, g)
	if err != nil {
		closeBufs(a)
		return nil, err
	}
	return &kvBanks{a: a, b: b}, nil
}

func (e *Engine) getKVBanks(single bool) (*kvBanks, error) {
	e.kvMu.Lock()
	n := len(e.kvPool)
	for i := 0; i < n; i++ {
		kv := e.kvPool[i]
		if kv.single == single {
			// Remove from pool
			e.kvPool[i] = e.kvPool[n-1]
			e.kvPool = e.kvPool[:n-1]
			e.kvMu.Unlock()
			if err := kv.clear(); err != nil {
				kv.close()
				return allocKVBanks(e.env, e.cm, e.pre.max(), single)
			}
			return kv, nil
		}
	}
	e.kvMu.Unlock()
	return allocKVBanks(e.env, e.cm, e.pre.max(), single)
}

func (e *Engine) putKVBanks(kv *kvBanks) {
	if kv == nil {
		return
	}
	e.kvMu.Lock()
	e.kvPool = append(e.kvPool, kv)
	e.kvMu.Unlock()
}

func (k *kvBanks) clear() error {
	for _, buf := range k.a {
		if err := buf.Clear(); err != nil {
			return err
		}
	}
	if !k.single {
		for _, buf := range k.b {
			if err := buf.Clear(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (k *kvBanks) in() map[string]litert.TensorBuffer {
	if k.bIn {
		return k.b
	}
	return k.a
}

func (k *kvBanks) out() map[string]litert.TensorBuffer {
	if k.bIn {
		return k.a
	}
	return k.b
}

// inBind returns the bank to bind as run inputs. In single-buffer mode it is
// nil: the delegate sources the cache from its registered external tensors,
// and the KV input slots are passed as null handles (the runtime accepts
// unbound inputs for delegate-managed tensors).
func (k *kvBanks) inBind() map[string]litert.TensorBuffer {
	if k.single {
		return nil
	}
	return k.in()
}

// allowNullIn reports whether a zero handle is acceptable for input name —
// KV inputs stay unbound in single-buffer mode.
func (k *kvBanks) allowNullIn(name string) bool { return k.single && isKV(name) }

func (k *kvBanks) swap() { k.bIn = !k.bIn }
func (k *kvBanks) close() {
	closeBufs(k.a)
	if !k.single {
		closeBufs(k.b)
	}
}

// prefillTokenRun ingests ids into the KV cache starting at position start,
// chunking across prefill buckets for prompts longer than one bucket.
func prefillTokenRun(ctx context.Context, env litert.Environment, cm litert.CompiledModel, pre prefiller, kv *kvBanks, ids []int32, start int) error {
	off := 0
	for _, c := range pre.plan(len(ids)) {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := prefillStep(env, cm, pre.sig[c.bucket], kv, ids[off:off+c.take], start+off); err != nil {
			return err
		}
		off += c.take
	}
	return nil
}

// --- prompt building ---

// turn is one message in a multi-turn conversation.
type turn struct {
	role string // "user", "model", "system"
	text string
}

// buildPrompt tokenizes text: as a raw completion (start token + text) or, with
// chat set, wrapped in the model's chat template.
func (e *Engine) buildPrompt(system, text string, chat bool) ([]int32, error) {
	if chat {
		return e.buildChatPrompt(system, text)
	}
	ids := append(startIDs(e.tok, e.md), e.tok.Encode(text)...)
	if len(ids) == 0 {
		return nil, fmt.Errorf("empty tokenization of %q", text)
	}
	return ids, nil
}

// renderSystem renders a system turn (System.Prefix + system + System.Suffix), or
// empty when no system prompt is set.
func renderSystem(tpl litertlm.PromptTemplates, system string) string {
	if system == "" {
		return ""
	}
	return tpl.System.Prefix + system + tpl.System.Suffix
}

// buildChatPrompt wraps userText in the model's single user turn: start ++
// user.prefix ++ userText ++ user.suffix ++ model.prefix. The turn markers in
// the affixes (e.g. <start_of_turn>) are user-defined tokenizer pieces, so
// encoding the rendered string yields their single vocab IDs.
func (e *Engine) buildChatPrompt(system, userText string) ([]int32, error) {
	tok, md := e.tok, e.md
	tpl, ok := e.templates()
	if !ok {
		return nil, fmt.Errorf("no chat template for model type %q", md.ModelType)
	}
	rendered := renderSystem(tpl, system) + tpl.User.Prefix + userText + tpl.User.Suffix + tpl.Model.Prefix
	ids := append(startIDs(tok, md), tok.Encode(rendered)...)
	if len(ids) < 2 {
		return nil, fmt.Errorf("empty chat tokenization of %q", userText)
	}
	return ids, nil
}

// buildConversation renders a chat history into prompt token IDs: the start
// token, then each turn wrapped in its role affixes, then the model prefix to
// open the assistant's reply. The whole conversation is encoded as one string
// so control-token boundaries tokenize correctly.
func buildConversation(tok tokenizer, md litertlm.Metadata, tpl litertlm.PromptTemplates, hist []turn) []int32 {
	affix := func(role string) litertlm.Affixes {
		switch role {
		case "model":
			return tpl.Model
		case "system":
			return tpl.System
		default:
			return tpl.User
		}
	}
	var sb strings.Builder
	for _, h := range hist {
		a := affix(h.role)
		sb.WriteString(a.Prefix)
		sb.WriteString(h.text)
		if h.role == "model" {
			sb.WriteString(modelBoundary(md, tpl))
		} else {
			sb.WriteString(a.Suffix)
		}
	}
	sb.WriteString(tpl.Model.Prefix)

	return append(startIDs(tok, md), tok.Encode(sb.String())...)
}

// modelBoundary returns the text that closes an assistant turn before the next
// turn: the model affix suffix, or — when a model leaves that empty (e.g. phi-4)
// — the first stop-token string, so the assistant turn stays delimited in
// multi-turn context.
func modelBoundary(md litertlm.Metadata, tpl litertlm.PromptTemplates) string {
	if tpl.Model.Suffix != "" {
		return tpl.Model.Suffix
	}
	for _, st := range md.StopTokens {
		if st.Str != "" {
			return st.Str
		}
	}
	return ""
}

// startIDs returns the start token IDs to prepend: the metadata start_token when
// given as IDs, none for the "None" sentinel, otherwise the tokenizer's BOS when
// it has one.
func startIDs(tok tokenizer, md litertlm.Metadata) []int32 {
	if len(md.StartToken.IDs) > 0 {
		return append([]int32(nil), md.StartToken.IDs...)
	}
	if md.StartToken.Str == "None" {
		return nil
	}
	if bos := tok.BOS(); bos >= 0 {
		return []int32{bos}
	}
	return nil
}

// stopSet collects the token IDs that end generation: the metadata stop_tokens
// (IDs directly, single-token strings resolved through the tokenizer) plus the
// tokenizer's end-of-sentence token. Multi-token stop strings are skipped.
func stopSet(tok tokenizer, md litertlm.Metadata) map[int32]bool {
	set := map[int32]bool{}
	if tok == nil {
		return set
	}
	for _, st := range md.StopTokens {
		for _, id := range st.IDs {
			set[id] = true
		}
		if st.Str != "" {
			if enc := tok.Encode(st.Str); len(enc) == 1 {
				set[enc[0]] = true
			}
		}
	}
	if eos := tok.EOS(); eos >= 0 {
		set[eos] = true
	}
	return set
}

// --- token-input decode ---

// decodeTokenInput runs the static token-input pipeline (gemma3, qwen3, …):
// allocate the KV cache once, prefill all but the last prompt token, then
// decode from the held-back token.
func (e *Engine) decodeTokenInput(ctx context.Context, prompt []int32, ngen int, stop map[int32]bool, smp *sampler, onToken func(int32)) ([]int, error) {
	kv, err := e.getKVBanks(false)
	if err != nil {
		return nil, err
	}
	defer e.putKVBanks(kv)

	tPrefillStart := time.Now()
	p := len(prompt) - 1
	if err := prefillTokenRun(ctx, e.env, e.cm, e.pre, kv, prompt[:p], 0); err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	prefillDuration := time.Since(tPrefillStart)

	dec, err := newDecoder(e.env, e.cm, e.decode, kv, smp)
	if err != nil {
		return nil, fmt.Errorf("decode setup: %w", err)
	}
	defer dec.close()

	next := prompt[p]
	pos := p
	var gen []int
	var timeToFirstToken time.Duration
	t0 := time.Now()
	for g := 0; g < ngen; g++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		id, err := dec.step(next, pos)
		if err != nil {
			return nil, fmt.Errorf("decode step %d: %w", g, err)
		}
		if g == 0 {
			timeToFirstToken = prefillDuration + time.Since(t0)
		}
		if stop[id] {
			break
		}
		gen = append(gen, int(id))
		if onToken != nil {
			onToken(id)
		}
		next = id
		pos++
	}
	decodeDuration := time.Since(t0)

	e.lastMetrics = PerformanceMetrics{
		PrefillDuration:  prefillDuration,
		DecodeDuration:   decodeDuration,
		TimeToFirstToken: timeToFirstToken,
		PrefillTokens:    p,
		DecodeTokens:     len(gen),
		CacheHits:        0,
	}
	if decodeDuration > 0 && len(gen) > 0 {
		e.lastMetrics.TokensPerSecond = float64(len(gen)) / decodeDuration.Seconds()
	}

	if e.metrics != nil {
		e.metrics(DecodeStats{Tokens: len(gen), Decode: decodeDuration})
	}
	return gen, nil
}

// prefillStep ingests ids into the KV cache at position start through bucket g:
// the tokens fill the first len(ids) slots, input_pos runs [start, start+seq),
// and a causal mask lets row r attend [0, start+r+1) so earlier chunks stay
// visible. Bucket slots past len(ids) hold padding whose KV is later overwritten
// or never attended.
func prefillStep(env litert.Environment, cm litert.CompiledModel, g sig, kv *kvBanks, ids []int32, start int) error {
	tokens, err := allocReqInput(env, cm, g, tokenInput(g))
	if err != nil {
		return err
	}
	defer tokens.Close()
	pos, err := allocReqInput(env, cm, g, "input_pos")
	if err != nil {
		return err
	}
	defer pos.Close()
	mask, err := allocReqInput(env, cm, g, "mask")
	if err != nil {
		return err
	}
	defer mask.Close()

	if err := writeInts(tokens, ids); err != nil { // first len(ids) slots; rest stay 0
		return err
	}
	maskShape, _ := inputShape(g, "mask") // [1,1,seq,ctx]
	seq, ctx := int(maskShape[2]), int(maskShape[3])
	posVals := make([]int32, seq)
	for i := range posVals {
		posVals[i] = int32(start + i)
	}
	if err := writeInts(pos, posVals); err != nil {
		return err
	}
	maskType, _ := g.s.InputType("mask")
	if err := fillCausalMask(mask, seq, ctx, start, maskType.ElementType); err != nil {
		return err
	}

	perCall := map[string]litert.TensorBuffer{tokenInput(g): tokens, "input_pos": pos, "mask": mask}
	in := assemble(g.inNames, perCall, kv.in())
	out := assemble(g.outNames, nil, kv.out()) // prefill outputs are all KV
	for i, b := range in {
		if b == 0 {
			return fmt.Errorf("unmapped input[%d] %q", i, g.inNames[i])
		}
	}
	for i, b := range out {
		if b == 0 {
			return fmt.Errorf("unmapped output[%d] %q", i, g.outNames[i])
		}
	}
	// KV banks carry events from prior asynchronous decode runs (turn 2+ of a
	// session); the runtime rejects output buffers with attached events.
	for _, b := range out {
		if err := b.ClearEvent(); err != nil {
			return err
		}
	}
	if err := cm.Run(g.idx, in, out); err != nil {
		return err
	}
	kv.swap()
	return nil
}

// decoder holds the fixed decode buffer set and a pair of litert.Runners whose
// Run arguments are pinned once. tokens/input_pos/mask/logits are shared; the
// two runners differ only in which KV bank they read vs write (runA reads bank
// a and writes bank b; runB the reverse). Each step uses the runner whose input
// is the cache's current bank, then the banks swap.
type decoder struct {
	tokens     litert.TensorBuffer
	posBuf     litert.TensorBuffer
	mask       litert.TensorBuffer
	maskET     litert.ElementType
	logits     litert.TensorBuffer
	runA, runB *litert.Runner
	kv         *kvBanks
	ctx        int
	vocab      int
	smp        *sampler
	pending    bool // an async run was submitted and not yet awaited
}

func newDecoder(env litert.Environment, cm litert.CompiledModel, g sig, kv *kvBanks, smp *sampler) (*decoder, error) {
	d := &decoder{smp: smp, kv: kv}
	var err error
	if d.tokens, err = allocReqInput(env, cm, g, tokenInput(g)); err != nil {
		return nil, err
	}
	if d.posBuf, err = allocReqInput(env, cm, g, "input_pos"); err != nil {
		return nil, err
	}
	if d.mask, err = allocReqInput(env, cm, g, "mask"); err != nil {
		return nil, err
	}
	if maskType, err := g.s.InputType("mask"); err == nil {
		d.maskET = maskType.ElementType
	}
	if d.logits, err = allocReqOutput(env, cm, g, "logits"); err != nil {
		return nil, err
	}

	maskShape, _ := inputShape(g, "mask") // [1,1,1,ctx]
	d.ctx = int(maskShape[3])
	logitsType, err := g.s.OutputType("logits")
	if err != nil {
		return nil, err
	}
	d.vocab = 1
	for _, dim := range logitsType.Shape {
		d.vocab *= int(dim)
	}

	perCall := map[string]litert.TensorBuffer{tokenInput(g): d.tokens, "input_pos": d.posBuf, "mask": d.mask}
	logitsOut := map[string]litert.TensorBuffer{"logits": d.logits}
	runner := func(in, out map[string]litert.TensorBuffer) (*litert.Runner, error) {
		inB := assemble(g.inNames, perCall, in)
		outB := assemble(g.outNames, logitsOut, out)
		for i, b := range inB {
			if b == 0 {
				return nil, fmt.Errorf("unmapped input[%d] %q", i, g.inNames[i])
			}
		}
		for i, b := range outB {
			if b == 0 {
				return nil, fmt.Errorf("unmapped output[%d] %q", i, g.outNames[i])
			}
		}
		return litert.NewRunner(cm, g.idx, inB, outB), nil
	}
	if d.runA, err = runner(kv.a, kv.b); err != nil {
		return nil, err
	}
	if d.runB, err = runner(kv.b, kv.a); err != nil {
		return nil, err
	}
	return d, nil
}

// feed writes one token at pos and runs the model, leaving logits in d.logits.
// It does not sample — used to ingest known tokens into the KV cache. The run
// is submitted asynchronously when the backend supports it; sampling (or the
// next feed) waits for completion through the logits buffer's event. Input
// buffers must not be rewritten while a submitted run is in flight, so feed
// awaits any pending run before writing.
func (d *decoder) feed(token int32, pos int) error {
	if d.pending {
		if err := d.logits.Wait(); err != nil {
			return err
		}
		d.pending = false
	}
	if err := writeInts(d.tokens, []int32{token}); err != nil {
		return err
	}
	if err := writeInts(d.posBuf, []int32{int32(pos)}); err != nil {
		return err
	}
	if err := fillCausalMask(d.mask, 1, d.ctx, pos, d.maskET); err != nil {
		return err
	}
	run := d.runA // reads bank a
	if d.kv.bIn {
		run = d.runB // reads bank b
	}
	async, err := run.RunAsync()
	if err != nil {
		return err
	}
	d.kv.swap()
	d.pending = async
	return nil
}

// sample picks the next token from the logits of the most recent feed. Locking
// the logits buffer waits out any in-flight async run.
func (d *decoder) sample() (int32, error) {
	d.pending = false
	return d.smp.sample(d.logits, d.vocab)
}

func (d *decoder) step(token int32, pos int) (int32, error) {
	if err := d.feed(token, pos); err != nil {
		return 0, err
	}
	return d.sample()
}

func (d *decoder) close() {
	d.runA.Close()
	d.runB.Close()
	d.logits.Close()
	d.mask.Close()
	d.posBuf.Close()
	d.tokens.Close()
}

// --- shared tensor helpers ---

// allocReqInput allocates a zeroed buffer for a signature input using the
// compiled model's required size and buffer type.
func allocReqInput(env litert.Environment, cm litert.CompiledModel, g sig, name string) (litert.TensorBuffer, error) {
	tt, err := g.s.InputType(name)
	if err != nil {
		return 0, err
	}
	size, bt, err := cm.InputBufferInfo(g.idx, g.inByName[name])
	if err != nil {
		return 0, err
	}
	return newZeroedSized(env, bt, tt, size)
}

func allocReqOutput(env litert.Environment, cm litert.CompiledModel, g sig, name string) (litert.TensorBuffer, error) {
	tt, err := g.s.OutputType(name)
	if err != nil {
		return 0, err
	}
	size, bt, err := cm.OutputBufferInfo(g.idx, g.outByName[name])
	if err != nil {
		return 0, err
	}
	return newZeroedSized(env, bt, tt, size)
}

// assemble orders buffers by signature name list, resolving each name from the
// per-call map then the KV map.
func assemble(names []string, perCall, kv map[string]litert.TensorBuffer) []litert.TensorBuffer {
	out := make([]litert.TensorBuffer, len(names))
	for i, name := range names {
		if b, ok := perCall[name]; ok {
			out[i] = b
		} else {
			out[i] = kv[name]
		}
	}
	return out
}

// newZeroedSized allocates a managed buffer and zeroes it. The runtime's
// Clear handles delegate-managed buffers whose mapped window is smaller than
// the requirements size (zeroing those by hand faults); buffer types Clear
// does not support (host memory) are zeroed through Lock, where the mapped
// window covers the full size.
func newZeroedSized(env litert.Environment, bt litert.BufferType, tt litert.TensorType, size uint64) (litert.TensorBuffer, error) {
	if size%4 != 0 {
		size = size + 4 - (size % 4)
	}
	buf, err := litert.NewManagedBuffer(env, bt, tt, size)
	if err != nil {
		return 0, err
	}
	if err := buf.Clear(); err == nil {
		return buf, nil
	}
	addr, err := buf.Lock(litert.LockWrite)
	if err != nil {
		buf.Close()
		return 0, err
	}
	clear(unsafe.Slice((*byte)(addr), size))
	if err := buf.Unlock(); err != nil {
		buf.Close()
		return 0, err
	}
	return buf, nil
}

func writeInts(b litert.TensorBuffer, vals []int32) error {
	addr, err := b.Lock(litert.LockWrite)
	if err != nil {
		return err
	}
	copy(unsafe.Slice((*int32)(addr), len(vals)), vals)
	return b.Unlock()
}

// fillCausalMask fills a [1,1,rows,ctx] mask: row r (position startPos+r)
// attends columns [0, startPos+r+1), the rest are masked.
func fillCausalMask(b litert.TensorBuffer, rows, ctx, startPos int, et litert.ElementType) error {
	addr, err := b.Lock(litert.LockWrite)
	if err != nil {
		return err
	}
	if et == litert.ElementBool {
		m := unsafe.Slice((*byte)(addr), rows*ctx)
		for i := range m {
			m[i] = 0 // false (masked)
		}
		for r := 0; r < rows; r++ {
			open := startPos + r + 1
			if open > ctx {
				open = ctx
			}
			row := m[r*ctx : r*ctx+ctx]
			for c := 0; c < open; c++ {
				row[c] = 1 // true (attend)
			}
		}
	} else {
		m := unsafe.Slice((*float32)(addr), rows*ctx)
		for i := range m {
			m[i] = maskNeg
		}
		for r := 0; r < rows; r++ {
			open := startPos + r + 1
			if open > ctx {
				open = ctx
			}
			row := m[r*ctx : r*ctx+ctx]
			for c := 0; c < open; c++ {
				row[c] = 0.0
			}
		}
	}
	return b.Unlock()
}

func argmaxF32(b litert.TensorBuffer, n int) (int32, error) {
	addr, err := b.Lock(litert.LockRead)
	if err != nil {
		return 0, err
	}
	defer b.Unlock()
	vals := unsafe.Slice((*float32)(addr), n)
	best, idx := vals[0], 0
	for i, v := range vals {
		if v > best {
			best, idx = v, i
		}
	}
	return int32(idx), nil
}

// buildHistory renders the conversation history into a single segment of token
// IDs, prepending the start token and system prompt if set, and preserving turn boundaries.
func buildHistory(tok tokenizer, md litertlm.Metadata, tpl litertlm.PromptTemplates, start string, history []Message) []int32 {
	if len(history) == 0 {
		return nil
	}
	affix := func(role string) litertlm.Affixes {
		switch role {
		case "model":
			return tpl.Model
		case "system":
			return tpl.System
		default:
			return tpl.User
		}
	}
	var sb strings.Builder
	sb.WriteString(start)
	for i, h := range history {
		a := affix(h.Role)
		sb.WriteString(a.Prefix)
		sb.WriteString(h.Text)
		if h.Role == "model" {
			if i < len(history)-1 {
				sb.WriteString(modelBoundary(md, tpl))
			}
		} else {
			sb.WriteString(a.Suffix)
		}
	}
	return append(startIDs(tok, md), tok.Encode(sb.String())...)
}

func (s *Session) ingestHistory(ctx context.Context, history []Message) error {
	ids := buildHistory(s.e.tok, s.e.md, s.tpl, s.start, history)
	if len(ids) == 0 {
		return nil
	}
	if err := prefillTokenRun(ctx, s.e.env, s.e.cm, s.e.pre, s.kv, ids, 0); err != nil {
		return err
	}
	s.pos = len(ids)
	s.started = true
	return nil
}
