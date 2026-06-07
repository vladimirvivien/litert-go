// Package lm is a small LLM runtime over the LiteRT CompiledModel C API. An
// Engine loads a .litertlm (or raw .tflite) model — its tokenizer, metadata,
// and prefill/decode/verify signatures — and generates text by driving the
// static-executor protocol: prefill an N-token prompt, then decode one token at
// a time against a fixed-context KV cache. It handles token-input models
// (gemma3, qwen3, …) and embedding-input models (gemma 3n/4, via separate
// embedder stages), with chat templating, temperature/top-k/top-p sampling, and
// MTP speculative decoding.
package lm

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"unsafe"

	sentencepiece "github.com/eliben/go-sentencepiece"
	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// maskNeg is the "masked" attention value LiteRT-LM uses for f32 (-0.7 * FLT_MAX).
const maskNeg = float32(-0.7 * math.MaxFloat32)

// Engine is a loaded, compiled model ready to generate text. It is not safe for
// concurrent use.
type Engine struct {
	env        litert.Environment
	model      litert.Model
	cm         litert.CompiledModel
	opts       litert.Options
	tok        *sentencepiece.Processor
	md         litertlm.Metadata
	fileBytes  []byte
	pre        prefiller
	decode     sig
	verify     sig
	haveVerify bool
	accel      litert.HwAccelerator
	lastText   string // most recent streamed output (GenerateStream/SendStream)
}

// Open loads libLiteRt from libDir (or the LITERT_LIB environment variable),
// reads the .litertlm or .tflite model at modelPath, extracts its main
// generation graph, tokenizer, and metadata, and compiles it for accel. Close
// releases everything.
func Open(libDir, modelPath string, accel litert.HwAccelerator) (*Engine, error) {
	if err := litert.Load(libDir); err != nil {
		return nil, err
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		return nil, err
	}
	e := &Engine{env: env, accel: accel}
	ok := false
	defer func() {
		if !ok {
			e.Close()
		}
	}()

	if e.fileBytes, err = os.ReadFile(modelPath); err != nil {
		return nil, err
	}
	tflite := e.fileBytes
	if litertlm.IsContainer(e.fileBytes) {
		if tflite, err = litertlm.SectionTFLite(e.fileBytes); err != nil {
			return nil, err
		}
		e.md, _ = litertlm.ReadMetadata(e.fileBytes)
		if sp, serr := litertlm.SectionBytes(e.fileBytes, litertlm.SectionSPTokenizer); serr == nil {
			if e.tok, err = sentencepiece.NewProcessor(bytes.NewReader(sp)); err != nil {
				return nil, fmt.Errorf("load tokenizer: %w", err)
			}
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

	if e.opts, err = litert.NewOptions(accel); err != nil {
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

// Metadata returns the model's parsed LlmMetadata.
func (e *Engine) Metadata() litertlm.Metadata { return e.md }

// HasTokenizer reports whether the model shipped a SentencePiece tokenizer.
func (e *Engine) HasTokenizer() bool { return e.tok != nil }

// HasChatTemplate reports whether the model has chat affixes (for Generate with
// chat=true or NewChat).
func (e *Engine) HasChatTemplate() bool {
	_, ok := e.md.Templates()
	return ok
}

// SupportsSpec reports whether MTP speculative decoding is available (a verify
// signature plus an embedding-input main graph).
func (e *Engine) SupportsSpec() bool {
	return e.haveVerify && sigHasInput(e.decode, "embeddings")
}

// GenOptions configures one generation.
type GenOptions struct {
	MaxTokens int     // cap on generated tokens
	Temp      float32 // sampling temperature; 0 = greedy
	TopK      int     // top-k filter; 0 = off
	TopP      float32 // top-p / nucleus filter; 0 = off
	Seed      int64   // sampling RNG seed
	Spec      bool    // use MTP speculative decoding when available (greedy only)
}

// Generate completes prompt and returns the generated text. When chat is true
// the prompt is wrapped in the model's chat template.
func (e *Engine) Generate(prompt string, chat bool, o GenOptions) (string, error) {
	if e.tok == nil {
		return "", fmt.Errorf("lm: model has no tokenizer; use GenerateIDs")
	}
	ids, err := buildPrompt(e.tok, e.md, prompt, chat)
	if err != nil {
		return "", err
	}
	gen, err := e.generate(ids, o, nil)
	if err != nil {
		return "", err
	}
	return e.tok.Decode(gen), nil
}

// GenerateIDs runs the model on explicit prompt token IDs and returns the
// generated token IDs. For models without a tokenizer.
func (e *Engine) GenerateIDs(prompt []int32, o GenOptions) ([]int, error) {
	return e.generate(prompt, o, nil)
}

// GenerateStream is Generate with incremental output: onPiece is called with
// each newly decoded text fragment as it is produced. It returns the full text.
func (e *Engine) GenerateStream(prompt string, chat bool, o GenOptions, onPiece func(string)) (string, error) {
	if e.tok == nil {
		return "", fmt.Errorf("lm: model has no tokenizer; use GenerateIDs")
	}
	ids, err := buildPrompt(e.tok, e.md, prompt, chat)
	if err != nil {
		return "", err
	}
	_, err = e.generate(ids, o, e.streamer(onPiece))
	if err != nil {
		return "", err
	}
	return e.lastText, nil
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

func (e *Engine) generate(prompt []int32, o GenOptions, onToken func(int32)) ([]int, error) {
	smp := newSampler(o.Temp, o.TopK, o.TopP, o.Seed)
	stop := stopSet(e.tok, e.md)
	switch {
	case o.Spec && e.SupportsSpec():
		return decodeSpeculative(e.env, e.cm, e.fileBytes, e.pre, e.decode, e.verify, prompt, o.MaxTokens, stop, e.accel, onToken)
	case sigHasInput(e.decode, "embeddings"):
		return decodeEmbeddingInput(e.env, e.cm, e.fileBytes, e.pre, e.decode, prompt, o.MaxTokens, stop, e.accel, smp, onToken)
	default:
		return decodeTokenInput(e.env, e.cm, e.pre, e.decode, prompt, o.MaxTokens, stop, smp, onToken)
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
		return nil, fmt.Errorf("lm: model has no tokenizer")
	}
	tpl, ok := e.md.Templates()
	if !ok {
		return nil, fmt.Errorf("lm: model has no chat template (model type %q)", e.md.ModelType)
	}
	return &Chat{e: e, o: o, tpl: tpl}, nil
}

// Close releases the Chat. Chat holds no resources of its own (the Engine owns
// the model); it exists to satisfy Conversation.
func (c *Chat) Close() {}

// Send adds a user message and returns the model's reply, retaining both for
// context on subsequent calls.
func (c *Chat) Send(userText string) (string, error) {
	return c.send(userText, nil)
}

// SendStream is Send with incremental output: onPiece is called with each newly
// decoded fragment of the reply as it is produced.
func (c *Chat) SendStream(userText string, onPiece func(string)) (string, error) {
	return c.send(userText, onPiece)
}

func (c *Chat) send(userText string, onPiece func(string)) (string, error) {
	c.hist = append(c.hist, turn{role: "user", text: userText})
	prompt := buildConversation(c.e.tok, c.e.md, c.tpl, c.hist)
	var reply string
	if onPiece != nil {
		_, err := c.e.generate(prompt, c.o, c.e.streamer(onPiece))
		if err != nil {
			return "", err
		}
		reply = c.e.lastText
	} else {
		gen, err := c.e.generate(prompt, c.o, nil)
		if err != nil {
			return "", err
		}
		reply = c.e.tok.Decode(gen)
	}
	c.hist = append(c.hist, turn{role: "model", text: reply})
	return reply, nil
}

// Conversation is a multi-turn chat session. Both Chat (re-prefills the history
// each turn) and Session (reuses the KV cache across turns) satisfy it.
type Conversation interface {
	Send(userText string) (string, error)
	SendStream(userText string, onPiece func(string)) (string, error)
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
	stop    map[int32]bool
	kv      map[string]litert.TensorBuffer
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
		return nil, fmt.Errorf("lm: model has no tokenizer")
	}
	if sigHasInput(e.decode, "embeddings") {
		return nil, fmt.Errorf("lm: NewSession supports token-input models only")
	}
	tpl, ok := e.md.Templates()
	if !ok {
		return nil, fmt.Errorf("lm: model has no chat template (model type %q)", e.md.ModelType)
	}
	kv, err := allocKV(e.env, e.cm, e.pre.max())
	if err != nil {
		return nil, err
	}
	s := &Session{e: e, o: o, tpl: tpl, stop: stopSet(e.tok, e.md), kv: kv}
	dec, err := newDecoder(e.env, e.cm, e.decode, s.kv, newSampler(o.Temp, o.TopK, o.TopP, o.Seed))
	if err != nil {
		s.Close()
		return nil, err
	}
	s.dec = dec
	return s, nil
}

// Close releases the session's KV cache and decode buffers.
func (s *Session) Close() {
	if s.dec != nil {
		s.dec.close()
	}
	for _, b := range s.kv {
		b.Close()
	}
}

// Send adds a user message and returns the reply.
func (s *Session) Send(userText string) (string, error) { return s.send(userText, nil) }

// SendStream is Send with incremental output.
func (s *Session) SendStream(userText string, onPiece func(string)) (string, error) {
	return s.send(userText, onPiece)
}

func (s *Session) send(userText string, onPiece func(string)) (string, error) {
	// Render the new turn. The first turn carries the start token; later turns
	// first close the previous model turn with its suffix.
	render := s.tpl.User.Prefix + userText + s.tpl.User.Suffix + s.tpl.Model.Prefix
	var ids []int32
	if !s.started {
		ids = startIDs(s.e.tok, s.e.md)
		s.started = true
	} else {
		render = s.tpl.Model.Suffix + render
	}
	for _, t := range s.e.tok.Encode(render) {
		ids = append(ids, int32(t.ID))
	}
	if len(ids) == 0 {
		return "", fmt.Errorf("empty turn")
	}

	// Ingest the turn tokens; the last one's logits start the reply.
	pos := s.pos
	for _, id := range ids {
		if err := s.dec.feed(id, pos); err != nil {
			return "", err
		}
		pos++
	}
	id, err := s.dec.sample()
	if err != nil {
		return "", err
	}

	var stream func(int32)
	if onPiece != nil {
		stream = s.e.streamer(onPiece)
	}
	var gen []int
	for g := 0; g < s.o.MaxTokens; g++ {
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
	s.pos = pos
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
	for _, name := range []string{"tokens", "embeddings"} {
		if sh, err := inputShape(g, name); err == nil && len(sh) >= 2 {
			return int(sh[1]), nil
		}
	}
	return 0, fmt.Errorf("prefill signature has no tokens/embeddings input")
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

// prefillTokenRun ingests ids into the KV cache starting at position start,
// chunking across prefill buckets for prompts longer than one bucket.
func prefillTokenRun(env litert.Environment, cm litert.CompiledModel, pre prefiller, kv map[string]litert.TensorBuffer, ids []int32, start int) error {
	off := 0
	for _, c := range pre.plan(len(ids)) {
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
func buildPrompt(tok *sentencepiece.Processor, md litertlm.Metadata, text string, chat bool) ([]int32, error) {
	if chat {
		return buildChatPrompt(tok, md, text)
	}
	ids := startIDs(tok, md)
	for _, t := range tok.Encode(text) {
		ids = append(ids, int32(t.ID))
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("empty tokenization of %q", text)
	}
	return ids, nil
}

// buildChatPrompt wraps userText in the model's single user turn: start ++
// user.prefix ++ userText ++ user.suffix ++ model.prefix. The turn markers in
// the affixes (e.g. <start_of_turn>) are user-defined tokenizer pieces, so
// encoding the rendered string yields their single vocab IDs.
func buildChatPrompt(tok *sentencepiece.Processor, md litertlm.Metadata, userText string) ([]int32, error) {
	tpl, ok := md.Templates()
	if !ok {
		return nil, fmt.Errorf("no chat template for model type %q", md.ModelType)
	}
	ids := startIDs(tok, md)
	rendered := tpl.User.Prefix + userText + tpl.User.Suffix + tpl.Model.Prefix
	for _, t := range tok.Encode(rendered) {
		ids = append(ids, int32(t.ID))
	}
	if len(ids) < 2 {
		return nil, fmt.Errorf("empty chat tokenization of %q", userText)
	}
	return ids, nil
}

// buildConversation renders a chat history into prompt token IDs: the start
// token, then each turn wrapped in its role affixes, then the model prefix to
// open the assistant's reply. The whole conversation is encoded as one string
// so control-token boundaries tokenize correctly.
func buildConversation(tok *sentencepiece.Processor, md litertlm.Metadata, tpl litertlm.PromptTemplates, hist []turn) []int32 {
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
		sb.WriteString(a.Suffix)
	}
	sb.WriteString(tpl.Model.Prefix)

	ids := startIDs(tok, md)
	for _, t := range tok.Encode(sb.String()) {
		ids = append(ids, int32(t.ID))
	}
	return ids
}

// startIDs returns the start token IDs to prepend: the metadata start_token when
// given as IDs, none for the "None" sentinel, otherwise the tokenizer's BOS when
// it has one.
func startIDs(tok *sentencepiece.Processor, md litertlm.Metadata) []int32 {
	if len(md.StartToken.IDs) > 0 {
		return append([]int32(nil), md.StartToken.IDs...)
	}
	if md.StartToken.Str == "None" {
		return nil
	}
	if bos := tok.ModelInfo().BeginningOfSentenceID; bos >= 0 {
		return []int32{int32(bos)}
	}
	return nil
}

// stopSet collects the token IDs that end generation: the metadata stop_tokens
// (IDs directly, single-token strings resolved through the tokenizer) plus the
// tokenizer's end-of-sentence token. Multi-token stop strings are skipped.
func stopSet(tok *sentencepiece.Processor, md litertlm.Metadata) map[int32]bool {
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
				set[int32(enc[0].ID)] = true
			}
		}
	}
	if eos := tok.ModelInfo().EndOfSentenceID; eos >= 0 {
		set[int32(eos)] = true
	}
	return set
}

// --- token-input decode ---

// decodeTokenInput runs the static token-input pipeline (gemma3, qwen3, …):
// allocate the KV cache once, prefill all but the last prompt token, then
// decode from the held-back token.
func decodeTokenInput(env litert.Environment, cm litert.CompiledModel, pre prefiller, decode sig, prompt []int32, ngen int, stop map[int32]bool, smp *sampler, onToken func(int32)) ([]int, error) {
	kv, err := allocKV(env, cm, pre.max())
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, b := range kv {
			b.Close()
		}
	}()

	p := len(prompt) - 1
	if err := prefillTokenRun(env, cm, pre, kv, prompt[:p], 0); err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}

	dec, err := newDecoder(env, cm, decode, kv, smp)
	if err != nil {
		return nil, fmt.Errorf("decode setup: %w", err)
	}
	defer dec.close()

	next := prompt[p]
	pos := p
	var gen []int
	for g := 0; g < ngen; g++ {
		id, err := dec.step(next, pos)
		if err != nil {
			return nil, fmt.Errorf("decode step %d: %w", g, err)
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
	return gen, nil
}

// prefillStep ingests ids into the KV cache at position start through bucket g:
// the tokens fill the first len(ids) slots, input_pos runs [start, start+seq),
// and a causal mask lets row r attend [0, start+r+1) so earlier chunks stay
// visible. Bucket slots past len(ids) hold padding whose KV is later overwritten
// or never attended.
func prefillStep(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer, ids []int32, start int) error {
	tokens, err := allocReqInput(env, cm, g, "tokens")
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
	if err := fillCausalMask(mask, seq, ctx, start); err != nil {
		return err
	}

	perCall := map[string]litert.TensorBuffer{"tokens": tokens, "input_pos": pos, "mask": mask}
	in := assemble(g.inNames, perCall, kv)
	out := assemble(g.outNames, nil, kv) // prefill outputs are all KV
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
	return cm.Run(g.idx, in, out)
}

// decoder holds the fixed decode buffer set and a litert.Runner whose Run
// arguments are pinned once. tokens/input_pos/mask are rewritten each step; the
// KV buffers persist across steps (single-bank cache) and logits is overwritten
// by each Run.
type decoder struct {
	tokens litert.TensorBuffer
	posBuf litert.TensorBuffer
	mask   litert.TensorBuffer
	logits litert.TensorBuffer
	runner *litert.Runner
	ctx    int
	vocab  int
	smp    *sampler
}

func newDecoder(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer, smp *sampler) (*decoder, error) {
	d := &decoder{smp: smp}
	var err error
	if d.tokens, err = allocReqInput(env, cm, g, "tokens"); err != nil {
		return nil, err
	}
	if d.posBuf, err = allocReqInput(env, cm, g, "input_pos"); err != nil {
		return nil, err
	}
	if d.mask, err = allocReqInput(env, cm, g, "mask"); err != nil {
		return nil, err
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

	perCall := map[string]litert.TensorBuffer{"tokens": d.tokens, "input_pos": d.posBuf, "mask": d.mask}
	in := assemble(g.inNames, perCall, kv)
	out := assemble(g.outNames, map[string]litert.TensorBuffer{"logits": d.logits}, kv)
	for i, b := range in {
		if b == 0 {
			return nil, fmt.Errorf("unmapped input[%d] %q", i, g.inNames[i])
		}
	}
	for i, b := range out {
		if b == 0 {
			return nil, fmt.Errorf("unmapped output[%d] %q", i, g.outNames[i])
		}
	}
	d.runner = litert.NewRunner(cm, g.idx, in, out)
	return d, nil
}

// feed writes one token at pos and runs the model, leaving logits in d.logits.
// It does not sample — used to ingest known tokens into the KV cache.
func (d *decoder) feed(token int32, pos int) error {
	if err := writeInts(d.tokens, []int32{token}); err != nil {
		return err
	}
	if err := writeInts(d.posBuf, []int32{int32(pos)}); err != nil {
		return err
	}
	if err := fillCausalMask(d.mask, 1, d.ctx, pos); err != nil {
		return err
	}
	return d.runner.Run()
}

// sample picks the next token from the logits of the most recent feed.
func (d *decoder) sample() (int32, error) { return d.smp.sample(d.logits, d.vocab) }

func (d *decoder) step(token int32, pos int) (int32, error) {
	if err := d.feed(token, pos); err != nil {
		return 0, err
	}
	return d.sample()
}

func (d *decoder) close() {
	d.runner.Close()
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

func newZeroedSized(env litert.Environment, bt litert.BufferType, tt litert.TensorType, size uint64) (litert.TensorBuffer, error) {
	buf, err := litert.NewManagedBuffer(env, bt, tt, size)
	if err != nil {
		return 0, err
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

// fillCausalMask fills a [1,1,rows,ctx] f32 mask: row r (position startPos+r)
// attends columns [0, startPos+r+1), the rest are masked.
func fillCausalMask(b litert.TensorBuffer, rows, ctx, startPos int) error {
	addr, err := b.Lock(litert.LockWrite)
	if err != nil {
		return err
	}
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
			row[c] = 0
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
