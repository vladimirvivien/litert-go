package lm

import (
	"context"
	"fmt"
	"time"
	"unsafe"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// Embedding-input pipeline (gemma 3n/4). The main graph consumes embeddings,
// not token IDs: each token is run through a text embedder (token -> embeddings
// f32[..,H]) and, when the graph asks for it, a per-layer embedder (token ->
// per_layer_embeddings f32[..,L,D]). Those feed prefill/decode along with
// input_pos, a boolean attention mask (true = attend), the i8 KV cache, and — on gemma 4 — a param_tensor ([start, start+len, start+len])
// driving the KV-cache write index. Semantics follow LiteRT-LM's
// llm_litert_compiled_model_executor. gemma 3n omits param_tensor and uses
// input_pos for the cache index.

// embedModel wraps one compiled embedder: it maps a single token ID to its
// embedding vector, reusing pinned buffers across calls.
type embedModel struct {
	model  litert.Model
	cm     litert.CompiledModel
	in     litert.TensorBuffer
	out    litert.TensorBuffer
	runner *litert.Runner
	floats int
}

func newEmbedModel(env litert.Environment, opts litert.Options, section []byte) (*embedModel, error) {
	model, err := litert.OpenModelFromBuffer(env, section)
	if err != nil {
		return nil, err
	}
	cm, err := litert.Compile(env, model, opts)
	if err != nil {
		model.Close()
		return nil, err
	}
	g, err := loadSig(model, 0)
	if err != nil {
		return nil, err
	}
	in, err := allocReqInput(env, cm, g, "token_ids")
	if err != nil {
		return nil, err
	}
	out, err := allocReqOutput(env, cm, g, "embeddings")
	if err != nil {
		return nil, err
	}
	ot, err := g.s.OutputType("embeddings")
	if err != nil {
		return nil, err
	}
	floats := 1
	for _, d := range ot.Shape {
		floats *= int(d)
	}
	return &embedModel{
		model:  model,
		cm:     cm,
		in:     in,
		out:    out,
		runner: litert.NewRunner(cm, g.idx, []litert.TensorBuffer{in}, []litert.TensorBuffer{out}),
		floats: floats,
	}, nil
}

// embed returns the embedding vector for a single token.
func (e *embedModel) embed(token int32) ([]float32, error) {
	if err := writeInts(e.in, []int32{token}); err != nil {
		return nil, err
	}
	if err := e.runner.Run(); err != nil {
		return nil, err
	}
	addr, err := e.out.Lock(litert.LockRead)
	if err != nil {
		return nil, err
	}
	defer e.out.Unlock()
	out := make([]float32, e.floats)
	copy(out, unsafe.Slice((*float32)(addr), e.floats))
	return out, nil
}

func (e *embedModel) close() {
	e.runner.Close()
	e.in.Close()
	e.out.Close()
	e.cm.Close()
	e.model.Close()
}

// decodeEmbeddingInput runs the gemma 3n/4 pipeline: allocate the double-banked
// i8 KV cache, prefill all but the last prompt token through the shared embedder
// stages, then greedily decode from the held-back token.
func decodeEmbeddingInput(ctx context.Context, env litert.Environment, cm litert.CompiledModel, pre prefiller, decode sig, emb, ple *embedModel, prompt []int32, ngen int, stop map[int32]bool, smp *sampler, singleKV bool, metrics func(DecodeStats), onToken func(int32)) ([]int, error) {
	kv, err := allocKVBanks(env, cm, pre.max(), singleKV)
	if err != nil {
		return nil, err
	}
	defer kv.close()

	p := len(prompt) - 1
	if err := prefillEmbedRun(ctx, env, cm, pre, kv, emb, ple, prompt[:p], 0); err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}

	dec, err := newEmbedDecoder(env, cm, decode, kv, emb, ple, smp)
	if err != nil {
		return nil, fmt.Errorf("decode setup: %w", err)
	}
	defer dec.close()

	next := prompt[p]
	pos := p
	var gen []int
	t0 := time.Now()
	for g := 0; g < ngen; g++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
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
	if metrics != nil {
		metrics(DecodeStats{Tokens: len(gen), Decode: time.Since(t0)})
	}
	return gen, nil
}

// embedTokens runs the embedder(s) over ids, returning the flattened text
// embeddings and (when ple is non-nil) per-layer embeddings.
func embedTokens(emb, ple *embedModel, ids []int32) (text, perLayer []float32, err error) {
	text = make([]float32, 0, len(ids)*emb.floats)
	if ple != nil {
		perLayer = make([]float32, 0, len(ids)*ple.floats)
	}
	for _, t := range ids {
		v, err := emb.embed(t)
		if err != nil {
			return nil, nil, err
		}
		text = append(text, v...)
		if ple != nil {
			w, err := ple.embed(t)
			if err != nil {
				return nil, nil, err
			}
			perLayer = append(perLayer, w...)
		}
	}
	return text, perLayer, nil
}

// allocNonKV allocates a buffer for every input (or output, when out) tensor of
// g that is not a KV-cache tensor, keyed by name. KV tensors are shared through
// the kv map and assembled separately.
func allocNonKV(env litert.Environment, cm litert.CompiledModel, g sig, out bool) (map[string]litert.TensorBuffer, error) {
	names := g.inNames
	alloc := func(name string) (litert.TensorBuffer, error) { return allocReqInput(env, cm, g, name) }
	if out {
		names = g.outNames
		alloc = func(name string) (litert.TensorBuffer, error) { return allocReqOutput(env, cm, g, name) }
	}
	bufs := map[string]litert.TensorBuffer{}
	for _, name := range names {
		if isKV(name) {
			continue
		}
		b, err := alloc(name)
		if err != nil {
			for _, x := range bufs {
				x.Close()
			}
			return nil, err
		}
		bufs[name] = b
	}
	return bufs, nil
}

func closeBufs(bufs map[string]litert.TensorBuffer) {
	for _, b := range bufs {
		b.Close()
	}
}

// prefillEmbed batch-ingests ids into the KV cache at position start: it embeds
// the tokens, writes input_pos [start, start+len), a causal mask whose row i
// attends [0, start+i+1) (so earlier cached turns stay visible), and a
// param_tensor that begins the KV write at start. With start 0 it is a fresh
// prefill; with start > 0 it appends a turn to an existing cache.
func prefillEmbed(env litert.Environment, cm litert.CompiledModel, g sig, kv *kvBanks, emb, ple *embedModel, ids []int32, start int) error {
	text, perLayer, err := embedTokens(emb, ple, ids)
	if err != nil {
		return err
	}
	return prefillEmbedFill(env, cm, g, kv, text, perLayer, len(ids), start)
}

// prefillEmbedFill ingests pre-computed text (and per-layer) embeddings for n
// tokens into the KV cache at position start through bucket g. Used directly by
// the multimodal path, which splices image embeddings into text before prefill.
func prefillEmbedFill(env litert.Environment, cm litert.CompiledModel, g sig, kv *kvBanks, text, perLayer []float32, n, start int) error {
	embShape, _ := inputShape(g, "embeddings") // [1, seq, H]
	seq := int(embShape[1])
	if n > seq {
		return fmt.Errorf("prompt (%d) exceeds prefill bucket (%d)", n, seq)
	}

	bufs, err := allocNonKV(env, cm, g, false)
	if err != nil {
		return err
	}
	defer closeBufs(bufs)

	maskShape, _ := inputShape(g, "mask") // [1,1,seq,ctx]
	rows, ctx := int(maskShape[2]), int(maskShape[3])
	pos := make([]int32, seq)
	for i := range pos {
		pos[i] = int32(start + i)
	}

	for name, b := range bufs {
		if err := fillEmbedInput(b, name, text, perLayer, pos, rows, ctx, n, start); err != nil {
			return err
		}
	}

	in := assemble(g.inNames, bufs, kv.inBind())
	out := assemble(g.outNames, nil, kv.out())
	for i, b := range in {
		if b == 0 && !kv.allowNullIn(g.inNames[i]) {
			return fmt.Errorf("unmapped input[%d] %q", i, g.inNames[i])
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

// prefillEmbedRun ingests ids into the KV cache starting at position start,
// chunking across prefill buckets for prompts longer than one bucket.
func prefillEmbedRun(ctx context.Context, env litert.Environment, cm litert.CompiledModel, pre prefiller, kv *kvBanks, emb, ple *embedModel, ids []int32, start int) error {
	off := 0
	for _, c := range pre.plan(len(ids)) {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := prefillEmbed(env, cm, pre.sig[c.bucket], kv, emb, ple, ids[off:off+c.take], start+off); err != nil {
			return err
		}
		off += c.take
	}
	return nil
}

// prefillEmbedDataRun ingests pre-computed text (and per-layer) embeddings for n
// tokens into the KV cache at start, chunking across prefill buckets. text is
// [n*H], perLayer is [n*L] (or empty).
func prefillEmbedDataRun(ctx context.Context, env litert.Environment, cm litert.CompiledModel, pre prefiller, kv *kvBanks, text, perLayer []float32, n, start int) error {
	if n == 0 {
		return nil
	}
	h := len(text) / n
	l := 0
	if len(perLayer) > 0 {
		l = len(perLayer) / n
	}
	off := 0
	for _, c := range pre.plan(n) {
		if err := ctx.Err(); err != nil {
			return err
		}
		t := text[off*h : (off+c.take)*h]
		var pl []float32
		if l > 0 {
			pl = perLayer[off*l : (off+c.take)*l]
		}
		if err := prefillEmbedFill(env, cm, pre.sig[c.bucket], kv, t, pl, c.take, start+off); err != nil {
			return err
		}
		off += c.take
	}
	return nil
}

// fillEmbedInput writes one non-KV prefill/decode input by name. steps rows of
// the mask are filled starting at startPos; for a single decode step rows=1 and
// startPos is the current position.
func fillEmbedInput(b litert.TensorBuffer, name string, text, perLayer []float32, pos []int32, rows, ctx, steps, startPos int) error {
	switch name {
	case "embeddings":
		return writeFloats(b, text)
	case "per_layer_embeddings":
		return writeFloats(b, perLayer)
	case "input_pos":
		return writeInts(b, pos)
	case "mask":
		return fillBoolMask(b, rows, ctx, steps, startPos)
	case "param_tensor":
		end := int32(startPos + steps)
		return writeInts(b, []int32{int32(startPos), end, end})
	default:
		return fmt.Errorf("unsupported embedding-input tensor %q", name)
	}
}

// embedDecoder holds the fixed decode buffer set for the embedding-input path
// and a pair of litert.Runners whose arguments are pinned once. The non-KV
// buffers are shared; the two runners differ only in which KV bank they read
// vs write (runA reads bank a and writes bank b; runB the reverse). Each step
// uses the runner whose input is the cache's current bank, then the banks
// swap.
type embedDecoder struct {
	emb, ple   *embedModel
	in         map[string]litert.TensorBuffer
	out        map[string]litert.TensorBuffer
	runA, runB *litert.Runner
	kv         *kvBanks
	ctx        int
	vocab      int
	actLen     int     // elements in the activations output (0 if absent)
	pos        []int32 // scratch: single-element input_pos
	smp        *sampler
	pending    bool // an async run was submitted and not yet awaited
}

func newEmbedDecoder(env litert.Environment, cm litert.CompiledModel, g sig, kv *kvBanks, emb, ple *embedModel, smp *sampler) (*embedDecoder, error) {
	in, err := allocNonKV(env, cm, g, false)
	if err != nil {
		return nil, err
	}
	out, err := allocNonKV(env, cm, g, true)
	if err != nil {
		closeBufs(in)
		return nil, err
	}
	d := &embedDecoder{emb: emb, ple: ple, in: in, out: out, kv: kv, pos: make([]int32, 1), smp: smp}
	fail := func(err error) (*embedDecoder, error) {
		closeBufs(in)
		closeBufs(out)
		return nil, err
	}

	maskShape, _ := inputShape(g, "mask") // [1,1,1,ctx]
	d.ctx = int(maskShape[3])
	lt, err := g.s.OutputType("logits")
	if err != nil {
		return fail(err)
	}
	d.vocab = 1
	for _, dim := range lt.Shape {
		d.vocab *= int(dim)
	}
	if _, ok := g.outByName["activations"]; ok {
		if at, err := g.s.OutputType("activations"); err == nil {
			d.actLen = 1
			for _, dim := range at.Shape {
				d.actLen *= int(dim)
			}
		}
	}

	runner := func(inBank, outBank map[string]litert.TensorBuffer) (*litert.Runner, error) {
		inArr := assemble(g.inNames, in, inBank)
		outArr := assemble(g.outNames, out, outBank)
		for i, b := range inArr {
			if b == 0 && !kv.allowNullIn(g.inNames[i]) {
				return nil, fmt.Errorf("unmapped input[%d] %q", i, g.inNames[i])
			}
		}
		return litert.NewRunner(cm, g.idx, inArr, outArr), nil
	}
	inA, inB := kv.a, kv.b
	if kv.single {
		inA, inB = nil, nil
	}
	if d.runA, err = runner(inA, kv.b); err != nil {
		return fail(err)
	}
	if d.runB, err = runner(inB, kv.a); err != nil {
		return fail(err)
	}
	return d, nil
}

// feed embeds one token, writes it into the KV cache at pos through the decode
// graph, and leaves logits in d.out["logits"]. It does not sample. The run is
// submitted asynchronously when the backend supports it; sampling (or the next
// feed) waits for completion through the logits buffer's event.
func (d *embedDecoder) feed(token int32, pos int) error {
	if d.pending {
		if err := d.out["logits"].Wait(); err != nil {
			return err
		}
		d.pending = false
	}
	text, perLayer, err := embedTokens(d.emb, d.ple, []int32{token})
	if err != nil {
		return err
	}
	d.pos[0] = int32(pos)
	for name, b := range d.in {
		if err := fillEmbedInput(b, name, text, perLayer, d.pos, 1, d.ctx, 1, pos); err != nil {
			return err
		}
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
func (d *embedDecoder) sample() (int32, error) {
	d.pending = false
	return d.smp.sample(d.out["logits"], d.vocab)
}

func (d *embedDecoder) step(token int32, pos int) (int32, error) {
	if err := d.feed(token, pos); err != nil {
		return 0, err
	}
	return d.sample()
}

// stepAct is step plus the decode activations (the hidden state the MTP drafter
// consumes).
func (d *embedDecoder) stepAct(token int32, pos int) (int32, []float32, error) {
	id, err := d.step(token, pos)
	if err != nil {
		return 0, nil, err
	}
	act, err := readFloats(d.out["activations"], d.actLen)
	if err != nil {
		return 0, nil, err
	}
	return id, act, nil
}

func (d *embedDecoder) close() {
	d.runA.Close()
	d.runB.Close()
	closeBufs(d.in)
	closeBufs(d.out)
}

// embedSession is a KV-reuse chat session for embedding-input models (gemma
// 3n/4). Each turn batch-ingests its new tokens into the shared KV cache via a
// prefill-at-offset — the same prefill graph a fresh Generate uses, so the
// cached i8 KV matches — then decodes the reply one token at a time. The
// embedder stages are the Engine's (compiled once); the session owns only its KV
// cache and decode buffers. It satisfies Conversation; NewConversation returns
// it for embedding-input models.
type embedSession struct {
	e       *Engine
	o       GenOptions
	tpl     litertlm.PromptTemplates
	start   string // conversation-start render: system prompt + tool declarations
	stop    map[int32]bool
	kv      *kvBanks
	dec     *embedDecoder
	pos     int
	started bool
}

func (e *Engine) newEmbedSession(o GenOptions) (*embedSession, error) {
	tpl, ok := e.templates()
	if !ok {
		return nil, fmt.Errorf("%w (model type %q)", ErrNoChatTemplate, e.md.ModelType)
	}
	start, err := conversationStart(e, tpl, o)
	if err != nil {
		return nil, err
	}
	emb, ple, err := e.ensureEmbedders()
	if err != nil {
		return nil, err
	}
	s := &embedSession{e: e, o: o, tpl: tpl, start: start, stop: stopSet(e.tok, e.md)}
	done := false
	defer func() {
		if !done {
			s.Close()
		}
	}()

	if s.kv, err = allocKVBanks(e.env, e.cm, e.pre.max(), e.singleKV()); err != nil {
		return nil, err
	}
	if s.dec, err = newEmbedDecoder(e.env, e.cm, e.decode, s.kv, emb, ple, newSampler(o.Temp, o.TopK, o.TopP, o.Seed)); err != nil {
		return nil, err
	}
	if len(o.History) > 0 {
		if err := s.ingestHistory(context.Background(), o.History); err != nil {
			s.Close()
			return nil, err
		}
	}
	done = true
	return s, nil
}

// Close releases the session's KV cache and decode buffers. The embedder stages
// belong to the Engine and outlive the session.
func (s *embedSession) Close() {
	if s.dec != nil {
		s.dec.close()
	}
	if s.kv != nil {
		s.kv.close()
	}
}

func (s *embedSession) ingestHistory(ctx context.Context, history []Message) error {
	ids := buildHistory(s.e.tok, s.e.md, s.tpl, s.start, history)
	if len(ids) == 0 {
		return nil
	}
	if err := prefillEmbedRun(ctx, s.e.env, s.e.cm, s.e.pre, s.kv, s.e.emb, s.e.ple, ids, 0); err != nil {
		return fmt.Errorf("prefill: %w", err)
	}
	s.pos = len(ids)
	s.started = true
	return nil
}

// TokenCount returns the number of tokens currently stored in the session's KV cache.
func (s *embedSession) TokenCount() int {
	return s.pos
}

// Send adds a user message and returns the reply.
func (s *embedSession) Send(ctx context.Context, userText string) (string, error) {
	return s.send(ctx, userText, nil)
}

// SendStream is Send with incremental output.
func (s *embedSession) SendStream(ctx context.Context, userText string, onPiece func(string)) (string, error) {
	return s.send(ctx, userText, onPiece)
}

func (s *embedSession) send(ctx context.Context, userText string, onPiece func(string)) (string, error) {
	return s.sendTurn(ctx, s.tpl.User.Prefix+userText+s.tpl.User.Suffix, onPiece)
}

// SendToolResults delivers function results to the model and decodes
// its follow-up turn. The conversation must target a tool-capable
// family.
func (s *embedSession) SendToolResults(ctx context.Context, results []ToolResult) (string, error) {
	return s.SendToolResultsStream(ctx, results, nil)
}

// SendToolResultsStream is SendToolResults with incremental output.
func (s *embedSession) SendToolResultsStream(ctx context.Context, results []ToolResult, onPiece func(string)) (string, error) {
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
func (s *embedSession) sendTurn(ctx context.Context, turn string, onPiece func(string)) (string, error) {
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

	// Batch-ingest all but the last turn token via prefill-at-offset; the decode
	// graph then feeds the held-back token, whose logits start the reply.
	p := len(ids) - 1
	if p > 0 {
		if err := prefillEmbedRun(ctx, s.e.env, s.e.cm, s.e.pre, s.kv, s.e.emb, s.e.ple, ids[:p], s.pos); err != nil {
			return "", fmt.Errorf("prefill: %w", err)
		}
	}
	pos := s.pos + p
	if err := s.dec.feed(ids[p], pos); err != nil {
		return "", err
	}
	pos++
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
	s.pos = pos
	return s.e.tok.Decode(gen), nil
}

// readFloats copies the first n elements of an f32 buffer out to Go memory.
func readFloats(b litert.TensorBuffer, n int) ([]float32, error) {
	addr, err := b.Lock(litert.LockRead)
	if err != nil {
		return nil, err
	}
	defer b.Unlock()
	out := make([]float32, n)
	copy(out, unsafe.Slice((*float32)(addr), n))
	return out, nil
}

// argmaxRows returns the argmax of each of rows consecutive vocab-sized logit
// rows in b (logits f32[1, rows, vocab]).
func argmaxRows(b litert.TensorBuffer, rows, vocab int) ([]int32, error) {
	addr, err := b.Lock(litert.LockRead)
	if err != nil {
		return nil, err
	}
	defer b.Unlock()
	all := unsafe.Slice((*float32)(addr), rows*vocab)
	out := make([]int32, rows)
	for r := 0; r < rows; r++ {
		row := all[r*vocab : r*vocab+vocab]
		best, idx := row[0], 0
		for i, v := range row {
			if v > best {
				best, idx = v, i
			}
		}
		out[r] = int32(idx)
	}
	return out, nil
}

// writeFloats copies vals into the first len(vals) elements of an f32 buffer. A
// nil/empty slice (e.g. absent per-layer embeddings) leaves the zeroed buffer.
func writeFloats(b litert.TensorBuffer, vals []float32) error {
	if len(vals) == 0 {
		return nil
	}
	addr, err := b.Lock(litert.LockWrite)
	if err != nil {
		return err
	}
	copy(unsafe.Slice((*float32)(addr), len(vals)), vals)
	return b.Unlock()
}

// fillBoolMask writes a [1,1,rows,ctx] boolean attention mask: row i (position
// startPos+i) attends columns [0, startPos+i+1) with true, the rest false. Only
// the first steps rows are filled; remaining rows stay false.
func fillBoolMask(b litert.TensorBuffer, rows, ctx, steps, startPos int) error {
	addr, err := b.Lock(litert.LockWrite)
	if err != nil {
		return err
	}
	m := unsafe.Slice((*bool)(addr), rows*ctx)
	clear(m)
	for i := 0; i < steps && i < rows; i++ {
		open := startPos + i + 1
		if open > ctx {
			open = ctx
		}
		row := m[i*ctx:]
		for c := 0; c < open; c++ {
			row[c] = true
		}
	}
	return b.Unlock()
}
