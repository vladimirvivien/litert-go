package lm

import (
	"fmt"
	"unsafe"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// Embedding-input pipeline (gemma 3n/4). The main graph consumes embeddings,
// not token IDs: each token is run through a text embedder (token -> embeddings
// f32[..,H]) and, when the graph asks for it, a per-layer embedder (token ->
// per_layer_embeddings f32[..,L,D]). Those feed prefill/decode along with
// input_pos, a boolean attention mask (true = attend), the single-bank i8 KV
// cache, and — on gemma 4 — a param_tensor ([start, start+len, start+len])
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

// decodeEmbeddingInput runs the gemma 3n/4 pipeline: allocate the i8 KV cache,
// prefill all but the last prompt token through the shared embedder stages, then
// greedily decode from the held-back token.
func decodeEmbeddingInput(env litert.Environment, cm litert.CompiledModel, pre prefiller, decode sig, emb, ple *embedModel, prompt []int32, ngen int, stop map[int32]bool, smp *sampler, onToken func(int32)) ([]int, error) {
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
	if err := prefillEmbedRun(env, cm, pre, kv, emb, ple, prompt[:p], 0); err != nil {
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
func prefillEmbed(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer, emb, ple *embedModel, ids []int32, start int) error {
	embShape, _ := inputShape(g, "embeddings") // [1, seq, H]
	seq := int(embShape[1])
	if len(ids) > seq {
		return fmt.Errorf("prompt (%d) exceeds prefill bucket (%d)", len(ids), seq)
	}

	bufs, err := allocNonKV(env, cm, g, false)
	if err != nil {
		return err
	}
	defer closeBufs(bufs)

	text, perLayer, err := embedTokens(emb, ple, ids)
	if err != nil {
		return err
	}
	maskShape, _ := inputShape(g, "mask") // [1,1,seq,ctx]
	rows, ctx := int(maskShape[2]), int(maskShape[3])
	pos := make([]int32, seq)
	for i := range pos {
		pos[i] = int32(start + i)
	}
	n := len(ids)

	for name, b := range bufs {
		if err := fillEmbedInput(b, name, text, perLayer, pos, rows, ctx, n, start); err != nil {
			return err
		}
	}

	in := assemble(g.inNames, bufs, kv)
	out := assemble(g.outNames, nil, kv)
	for i, b := range in {
		if b == 0 {
			return fmt.Errorf("unmapped input[%d] %q", i, g.inNames[i])
		}
	}
	return cm.Run(g.idx, in, out)
}

// prefillEmbedRun ingests ids into the KV cache starting at position start,
// chunking across prefill buckets for prompts longer than one bucket.
func prefillEmbedRun(env litert.Environment, cm litert.CompiledModel, pre prefiller, kv map[string]litert.TensorBuffer, emb, ple *embedModel, ids []int32, start int) error {
	off := 0
	for _, c := range pre.plan(len(ids)) {
		if err := prefillEmbed(env, cm, pre.sig[c.bucket], kv, emb, ple, ids[off:off+c.take], start+off); err != nil {
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
// and a Runner whose arguments are pinned once.
type embedDecoder struct {
	emb, ple *embedModel
	in       map[string]litert.TensorBuffer
	out      map[string]litert.TensorBuffer
	runner   *litert.Runner
	ctx      int
	vocab    int
	actLen   int     // elements in the activations output (0 if absent)
	pos      []int32 // scratch: single-element input_pos
	smp      *sampler
}

func newEmbedDecoder(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer, emb, ple *embedModel, smp *sampler) (*embedDecoder, error) {
	in, err := allocNonKV(env, cm, g, false)
	if err != nil {
		return nil, err
	}
	out, err := allocNonKV(env, cm, g, true)
	if err != nil {
		closeBufs(in)
		return nil, err
	}
	d := &embedDecoder{emb: emb, ple: ple, in: in, out: out, pos: make([]int32, 1), smp: smp}

	maskShape, _ := inputShape(g, "mask") // [1,1,1,ctx]
	d.ctx = int(maskShape[3])
	lt, err := g.s.OutputType("logits")
	if err != nil {
		closeBufs(in)
		closeBufs(out)
		return nil, err
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

	inArr := assemble(g.inNames, in, kv)
	outArr := assemble(g.outNames, out, kv)
	for i, b := range inArr {
		if b == 0 {
			closeBufs(in)
			closeBufs(out)
			return nil, fmt.Errorf("unmapped input[%d] %q", i, g.inNames[i])
		}
	}
	d.runner = litert.NewRunner(cm, g.idx, inArr, outArr)
	return d, nil
}

// feed embeds one token, writes it into the KV cache at pos through the decode
// graph, and leaves logits in d.out["logits"]. It does not sample.
func (d *embedDecoder) feed(token int32, pos int) error {
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
	return d.runner.Run()
}

// sample picks the next token from the logits of the most recent feed.
func (d *embedDecoder) sample() (int32, error) {
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
	d.runner.Close()
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
	stop    map[int32]bool
	kv      map[string]litert.TensorBuffer
	dec     *embedDecoder
	pos     int
	started bool
}

func (e *Engine) newEmbedSession(o GenOptions) (*embedSession, error) {
	tpl, ok := e.md.Templates()
	if !ok {
		return nil, fmt.Errorf("lm: model has no chat template (model type %q)", e.md.ModelType)
	}
	emb, ple, err := e.ensureEmbedders()
	if err != nil {
		return nil, err
	}
	s := &embedSession{e: e, o: o, tpl: tpl, stop: stopSet(e.tok, e.md)}
	done := false
	defer func() {
		if !done {
			s.Close()
		}
	}()

	if s.kv, err = allocKV(e.env, e.cm, e.pre.max()); err != nil {
		return nil, err
	}
	if s.dec, err = newEmbedDecoder(e.env, e.cm, e.decode, s.kv, emb, ple, newSampler(o.Temp, o.TopK, o.TopP, o.Seed)); err != nil {
		return nil, err
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
	for _, b := range s.kv {
		b.Close()
	}
}

// Send adds a user message and returns the reply.
func (s *embedSession) Send(userText string) (string, error) { return s.send(userText, nil) }

// SendStream is Send with incremental output.
func (s *embedSession) SendStream(userText string, onPiece func(string)) (string, error) {
	return s.send(userText, onPiece)
}

func (s *embedSession) send(userText string, onPiece func(string)) (string, error) {
	// Render the new turn. The first turn carries the start token; later turns
	// first close the previous model turn with its suffix.
	render := s.tpl.User.Prefix + userText + s.tpl.User.Suffix + s.tpl.Model.Prefix
	var ids []int32
	if !s.started {
		ids = startIDs(s.e.tok, s.e.md)
		render = renderSystem(s.tpl, s.o.System) + render
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
		if err := prefillEmbedRun(s.e.env, s.e.cm, s.e.pre, s.kv, s.e.emb, s.e.ple, ids[:p], s.pos); err != nil {
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
