package main

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

// decodeEmbeddingInput runs the gemma 3n/4 pipeline: compile the embedder
// stage(s) from the container, allocate the i8 KV cache, prefill all but the
// last prompt token, then greedily decode from the held-back token.
func decodeEmbeddingInput(env litert.Environment, cm litert.CompiledModel, fileBytes []byte, prefill, decode sig, prompt []int32, ngen int, stop map[int32]bool, accel litert.HwAccelerator) ([]int, error) {
	opts, err := litert.NewOptions(accel)
	if err != nil {
		return nil, err
	}
	defer opts.Close()

	embSec, err := litertlm.SectionTFLiteModelType(fileBytes, litertlm.TFLiteEmbedder)
	if err != nil {
		return nil, fmt.Errorf("embedder section: %w", err)
	}
	emb, err := newEmbedModel(env, opts, embSec)
	if err != nil {
		return nil, fmt.Errorf("embedder: %w", err)
	}
	defer emb.close()

	var ple *embedModel
	if sigHasInput(decode, "per_layer_embeddings") {
		pleSec, err := litertlm.SectionTFLiteModelType(fileBytes, litertlm.TFLitePerLayerEmbedder)
		if err != nil {
			return nil, fmt.Errorf("per-layer embedder section: %w", err)
		}
		if ple, err = newEmbedModel(env, opts, pleSec); err != nil {
			return nil, fmt.Errorf("per-layer embedder: %w", err)
		}
		defer ple.close()
	}

	kv := map[string]litert.TensorBuffer{}
	defer func() {
		for _, b := range kv {
			b.Close()
		}
	}()
	for _, name := range prefill.inNames {
		if !isKV(name) {
			continue
		}
		buf, err := allocReqInput(env, cm, prefill, name)
		if err != nil {
			return nil, fmt.Errorf("alloc %s: %w", name, err)
		}
		kv[name] = buf
	}

	p := len(prompt) - 1
	if err := prefillEmbed(env, cm, prefill, kv, emb, ple, prompt[:p]); err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}

	dec, err := newEmbedDecoder(env, cm, decode, kv, emb, ple)
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

func prefillEmbed(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer, emb, ple *embedModel, ids []int32) error {
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
		pos[i] = int32(i)
	}
	n := int32(len(ids))

	for name, b := range bufs {
		if err := fillEmbedInput(b, name, text, perLayer, pos, rows, ctx, int(n), 0); err != nil {
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
}

func newEmbedDecoder(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer, emb, ple *embedModel) (*embedDecoder, error) {
	in, err := allocNonKV(env, cm, g, false)
	if err != nil {
		return nil, err
	}
	out, err := allocNonKV(env, cm, g, true)
	if err != nil {
		closeBufs(in)
		return nil, err
	}
	d := &embedDecoder{emb: emb, ple: ple, in: in, out: out, pos: make([]int32, 1)}

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

func (d *embedDecoder) step(token int32, pos int) (int32, error) {
	text, perLayer, err := embedTokens(d.emb, d.ple, []int32{token})
	if err != nil {
		return 0, err
	}
	d.pos[0] = int32(pos)
	for name, b := range d.in {
		if err := fillEmbedInput(b, name, text, perLayer, d.pos, 1, d.ctx, 1, pos); err != nil {
			return 0, err
		}
	}
	if err := d.runner.Run(); err != nil {
		return 0, err
	}
	return argmaxF32(d.out["logits"], d.vocab)
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
