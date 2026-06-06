// Command decode is the litert-go Phase 1 harness: a real greedy decode.
//
// It runs the LiteRT-LM static-executor protocol for a fixed-context model
// (e.g. Qwen3-0.6B, 4096-token cache) on CPU: prefill an N-token prompt, then
// greedily decode tokens one at a time. The model's signatures are statically
// shaped, so no resizing or KV-cache growth is needed — the KV cache is the
// full fixed context, the model writes each step's K/V by position, and the
// causal attention mask gates which positions are visible.
//
// The prompt is supplied as token IDs (-prompt); a tokenizer is future work, so
// this prints generated token IDs, not text.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"unsafe"

	sentencepiece "github.com/eliben/go-sentencepiece"
	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// maskNeg is the "masked" attention value LiteRT-LM uses for f32 (-0.7 * FLT_MAX).
const maskNeg = float32(-0.7 * math.MaxFloat32)

func main() {
	libDir := flag.String("lib", "", "directory or path of libLiteRt (or set LITERT_LIB)")
	modelPath := flag.String("model", "", "path to a .litertlm container or raw .tflite")
	text := flag.String("text", "", "text prompt (uses the model's embedded SentencePiece tokenizer)")
	promptCSV := flag.String("prompt", "", "prompt token IDs, comma-separated (alternative to -text)")
	ngen := flag.Int("n", 16, "max number of tokens to generate")
	chat := flag.Bool("chat", false, "wrap -text in the model's chat template (from container metadata)")
	flag.Parse()

	if *modelPath == "" || (*text == "" && *promptCSV == "") {
		fmt.Fprintln(os.Stderr, "decode: -model and one of -text/-prompt are required")
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*libDir, *modelPath, *text, *promptCSV, *ngen, *chat); err != nil {
		fmt.Fprintln(os.Stderr, "decode:", err)
		os.Exit(1)
	}
}

func parseIDs(csv string) ([]int32, error) {
	parts := strings.Split(csv, ",")
	ids := make([]int32, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, fmt.Errorf("bad token id %q: %w", p, err)
		}
		ids = append(ids, int32(n))
	}
	if len(ids) < 2 {
		return nil, fmt.Errorf("need at least 2 prompt tokens")
	}
	return ids, nil
}

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

func run(libDir, modelPath, text, promptCSV string, ngen int, chat bool) error {
	if err := litert.Load(libDir); err != nil {
		return err
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		return err
	}
	defer env.Close()

	fileBytes, err := os.ReadFile(modelPath)
	if err != nil {
		return err
	}
	tflite := fileBytes
	var tok *sentencepiece.Processor
	var md litertlm.Metadata
	if litertlm.IsContainer(fileBytes) {
		if tflite, err = litertlm.SectionTFLite(fileBytes); err != nil {
			return err
		}
		md, _ = litertlm.ReadMetadata(fileBytes)
		if sp, e := litertlm.SectionBytes(fileBytes, litertlm.SectionSPTokenizer); e == nil {
			if tok, err = sentencepiece.NewProcessor(bytes.NewReader(sp)); err != nil {
				return fmt.Errorf("load tokenizer: %w", err)
			}
		}
	}
	model, err := litert.OpenModelFromBuffer(env, tflite)
	if err != nil {
		return err
	}
	defer model.Close()
	defer runtime.KeepAlive(fileBytes) // model holds C pointers into tflite (a slice of fileBytes)

	nsig, _ := model.NumSignatures()
	var prefill, decode sig
	var havePrefill, haveDecode bool
	for i := 0; i < nsig; i++ {
		s, _ := model.Signature(i)
		key, _ := s.Key()
		switch {
		case key == "decode":
			if decode, err = loadSig(model, i); err != nil {
				return err
			}
			haveDecode = true
		case strings.HasPrefix(key, "prefill") && !havePrefill:
			if prefill, err = loadSig(model, i); err != nil {
				return err
			}
			havePrefill = true
		}
	}
	if !havePrefill || !haveDecode {
		return fmt.Errorf("model lacks prefill/decode signatures")
	}

	opts, err := litert.NewOptions(litert.AccelCPU)
	if err != nil {
		return err
	}
	defer opts.Close()
	cm, err := litert.Compile(env, model, opts)
	if err != nil {
		return err
	}
	defer cm.Close()

	// KV cache: allocate once at the model's fixed (concrete) shapes, shared by
	// both prefill and decode.
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
			return fmt.Errorf("alloc %s: %w", name, err)
		}
		kv[name] = buf
	}

	prompt, err := buildPrompt(tok, md, text, promptCSV, chat)
	if err != nil {
		return err
	}

	// Prefill the first P tokens (hold back the last prompt token for decode).
	p := len(prompt) - 1
	if err := prefillStep(env, cm, prefill, kv, prompt[:p]); err != nil {
		return fmt.Errorf("prefill: %w", err)
	}

	// Greedy decode: first step consumes the held-back token at position P;
	// stop at end-of-sentence. The decode buffer set is fixed, so allocate it
	// once and reuse it — and its pinned Run arguments — across every step.
	dec, err := newDecoder(env, cm, decode, kv)
	if err != nil {
		return fmt.Errorf("decode setup: %w", err)
	}
	defer dec.close()

	stop := stopSet(tok, md)
	next := prompt[p]
	pos := p
	var gen []int
	for g := 0; g < ngen; g++ {
		id, err := dec.step(next, pos)
		if err != nil {
			return fmt.Errorf("decode step %d: %w", g, err)
		}
		if stop[id] {
			break
		}
		gen = append(gen, int(id))
		next = id
		pos++
	}

	if tok != nil {
		fmt.Printf("prompt: %q\noutput: %q\n", text, tok.Decode(gen))
	} else {
		fmt.Printf("prompt=%v\noutput tokens=%v\n", prompt, gen)
	}
	return nil
}

// buildPrompt produces prompt token IDs. With -prompt it parses raw IDs; with
// -text it tokenizes through the embedded SentencePiece tokenizer — as a raw
// completion (start token + text), or, with chat set, wrapped in the model's
// chat template from the container metadata.
func buildPrompt(tok *sentencepiece.Processor, md litertlm.Metadata, text, promptCSV string, chat bool) ([]int32, error) {
	if promptCSV != "" {
		return parseIDs(promptCSV)
	}
	if tok == nil {
		return nil, fmt.Errorf("model has no SentencePiece tokenizer; use -prompt with token IDs")
	}
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

// buildChatPrompt wraps userText in the model's single user turn:
// start ++ user.prefix ++ userText ++ user.suffix ++ model.prefix. The turn
// markers in the affixes (e.g. <start_of_turn>) are user-defined tokenizer
// pieces, so encoding the rendered string yields their single vocab IDs.
func buildChatPrompt(tok *sentencepiece.Processor, md litertlm.Metadata, userText string) ([]int32, error) {
	tpl, ok := md.Templates()
	if !ok {
		return nil, fmt.Errorf("no chat template for model type %q; use -text without -chat, or -prompt", md.ModelType)
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

func prefillStep(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer, ids []int32) error {
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
		posVals[i] = int32(i)
	}
	if err := writeInts(pos, posVals); err != nil {
		return err
	}
	// Causal mask: row i (position i) attends columns [0, i+1). Only the first p
	// rows carry real tokens; filling all rows causally is harmless.
	if err := fillCausalMask(mask, seq, ctx, 0); err != nil {
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
}

func newDecoder(env litert.Environment, cm litert.CompiledModel, g sig, kv map[string]litert.TensorBuffer) (*decoder, error) {
	d := &decoder{}
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

func (d *decoder) step(token int32, pos int) (int32, error) {
	if err := writeInts(d.tokens, []int32{token}); err != nil {
		return 0, err
	}
	if err := writeInts(d.posBuf, []int32{int32(pos)}); err != nil {
		return 0, err
	}
	if err := fillCausalMask(d.mask, 1, d.ctx, pos); err != nil {
		return 0, err
	}
	if err := d.runner.Run(); err != nil {
		return 0, err
	}
	return argmaxF32(d.logits, d.vocab)
}

func (d *decoder) close() {
	d.runner.Close()
	d.logits.Close()
	d.mask.Close()
	d.posBuf.Close()
	d.tokens.Close()
}

func inputShape(g sig, name string) ([]int32, error) {
	tt, err := g.s.InputType(name)
	if err != nil {
		return nil, err
	}
	return tt.Shape, nil
}

// allocReqInput allocates a zeroed buffer for a signature input using the
// compiled model's required size and buffer type (not a hand-computed size).
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
