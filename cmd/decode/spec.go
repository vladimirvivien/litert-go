package main

import (
	"fmt"
	"os"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// MTP speculative decoding (gemma 3n/4). One decode step produces a token plus
// its hidden state; a small MTP drafter — sharing the base model's KV for its
// final layers — recurrently proposes K draft tokens from embedding ⊕ activation;
// the base model's `verify` signature checks all K+1 in one pass; the matching
// prefix is accepted plus one bonus token. Output is identical to greedy decode
// (verification guarantees it), with fewer base-model forward passes. Follows
// LiteRT-LM's llm_litert_mtp_drafter.

// mtpDraft wraps the compiled tf_lite_mtp_drafter model. Its KV inputs are the
// base model's KV buffers (shared, layers it owns); it writes no KV. Per draft
// step it consumes embedding ⊕ activation (f32[1,1,2H]) and emits a token plus
// projected_activations (the recurrent state for the next step).
type mtpDraft struct {
	model  litert.Model
	cm     litert.CompiledModel
	in     map[string]litert.TensorBuffer // input_pos, activations, param_tensor, mask
	out    map[string]litert.TensorBuffer // logits, projected_activations
	runner *litert.Runner
	ctx    int
	vocab  int
	actLen int // projected_activations elements
	pos    []int32
}

func newMtpDraft(env litert.Environment, opts litert.Options, section []byte, mainKV map[string]litert.TensorBuffer) (*mtpDraft, error) {
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
	in, err := allocNonKV(env, cm, g, false)
	if err != nil {
		return nil, err
	}
	out, err := allocNonKV(env, cm, g, true)
	if err != nil {
		closeBufs(in)
		return nil, err
	}
	d := &mtpDraft{model: model, cm: cm, in: in, out: out, pos: make([]int32, 1)}
	maskShape, _ := inputShape(g, "mask")
	d.ctx = int(maskShape[3])
	d.vocab = elemCount(g, "logits", true)
	d.actLen = elemCount(g, "projected_activations", true)

	inArr := assemble(g.inNames, in, mainKV)
	outArr := assemble(g.outNames, out, mainKV)
	for i, b := range inArr {
		if b == 0 {
			closeBufs(in)
			closeBufs(out)
			return nil, fmt.Errorf("drafter unmapped input[%d] %q", i, g.inNames[i])
		}
	}
	d.runner = litert.NewRunner(cm, g.idx, inArr, outArr)
	return d, nil
}

// setPos fixes the drafter's input_pos / mask / param_tensor for the draft
// burst (a single base position; LiteRT-LM does not advance them per step).
func (d *mtpDraft) setPos(pos int) error {
	d.pos[0] = int32(pos)
	for name, b := range d.in {
		switch name {
		case "input_pos":
			if err := writeInts(b, d.pos); err != nil {
				return err
			}
		case "mask":
			if err := fillBoolMask(b, 1, d.ctx, 1, pos); err != nil {
				return err
			}
		case "param_tensor":
			if err := writeInts(b, []int32{int32(pos), int32(pos + 1), int32(pos + 1)}); err != nil {
				return err
			}
		}
	}
	return nil
}

// draft runs one step on embedding ⊕ activation, returning the draft token and
// the projected activation to chain into the next step.
func (d *mtpDraft) draft(embedding, activation []float32) (int32, []float32, error) {
	cat := make([]float32, len(embedding)+len(activation))
	copy(cat, embedding)
	copy(cat[len(embedding):], activation)
	if err := writeFloats(d.in["activations"], cat); err != nil {
		return 0, nil, err
	}
	if err := d.runner.Run(); err != nil {
		return 0, nil, err
	}
	id, err := argmaxF32(d.out["logits"], d.vocab)
	if err != nil {
		return 0, nil, err
	}
	proj, err := readFloats(d.out["projected_activations"], d.actLen)
	if err != nil {
		return 0, nil, err
	}
	return id, proj, nil
}

func (d *mtpDraft) close() {
	d.runner.Close()
	closeBufs(d.in)
	closeBufs(d.out)
	d.cm.Close()
	d.model.Close()
}

// verifier runs the base model's `verify` signature over width tokens at once.
type verifier struct {
	emb, ple *embedModel
	in       map[string]litert.TensorBuffer
	out      map[string]litert.TensorBuffer
	runner   *litert.Runner
	width    int
	ctx      int
	vocab    int
}

func newVerifier(env litert.Environment, cm litert.CompiledModel, g sig, mainKV map[string]litert.TensorBuffer, emb, ple *embedModel) (*verifier, error) {
	in, err := allocNonKV(env, cm, g, false)
	if err != nil {
		return nil, err
	}
	out, err := allocNonKV(env, cm, g, true)
	if err != nil {
		closeBufs(in)
		return nil, err
	}
	posShape, _ := inputShape(g, "input_pos")
	maskShape, _ := inputShape(g, "mask")
	v := &verifier{
		emb: emb, ple: ple, in: in, out: out,
		width: int(posShape[0]),
		ctx:   int(maskShape[3]),
	}
	v.vocab = elemCount(g, "logits", true) / v.width

	inArr := assemble(g.inNames, in, mainKV)
	outArr := assemble(g.outNames, out, mainKV)
	for i, b := range inArr {
		if b == 0 {
			closeBufs(in)
			closeBufs(out)
			return nil, fmt.Errorf("verifier unmapped input[%d] %q", i, g.inNames[i])
		}
	}
	v.runner = litert.NewRunner(cm, g.idx, inArr, outArr)
	return v, nil
}

// verify runs the width tokens at positions [pos, pos+width) and returns the
// argmax token at each position.
func (v *verifier) verify(pos int, tokens []int32) ([]int32, error) {
	text, perLayer, err := embedTokens(v.emb, v.ple, tokens)
	if err != nil {
		return nil, err
	}
	posArr := make([]int32, v.width)
	for i := range posArr {
		posArr[i] = int32(pos + i)
	}
	for name, b := range v.in {
		if err := fillEmbedInput(b, name, text, perLayer, posArr, v.width, v.ctx, v.width, pos); err != nil {
			return nil, err
		}
	}
	if err := v.runner.Run(); err != nil {
		return nil, err
	}
	return argmaxRows(v.out["logits"], v.width, v.vocab)
}

func (v *verifier) close() {
	v.runner.Close()
	closeBufs(v.in)
	closeBufs(v.out)
}

// elemCount returns the product of a signature input/output tensor's shape.
func elemCount(g sig, name string, out bool) int {
	tt, err := g.s.InputType(name)
	if out {
		tt, err = g.s.OutputType(name)
	}
	if err != nil {
		return 0
	}
	n := 1
	for _, d := range tt.Shape {
		n *= int(d)
	}
	return n
}

// decodeSpeculative runs MTP speculative decoding on an embedding-input model
// that has a verify signature and an mtp_drafter section.
func decodeSpeculative(env litert.Environment, cm litert.CompiledModel, fileBytes []byte, prefill, decode, verifySig sig, prompt []int32, ngen int, stop map[int32]bool, accel litert.HwAccelerator) ([]int, error) {
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

	draftSec, err := litertlm.SectionTFLiteModelType(fileBytes, litertlm.TFLiteMTPDrafter)
	if err != nil {
		return nil, fmt.Errorf("mtp drafter section: %w", err)
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

	dec, err := newEmbedDecoder(env, cm, decode, kv, emb, ple, &sampler{}) // greedy
	if err != nil {
		return nil, fmt.Errorf("decode setup: %w", err)
	}
	defer dec.close()
	ver, err := newVerifier(env, cm, verifySig, kv, emb, ple)
	if err != nil {
		return nil, fmt.Errorf("verify setup: %w", err)
	}
	defer ver.close()
	drf, err := newMtpDraft(env, opts, draftSec, kv)
	if err != nil {
		return nil, fmt.Errorf("drafter setup: %w", err)
	}
	defer drf.close()

	draftSteps := ver.width - 1
	var gen []int
	passes := 0 // verify passes (≈ one base-model forward each)
	pos := p
	pending := prompt[p]
	defer func() {
		if passes > 0 {
			fmt.Fprintf(os.Stderr, "spec: %d tokens over %d verify passes (%.2f tokens/pass)\n",
				len(gen), passes, float64(len(gen))/float64(passes))
		}
	}()
	for len(gen) < ngen {
		passes++
		tokenID, act, err := dec.stepAct(pending, pos)
		if err != nil {
			return nil, fmt.Errorf("decode @%d: %w", pos, err)
		}
		if stop[tokenID] {
			break
		}
		gen = append(gen, int(tokenID))
		if len(gen) >= ngen {
			break
		}

		// Draft K tokens from the decode token + activation.
		if err := drf.setPos(pos); err != nil {
			return nil, err
		}
		drafts := make([]int32, 0, draftSteps)
		curTok, curAct := tokenID, act
		for i := 0; i < draftSteps; i++ {
			ev, err := emb.embed(curTok)
			if err != nil {
				return nil, err
			}
			d, proj, err := drf.draft(ev, curAct)
			if err != nil {
				return nil, fmt.Errorf("draft %d: %w", i, err)
			}
			drafts = append(drafts, d)
			curTok, curAct = d, proj
		}

		// Verify [tokenID, drafts...] at positions [pos+1, ...].
		v, err := ver.verify(pos+1, append([]int32{tokenID}, drafts...))
		if err != nil {
			return nil, fmt.Errorf("verify @%d: %w", pos+1, err)
		}

		// Accept the matching prefix; first mismatch yields the bonus token.
		nc, bonus := 0, v[draftSteps]
		for i := 0; i < draftSteps; i++ {
			if v[i] != drafts[i] {
				bonus = v[i]
				break
			}
			nc++
		}

		done := false
		for i := 0; i < nc; i++ {
			if stop[drafts[i]] {
				done = true
				break
			}
			gen = append(gen, int(drafts[i]))
			if len(gen) >= ngen {
				done = true
				break
			}
		}
		if done {
			break
		}
		if stop[bonus] {
			break
		}
		gen = append(gen, int(bonus))

		pos = pos + nc + 2
		pending = bonus
	}
	return gen, nil
}
