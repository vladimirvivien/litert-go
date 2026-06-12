package lm

import (
	"context"
	"fmt"
	"strings"

	"github.com/vladimirvivien/litert-go/litert"
)

// Multimodal sentinel token ids marking modality-embedding positions in the
// token sequence (LiteRT-LM ExecutorVisionData/AudioData::kSpecialToken). A
// sentinel's `embeddings` row is the modality embedding; its per-layer embedding
// is left zero (the vision/audio adapters produce no per-layer output).
const (
	visionSoftToken int32 = -1
	audioSoftToken  int32 = -2
)

// requireMultiModal checks the model can splice modality embeddings.
func (e *Engine) requireMultiModal(fn string) error {
	if e.tok == nil {
		return ErrNoTokenizer
	}
	if !sigHasInput(e.decode, "embeddings") {
		return fmt.Errorf("%s: %w", fn, ErrNotEmbeddingModel)
	}
	return nil
}

// generateModal splices precomputed modality embeddings (mm = tReal rows of h
// floats) into the prompt at marker — wrapped beginTok…endTok, each modality
// position a softToken sentinel — and decodes the reply.
func (e *Engine) generateModal(ctx context.Context, prompt, marker, beginTok, endTok string, softToken int32, mm []float32, tReal int, o GenOptions) (string, error) {
	emb, ple, err := e.ensureEmbedders()
	if err != nil {
		return "", err
	}
	h := emb.floats
	if tReal == 0 || len(mm) != tReal*h {
		return "", fmt.Errorf("lm: modality embedding mismatch (%d rows, %d floats, h=%d)", tReal, len(mm), h)
	}
	ids, err := e.buildModalPrompt(prompt, o.System, marker, beginTok, endTok, softToken, tReal)
	if err != nil {
		return "", err
	}
	text, perLayer, err := e.embedModal(emb, ple, ids, mm, h, softToken)
	if err != nil {
		return "", err
	}
	gen, err := decodeMultiModal(ctx, e.env, e.cm, e.pre, e.decode, emb, ple, ids, text, perLayer,
		o.MaxTokens, stopSet(e.tok, e.md), newSampler(o.Temp, o.TopK, o.TopP, o.Seed), e.singleKV())
	if err != nil {
		return "", err
	}
	return e.tok.Decode(gen), nil
}

// buildModalPrompt renders the chat turn and inserts a modality block at the
// marker position: beginTok, tReal softToken sentinels, endTok, at the token
// level.
func (e *Engine) buildModalPrompt(prompt, system, marker, beginTok, endTok string, softToken int32, tReal int) ([]int32, error) {
	tpl, ok := e.templates()
	if !ok {
		return nil, fmt.Errorf("%w (model type %q)", ErrNoChatTemplate, e.md.ModelType)
	}
	render := renderSystem(tpl, system) + tpl.User.Prefix + prompt + tpl.User.Suffix + tpl.Model.Prefix
	before, after, found := strings.Cut(render, marker)
	if !found {
		return nil, fmt.Errorf("lm: prompt has no %q marker", marker)
	}
	ids := startIDs(e.tok, e.md)
	ids = append(ids, e.tok.Encode(before+beginTok)...)
	for i := 0; i < tReal; i++ {
		ids = append(ids, softToken)
	}
	ids = append(ids, e.tok.Encode(endTok+after)...)
	return ids, nil
}

// embedModal builds the text and per-layer embeddings for ids: real tokens go
// through the text/per-layer embedders; softToken positions take the next
// modality row into text and leave per-layer zero.
func (e *Engine) embedModal(emb, ple *embedModel, ids []int32, mm []float32, h int, softToken int32) (text, perLayer []float32, err error) {
	text = make([]float32, len(ids)*h)
	l := 0
	if ple != nil {
		l = ple.floats
		perLayer = make([]float32, len(ids)*l)
	}
	mmIdx := 0
	for i, id := range ids {
		if id == softToken {
			copy(text[i*h:(i+1)*h], mm[mmIdx*h:(mmIdx+1)*h])
			mmIdx++
			continue
		}
		v, err := emb.embed(id)
		if err != nil {
			return nil, nil, err
		}
		copy(text[i*h:(i+1)*h], v)
		if ple != nil {
			w, err := ple.embed(id)
			if err != nil {
				return nil, nil, err
			}
			copy(perLayer[i*l:(i+1)*l], w)
		}
	}
	return text, perLayer, nil
}

// decodeMultiModal prefills a prompt whose text embeddings already have the
// modality rows spliced in (text/perLayer cover all promptIDs), then decodes
// from the held-back last token.
func decodeMultiModal(ctx context.Context, env litert.Environment, cm litert.CompiledModel, pre prefiller, decode sig, emb, ple *embedModel, promptIDs []int32, text, perLayer []float32, ngen int, stop map[int32]bool, smp *sampler, singleKV bool) ([]int, error) {
	kv, err := allocKVBanks(env, cm, pre.max(), singleKV)
	if err != nil {
		return nil, err
	}
	defer kv.close()

	n := len(promptIDs)
	h := len(text) / n
	l := 0
	if len(perLayer) > 0 {
		l = len(perLayer) / n
	}
	p := n - 1
	if err := prefillEmbedDataRun(ctx, env, cm, pre, kv, text[:p*h], perLayer[:p*l], p, 0); err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}

	dec, err := newEmbedDecoder(env, cm, decode, kv, emb, ple, smp)
	if err != nil {
		return nil, fmt.Errorf("decode setup: %w", err)
	}
	defer dec.close()

	next := promptIDs[p]
	pos := p
	var gen []int
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
		next = id
		pos++
	}
	return gen, nil
}
