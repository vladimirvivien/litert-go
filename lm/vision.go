package lm

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unsafe"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
	"github.com/vladimirvivien/litert-go/vision"
)

// visionPipeline runs gemma-4's two-stage image path: the ViT encoder (image
// patches -> visual features + a validity mask) and the adapter (features ->
// embeddings at the text model's dimension). Both are compiled once on the
// Engine and reused; each exposes per-token-count buckets (vision_70/140/280).
type visionPipeline struct {
	encModel, adpModel litert.Model
	encCM, adpCM       litert.CompiledModel
	opts               litert.Options
	encSigs            map[int]sig
	adpSigs            map[int]sig
	sizes              []int // bucket token counts, ascending
}

func (e *Engine) ensureVision() (*visionPipeline, error) {
	if e.vision != nil {
		return e.vision, nil
	}
	encSec, err := litertlm.SectionTFLiteModelType(e.fileBytes, litertlm.TFLiteVisionEncoder)
	if err != nil {
		return nil, fmt.Errorf("vision encoder section: %w", err)
	}
	adpSec, err := litertlm.SectionTFLiteModelType(e.fileBytes, litertlm.TFLiteVisionAdapter)
	if err != nil {
		return nil, fmt.Errorf("vision adapter section: %w", err)
	}
	opts, err := litert.NewOptions(e.accel)
	if err != nil {
		return nil, err
	}
	v := &visionPipeline{opts: opts}
	done := false
	defer func() {
		if !done {
			v.close()
		}
	}()

	if v.encModel, err = litert.OpenModelFromBuffer(e.env, encSec); err != nil {
		return nil, err
	}
	if v.encCM, err = litert.Compile(e.env, v.encModel, opts); err != nil {
		return nil, err
	}
	if v.adpModel, err = litert.OpenModelFromBuffer(e.env, adpSec); err != nil {
		return nil, err
	}
	if v.adpCM, err = litert.Compile(e.env, v.adpModel, opts); err != nil {
		return nil, err
	}
	if v.encSigs, v.sizes, err = visionBuckets(v.encModel); err != nil {
		return nil, fmt.Errorf("encoder buckets: %w", err)
	}
	if v.adpSigs, _, err = visionBuckets(v.adpModel); err != nil {
		return nil, fmt.Errorf("adapter buckets: %w", err)
	}
	done = true
	e.vision = v
	return v, nil
}

// visionBuckets maps each "name_<N>" signature to its token-count bucket N.
func visionBuckets(m litert.Model) (map[int]sig, []int, error) {
	n, _ := m.NumSignatures()
	sigs := map[int]sig{}
	var sizes []int
	for i := 0; i < n; i++ {
		g, err := loadSig(m, i)
		if err != nil {
			return nil, nil, err
		}
		key, _ := g.s.Key()
		u := strings.LastIndexByte(key, '_')
		if u < 0 {
			continue
		}
		size, err := strconv.Atoi(key[u+1:])
		if err != nil {
			continue
		}
		sigs[size] = g
		sizes = append(sizes, size)
	}
	if len(sizes) == 0 {
		return nil, nil, fmt.Errorf("no bucketed signatures")
	}
	sort.Ints(sizes)
	return sigs, sizes, nil
}

// encode runs the encoder + adapter for one preprocessed image, returning the
// adapter embeddings [tReal, embDim] (flattened) and the real visual-token count.
func (v *visionPipeline) encode(env litert.Environment, p *vision.Patches) ([]float32, int, error) {
	bucket := 0
	for _, s := range v.sizes {
		if s >= p.Tokens {
			bucket = s
			break
		}
	}
	if bucket == 0 {
		return nil, 0, fmt.Errorf("vision: %d tokens exceeds largest bucket %d", p.Tokens, v.sizes[len(v.sizes)-1])
	}
	encG, ok := v.encSigs[bucket]
	if !ok {
		return nil, 0, fmt.Errorf("vision: no encoder bucket %d", bucket)
	}
	adpG, ok := v.adpSigs[bucket]
	if !ok {
		return nil, 0, fmt.Errorf("vision: no adapter bucket %d", bucket)
	}

	imgShape, _ := inputShape(encG, "images") // [1, capPatches, patchDim]
	capPatches := int(imgShape[1])
	if p.Count > capPatches {
		return nil, 0, fmt.Errorf("vision: %d patches exceeds bucket capacity %d", p.Count, capPatches)
	}

	images, err := allocReqInput(env, v.encCM, encG, "images")
	if err != nil {
		return nil, 0, err
	}
	defer images.Close()
	positions, err := allocReqInput(env, v.encCM, encG, "positions_xy")
	if err != nil {
		return nil, 0, err
	}
	defer positions.Close()
	features, err := allocReqOutput(env, v.encCM, encG, "features")
	if err != nil {
		return nil, 0, err
	}
	defer features.Close()
	mask, err := allocReqOutput(env, v.encCM, encG, "mask")
	if err != nil {
		return nil, 0, err
	}
	defer mask.Close()

	if err := writeFloats(images, p.Images); err != nil { // rest of the bucket stays zero
		return nil, 0, err
	}
	pos := make([]int32, capPatches*2)
	for i := range pos {
		pos[i] = -1 // padding positions
	}
	copy(pos, p.Positions)
	if err := writeInts(positions, pos); err != nil {
		return nil, 0, err
	}

	encIn := assemble(encG.inNames, map[string]litert.TensorBuffer{"images": images, "positions_xy": positions}, nil)
	encOut := assemble(encG.outNames, map[string]litert.TensorBuffer{"features": features, "mask": mask}, nil)
	if err := v.encCM.Run(encG.idx, encIn, encOut); err != nil {
		return nil, 0, fmt.Errorf("vision encoder: %w", err)
	}

	featType, err := encG.s.OutputType("features") // [1, tBucket, featDim]
	if err != nil {
		return nil, 0, err
	}
	tBucket, featDim := int(featType.Shape[1]), int(featType.Shape[2])
	maskBytes, err := readBytes(mask, tBucket)
	if err != nil {
		return nil, 0, err
	}
	tReal := 0
	for _, b := range maskBytes {
		if b != 0 {
			tReal++
		}
	}
	if tReal == 0 {
		tReal = p.Tokens
	}
	feat, err := readFloats(features, tBucket*featDim)
	if err != nil {
		return nil, 0, err
	}

	soft, err := allocReqInput(env, v.adpCM, adpG, "soft_tokens")
	if err != nil {
		return nil, 0, err
	}
	defer soft.Close()
	mm, err := allocReqOutput(env, v.adpCM, adpG, "mm_embedding")
	if err != nil {
		return nil, 0, err
	}
	defer mm.Close()
	if err := writeFloats(soft, feat[:tReal*featDim]); err != nil {
		return nil, 0, err
	}
	adpIn := assemble(adpG.inNames, map[string]litert.TensorBuffer{"soft_tokens": soft}, nil)
	adpOut := assemble(adpG.outNames, map[string]litert.TensorBuffer{"mm_embedding": mm}, nil)
	if err := v.adpCM.Run(adpG.idx, adpIn, adpOut); err != nil {
		return nil, 0, fmt.Errorf("vision adapter: %w", err)
	}

	mmType, err := adpG.s.OutputType("mm_embedding") // [1, tBucket, embDim]
	if err != nil {
		return nil, 0, err
	}
	embDim := int(mmType.Shape[2])
	all, err := readFloats(mm, tBucket*embDim)
	if err != nil {
		return nil, 0, err
	}
	return all[:tReal*embDim], tReal, nil
}

func (v *visionPipeline) close() {
	if v.encCM != 0 {
		v.encCM.Close()
	}
	if v.adpCM != 0 {
		v.adpCM.Close()
	}
	if v.encModel != 0 {
		v.encModel.Close()
	}
	if v.adpModel != 0 {
		v.adpModel.Close()
	}
	if v.opts != 0 {
		v.opts.Close()
	}
}

// imageMarker is the placeholder a prompt uses to position an image; it expands
// to <start_of_image> + the image's soft-token sentinels + <end_of_image>.
const imageMarker = "<start_of_image>"

// visionSoftToken is the placeholder token id at each image position in the token
// sequence (LiteRT-LM's ExecutorVisionData::kSpecialToken). Its `embeddings` row
// is the image embedding; its per-layer embedding is left zero (the vision
// adapter produces no per-layer output).
const visionSoftToken int32 = -1

// GenerateFromImage generates text for a prompt that references a single image.
// The prompt must contain imageMarker ("<start_of_image>") where the image
// belongs. budget is the visual-token budget (0 = default). Embedding-input
// (gemma 3n/4) models with a vision stack only.
func (e *Engine) GenerateFromImage(prompt string, imageData []byte, budget int, o GenOptions) (string, error) {
	if e.tok == nil {
		return "", fmt.Errorf("lm: model has no tokenizer")
	}
	if !sigHasInput(e.decode, "embeddings") {
		return "", fmt.Errorf("lm: GenerateFromImage requires an embedding-input model")
	}
	emb, ple, err := e.ensureEmbedders()
	if err != nil {
		return "", err
	}
	vp, err := e.ensureVision()
	if err != nil {
		return "", err
	}

	patches, err := vision.Preprocess(imageData, budget)
	if err != nil {
		return "", err
	}
	mm, tReal, err := vp.encode(e.env, patches)
	if err != nil {
		return "", err
	}

	ids, err := e.buildImagePrompt(prompt, o.System, tReal)
	if err != nil {
		return "", err
	}

	h := emb.floats
	if len(mm) != tReal*h {
		return "", fmt.Errorf("lm: image embedding dim %d != text embedding dim %d", len(mm)/tReal, h)
	}
	text, perLayer, err := e.embedMultiModal(emb, ple, ids, mm, h)
	if err != nil {
		return "", err
	}

	gen, err := decodeMultiModal(e.env, e.cm, e.pre, e.decode, emb, ple, ids, text, perLayer,
		o.MaxTokens, stopSet(e.tok, e.md), newSampler(o.Temp, o.TopK, o.TopP, o.Seed))
	if err != nil {
		return "", err
	}
	return e.tok.Decode(gen), nil
}

// buildImagePrompt renders the chat turn and inserts the image block at the
// imageMarker position: <start_of_image>, tReal soft-token sentinels (-1),
// <end_of_image>, at the token level.
func (e *Engine) buildImagePrompt(prompt, system string, tReal int) ([]int32, error) {
	tpl, ok := e.md.Templates()
	if !ok {
		return nil, fmt.Errorf("lm: model has no chat template (model type %q)", e.md.ModelType)
	}
	render := renderSystem(tpl, system) + tpl.User.Prefix + prompt + tpl.User.Suffix + tpl.Model.Prefix
	before, after, found := strings.Cut(render, imageMarker)
	if !found {
		return nil, fmt.Errorf("lm: prompt has no %q marker", imageMarker)
	}
	ids := startIDs(e.tok, e.md)
	ids = append(ids, e.tok.Encode(before+"<start_of_image>")...)
	for i := 0; i < tReal; i++ {
		ids = append(ids, visionSoftToken)
	}
	ids = append(ids, e.tok.Encode("<end_of_image>"+after)...)
	return ids, nil
}

// embedMultiModal builds the text and per-layer embeddings for ids: real tokens
// go through the text/per-layer embedders; visionSoftToken (-1) positions take
// the next image row into text and leave per-layer zero.
func (e *Engine) embedMultiModal(emb, ple *embedModel, ids []int32, mm []float32, h int) (text, perLayer []float32, err error) {
	text = make([]float32, len(ids)*h)
	l := 0
	if ple != nil {
		l = ple.floats
		perLayer = make([]float32, len(ids)*l)
	}
	mmIdx := 0
	for i, id := range ids {
		if id == visionSoftToken {
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
// image rows spliced in (text/perLayer cover all promptIDs), then decodes from
// the held-back last token.
func decodeMultiModal(env litert.Environment, cm litert.CompiledModel, pre prefiller, decode sig, emb, ple *embedModel, promptIDs []int32, text, perLayer []float32, ngen int, stop map[int32]bool, smp *sampler) ([]int, error) {
	kv, err := allocKV(env, cm, pre.max())
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, b := range kv {
			b.Close()
		}
	}()

	n := len(promptIDs)
	h := len(text) / n
	l := 0
	if len(perLayer) > 0 {
		l = len(perLayer) / n
	}
	p := n - 1
	if err := prefillEmbedDataRun(env, cm, pre, kv, text[:p*h], perLayer[:p*l], p, 0); err != nil {
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

// readBytes copies the first n bytes of a buffer out to Go memory.
func readBytes(b litert.TensorBuffer, n int) ([]byte, error) {
	addr, err := b.Lock(litert.LockRead)
	if err != nil {
		return nil, err
	}
	defer b.Unlock()
	out := make([]byte, n)
	copy(out, unsafe.Slice((*byte)(addr), n))
	return out, nil
}
