package lm

import (
	"fmt"

	"github.com/vladimirvivien/litert-go/audio"
	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// audioPipeline runs gemma-4's audio path: the encoder (log-mel frames + a
// validity mask -> features + a token mask) and the adapter (features ->
// embeddings at the text dimension). Both are single-signature (no buckets) and
// compiled once on the Engine.
type audioPipeline struct {
	encModel, adpModel litert.Model
	encCM, adpCM       litert.CompiledModel
	opts               litert.Options
	encG, adpG         sig
}

func (e *Engine) ensureAudio() (*audioPipeline, error) {
	if e.audio != nil {
		return e.audio, nil
	}
	encSec, err := litertlm.SectionTFLiteModelType(e.fileBytes, litertlm.TFLiteAudioEncoder)
	if err != nil {
		return nil, fmt.Errorf("audio encoder section: %w", err)
	}
	adpSec, err := litertlm.SectionTFLiteModelType(e.fileBytes, litertlm.TFLiteAudioAdapter)
	if err != nil {
		return nil, fmt.Errorf("audio adapter section: %w", err)
	}
	opts, err := e.compileOptions("audio")
	if err != nil {
		return nil, err
	}
	a := &audioPipeline{opts: opts}
	done := false
	defer func() {
		if !done {
			a.close()
		}
	}()

	if a.encModel, err = litert.OpenModelFromBuffer(e.env, encSec); err != nil {
		return nil, err
	}
	if a.encCM, err = litert.Compile(e.env, a.encModel, opts); err != nil {
		return nil, err
	}
	if a.encG, err = loadSig(a.encModel, 0); err != nil {
		return nil, err
	}
	if a.adpModel, err = litert.OpenModelFromBuffer(e.env, adpSec); err != nil {
		return nil, err
	}
	if a.adpCM, err = litert.Compile(e.env, a.adpModel, opts); err != nil {
		return nil, err
	}
	if a.adpG, err = loadSig(a.adpModel, 0); err != nil {
		return nil, err
	}
	done = true
	e.audio = a
	return a, nil
}

// encode runs the encoder + adapter for one log-mel spectrogram, returning the
// adapter embeddings [tReal, embDim] (flattened) and the real audio-token count.
func (a *audioPipeline) encode(env litert.Environment, mel *audio.MelSpectrogram) ([]float32, int, error) {
	srcShape, _ := inputShape(a.encG, "src_inputs") // [1, 1, capFrames, melBins]
	capFrames, melBins := int(srcShape[2]), int(srcShape[3])
	frames := mel.Frames
	if frames > capFrames {
		frames = capFrames
	}

	src, err := allocReqInput(env, a.encCM, a.encG, "src_inputs")
	if err != nil {
		return nil, 0, err
	}
	defer src.Close()
	inMask, err := allocReqInput(env, a.encCM, a.encG, "mask")
	if err != nil {
		return nil, 0, err
	}
	defer inMask.Close()
	features, err := allocReqOutput(env, a.encCM, a.encG, "features")
	if err != nil {
		return nil, 0, err
	}
	defer features.Close()
	outMask, err := allocReqOutput(env, a.encCM, a.encG, "mask")
	if err != nil {
		return nil, 0, err
	}
	defer outMask.Close()

	if err := writeFloats(src, mel.Mel[:frames*melBins]); err != nil { // rest stays zero
		return nil, 0, err
	}
	mask := make([]byte, capFrames)
	for i := 0; i < frames; i++ {
		mask[i] = 1
	}
	if err := writeBytes(inMask, mask); err != nil {
		return nil, 0, err
	}

	encIn := assemble(a.encG.inNames, map[string]litert.TensorBuffer{"src_inputs": src, "mask": inMask}, nil)
	encOut := assemble(a.encG.outNames, map[string]litert.TensorBuffer{"features": features, "mask": outMask}, nil)
	if err := a.encCM.Run(a.encG.idx, encIn, encOut); err != nil {
		return nil, 0, fmt.Errorf("audio encoder: %w", err)
	}

	featType, err := a.encG.s.OutputType("features") // [1, tBucket, featDim]
	if err != nil {
		return nil, 0, err
	}
	tBucket, featDim := int(featType.Shape[1]), int(featType.Shape[2])
	maskBytes, err := readBytes(outMask, tBucket)
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
		return nil, 0, fmt.Errorf("audio encoder produced no tokens")
	}
	feat, err := readFloats(features, tBucket*featDim)
	if err != nil {
		return nil, 0, err
	}

	adpFeat, err := allocReqInput(env, a.adpCM, a.adpG, "features")
	if err != nil {
		return nil, 0, err
	}
	defer adpFeat.Close()
	adpMask, err := allocReqInput(env, a.adpCM, a.adpG, "mask")
	if err != nil {
		return nil, 0, err
	}
	defer adpMask.Close()
	out, err := allocReqOutput(env, a.adpCM, a.adpG, "output_0")
	if err != nil {
		return nil, 0, err
	}
	defer out.Close()
	if err := writeFloats(adpFeat, feat); err != nil {
		return nil, 0, err
	}
	if err := writeBytes(adpMask, maskBytes); err != nil {
		return nil, 0, err
	}
	adpIn := assemble(a.adpG.inNames, map[string]litert.TensorBuffer{"features": adpFeat, "mask": adpMask}, nil)
	adpOut := assemble(a.adpG.outNames, map[string]litert.TensorBuffer{"output_0": out}, nil)
	if err := a.adpCM.Run(a.adpG.idx, adpIn, adpOut); err != nil {
		return nil, 0, fmt.Errorf("audio adapter: %w", err)
	}

	outType, err := a.adpG.s.OutputType("output_0") // [1, tBucket, embDim]
	if err != nil {
		return nil, 0, err
	}
	embDim := int(outType.Shape[2])
	all, err := readFloats(out, tBucket*embDim)
	if err != nil {
		return nil, 0, err
	}
	return all[:tReal*embDim], tReal, nil
}

// GenerateFromAudio generates text for a prompt that references a single audio
// clip given as 16 kHz mono PCM samples. The prompt must contain
// "<start_of_audio>" where the audio belongs. Embedding-input (gemma 3n/4) models
// with an audio stack only.
func (e *Engine) GenerateFromAudio(prompt string, pcm []float32, o GenOptions) (string, error) {
	if err := e.requireMultiModal("GenerateFromAudio"); err != nil {
		return "", err
	}
	ap, err := e.ensureAudio()
	if err != nil {
		return "", err
	}
	mel := audio.Preprocess(pcm)
	mm, tReal, err := ap.encode(e.env, mel)
	if err != nil {
		return "", err
	}
	return e.generateModal(prompt, "<start_of_audio>", "<start_of_audio>", "<end_of_audio>", audioSoftToken, mm, tReal, o)
}

func (a *audioPipeline) close() {
	if a.encCM != 0 {
		a.encCM.Close()
	}
	if a.adpCM != 0 {
		a.adpCM.Close()
	}
	if a.encModel != 0 {
		a.encModel.Close()
	}
	if a.adpModel != 0 {
		a.adpModel.Close()
	}
	if a.opts != 0 {
		a.opts.Close()
	}
}
