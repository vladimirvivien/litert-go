//go:build integration

package litert_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/vladimirvivien/litert-go/pkg/audio"
	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/tensor"
	"github.com/vladimirvivien/litert-go/pkg/vision"
)

func requireRealModel(t *testing.T, name string) string {
	t.Helper()
	requireLiveRuntime(t)

	dir := os.Getenv("LITERT_TEST_MODELS_DIR")
	if dir == "" {
		t.Skip("set LITERT_TEST_MODELS_DIR to run real-model integration tests")
	}
	p := filepath.Join(dir, name)
	if _, err := os.Stat(p); err != nil {
		if os.IsNotExist(err) {
			t.Skipf("model fixture %q not found in %s", name, dir)
		}
		t.Fatalf("stat %s: %v", p, err)
	}
	return p
}

func acceleratorConfigs(t *testing.T) []struct {
	name string
	opts []litert.OpenOption
} {
	t.Helper()
	cacheDir := t.TempDir()
	return []struct {
		name string
		opts []litert.OpenOption
	}{
		{
			name: "CPU",
			opts: []litert.OpenOption{
				litert.WithAccelerator(litert.AccelCPU),
				litert.WithCPUOptions(litert.CPUOptions{NumThreads: 4}),
				litert.WithLogSeverity(litert.LogWarning),
			},
		},
		{
			name: "GPU+CPU",
			opts: []litert.OpenOption{
				litert.WithAccelerator(litert.AccelGPU | litert.AccelCPU),
				litert.WithCPUOptions(litert.CPUOptions{NumThreads: 2}),
				litert.WithGPUOptions(litert.GPUOptions{
					SerializationDir:      cacheDir,
					ModelCacheKey:         "litert-go-integration",
					SerializeProgramCache: new(true),
				}),
				litert.WithLogSeverity(litert.LogWarning),
			},
		},
	}
}

func logEngineMetrics(t *testing.T, prefix string, eng *litert.Engine) {
	t.Helper()
	cm := eng.CompiledModel()
	full, fErr := cm.FullyAccelerated()
	nonCPU, nErr := cm.NonCPUFullyAccelerated()
	dur, dErr := cm.LastInferenceDuration()
	dm, mErr := cm.DelegationMetrics()
	t.Logf("%s: fullyAccelerated=%v(%v) nonCPU=%v(%v) lastDur=%v(%v) nodes=%d accels=%+v(%v)",
		prefix, full, fErr, nonCPU, nErr, dur, dErr, dm.TotalNodeCount, dm.Accelerators, mErr)
}

func makePatternImage(w, h int, mode string) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			var c color.RGBA
			switch mode {
			case "solid":
				c = color.RGBA{R: 220, G: 40, B: 30, A: 255}
			default:
				if ((x/16)+(y/16))%2 == 0 {
					c = color.RGBA{R: 240, G: 220, B: 40, A: 255}
				} else {
					c = color.RGBA{R: 20, G: 60, B: 200, A: 255}
				}
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

func preprocessForSpec(img image.Image, sp litert.TensorSpec, mode vision.ResizeMode) (any, vision.LetterboxInfo, error) {
	cfg := vision.Config{
		Width:      int(sp.Type.Shape[2]),
		Height:     int(sp.Type.Shape[1]),
		ResizeMode: mode,
	}
	switch sp.Type.ElementType {
	case litert.ElementUInt8:
		return vision.PreprocessUint8(img, cfg)
	case litert.ElementInt8:
		return vision.PreprocessInt8(img, cfg)
	default:
		return vision.PreprocessFloat32(img, cfg)
	}
}

func decodeClassProbabilities(raw any, sp litert.TensorSpec) ([]float32, error) {
	q := tensor.QuantParams{Scale: sp.Quantization.Scale, ZeroPoint: sp.Quantization.ZeroPoint}
	switch v := raw.(type) {
	case []float32:
		return tensor.Softmax(v), nil
	case []uint8:
		if q.Scale > 0 {
			return tensor.DequantizeUint8(v, q)
		}
		out := make([]float32, len(v))
		for i, b := range v {
			out[i] = float32(b) / 255.0
		}
		return out, nil
	case []int8:
		if q.Scale > 0 {
			return tensor.DequantizeInt8(v, q)
		}
		out := make([]float32, len(v))
		for i, s := range v {
			out[i] = float32(s)
		}
		return tensor.Softmax(out), nil
	default:
		return nil, litert.ErrInvalidArgument
	}
}

func TestRealModelsClassification(t *testing.T) {
	models := []struct {
		file   string
		minCls int
		useRun bool
	}{
		{file: "mobilenet_v2_1.0_224.tflite", minCls: 1000, useRun: true},
		{file: "efficientnet_lite0.tflite", minCls: 1000},
	}

	ctx := context.Background()
	img := makePatternImage(320, 240, "checker")

	for _, mc := range models {
		t.Run(mc.file, func(t *testing.T) {
			modelPath := requireRealModel(t, mc.file)
			for _, ac := range acceleratorConfigs(t) {
				t.Run(ac.name, func(t *testing.T) {
					eng, err := litert.Open(ctx, modelPath, ac.opts...)
					if err != nil {
						t.Fatalf("Open(%s): %v", ac.name, err)
					}
					defer func() { _ = eng.Close() }()

					sig := eng.DefaultSignature()
					inSpec := sig.Inputs()[0]
					outSpec := sig.Outputs()[0]

					inVal, _, prepErr := preprocessForSpec(img, inSpec, vision.ResizeCenterCrop)
					if prepErr != nil {
						t.Fatalf("preprocess: %v", prepErr)
					}

					outputs, runErr := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inVal})
					if runErr != nil {
						t.Fatalf("Engine.Run: %v", runErr)
					}
					probs, decErr := decodeClassProbabilities(outputs[outSpec.Name], outSpec)
					if decErr != nil {
						t.Fatalf("decode: %v", decErr)
					}
					if len(probs) < mc.minCls {
						t.Fatalf("classes = %d, want >= %d", len(probs), mc.minCls)
					}
					top5 := vision.ClassifyTopK(probs, nil, 5)
					if len(top5) != 5 || math.IsNaN(float64(top5[0].Score)) || top5[0].Score <= 0 {
						t.Fatalf("unexpected TopK: %+v", top5)
					}
					logEngineMetrics(t, mc.file+"/"+ac.name, eng)

					if mc.useRun {
						runner, rErr := eng.NewRunner()
						if rErr != nil {
							t.Fatalf("NewRunner: %v", rErr)
						}
						defer func() { _ = runner.Close() }()

						if fSlice, ok := inVal.([]float32); ok {
							if wErr := litert.WriteSlice(runner.Input(0), fSlice); wErr != nil {
								t.Fatalf("WriteSlice: %v", wErr)
							}
						}
						if rrErr := runner.Run(ctx); rrErr != nil {
							t.Fatalf("runner.Run: %v", rrErr)
						}
						got, rdErr := litert.ReadSlice[float32](runner.Output(0))
						if rdErr != nil || len(got) != len(probs) {
							t.Fatalf("ReadSlice: len=%d err=%v", len(got), rdErr)
						}
					}
				})
			}
		})
	}
}

func TestRealModelsDetection(t *testing.T) {
	modelPath := requireRealModel(t, "efficientdet_lite0.tflite")
	ctx := context.Background()
	img := makePatternImage(640, 480, "checker")

	for _, ac := range acceleratorConfigs(t) {
		t.Run(ac.name, func(t *testing.T) {
			eng, err := litert.Open(ctx, modelPath, ac.opts...)
			if err != nil {
				t.Fatalf("Open(%s): %v", ac.name, err)
			}
			defer func() { _ = eng.Close() }()

			sig := eng.DefaultSignature()
			inSpec := sig.Inputs()[0]
			inVal, lb, err := preprocessForSpec(img, inSpec, vision.ResizeLetterbox)
			if err != nil {
				t.Fatalf("preprocess: %v", err)
			}

			outputs, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inVal})
			if err != nil {
				t.Fatalf("Engine.Run: %v", err)
			}

			outSpecs := sig.Outputs()
			if len(outSpecs) == 2 {
				rawScores, okS := outputs[outSpecs[0].Name].([]float32)
				rawBoxes, okB := outputs[outSpecs[1].Name].([]float32)
				if !okS || !okB {
					t.Fatalf("unexpected raw detection output types %T / %T", outputs[outSpecs[0].Name], outputs[outSpecs[1].Name])
				}
				anchors := int(outSpecs[1].Type.Shape[1])
				classesCount := int(outSpecs[0].Type.Shape[2])
				if len(rawBoxes) != anchors*4 || len(rawScores) != anchors*classesCount {
					t.Fatalf("unexpected raw shapes boxes=%d scores=%d", len(rawBoxes), len(rawScores))
				}

				// Extract top class per anchor to exercise DecodeDetections + NMS + UnletterboxBox.
				clsIDs := make([]float32, anchors)
				bestScores := make([]float32, anchors)
				for a := range anchors {
					base := a * classesCount
					bestC := tensor.Argmax(rawScores[base : base+classesCount])
					clsIDs[a] = float32(bestC)
					bestScores[a] = rawScores[base+bestC]
				}
				dets, dErr := vision.DecodeDetections(rawBoxes, clsIDs, bestScores, anchors, -100)
				if dErr != nil {
					t.Fatalf("DecodeDetections: %v", dErr)
				}
				suppressed := tensor.NMS(dets[:min(len(dets), 128)], 0.5)
				for _, d := range suppressed {
					_ = vision.UnletterboxBox(d.Box, lb)
				}
			}
			logEngineMetrics(t, "efficientdet_lite0/"+ac.name, eng)
		})
	}
}

func TestRealModelsAudioYAMNet(t *testing.T) {
	modelPath := requireRealModel(t, "yamnet.tflite")
	ctx := context.Background()

	// Synthesize a 1-second 440 Hz stereo 16-bit WAV at 44.1 kHz to test DecodeWAVBytes + PreparePCM.
	const srcRate = 44100
	var pcmData bytes.Buffer
	for i := range srcRate {
		sample := int16(12000 * math.Sin(2*math.Pi*440*float64(i)/float64(srcRate)))
		_ = binary.Write(&pcmData, binary.LittleEndian, sample)
		_ = binary.Write(&pcmData, binary.LittleEndian, sample)
	}
	wavBytes := buildSyntheticStereoWAV(srcRate, pcmData.Bytes())
	pcm, err := audio.DecodeWAVBytes(wavBytes)
	if err != nil {
		t.Fatalf("DecodeWAVBytes: %v", err)
	}
	mono := audio.PreparePCM(pcm, 16000)

	// YAMNet embeds STFT/COMPLEX64/RFFT2D ops unsupported on LiteRT 2.4.1 partial WebGPU delegation.
	eng, openErr := litert.Open(ctx, modelPath,
		litert.WithAccelerator(litert.AccelCPU),
		litert.WithCPUOptions(litert.CPUOptions{NumThreads: 4}),
		litert.WithLogSeverity(litert.LogWarning),
	)
	if openErr != nil {
		t.Fatalf("Open(CPU): %v", openErr)
	}
	defer func() { _ = eng.Close() }()

	sig := eng.DefaultSignature()
	inSpec := sig.Inputs()[0]
	outSpec := sig.Outputs()[0]

	inCount, ok := inSpec.Type.NumElements()
	if !ok || inCount == 0 {
		t.Fatalf("unexpected yamnet input shape %v", inSpec.Type.Shape)
	}
	input := make([]float32, inCount)
	copy(input, mono)

	outputs, runErr := eng.Run(ctx, litert.NamedInputs{inSpec.Name: input})
	if runErr != nil {
		t.Fatalf("Engine.Run: %v", runErr)
	}
	scores, ok := outputs[outSpec.Name].([]float32)
	if !ok || len(scores) < 521 {
		t.Fatalf("unexpected yamnet output %T len=%d", outputs[outSpec.Name], len(scores))
	}
	top5 := vision.ClassifyTopK(scores[:521], nil, 5)
	if len(top5) != 5 || math.IsNaN(float64(top5[0].Score)) {
		t.Fatalf("unexpected yamnet TopK: %+v", top5)
	}
	logEngineMetrics(t, "yamnet/CPU", eng)
}

func TestRealModelsImageEmbedder(t *testing.T) {
	modelPath := requireRealModel(t, "mobilenet_v3_small_embedder.tflite")
	ctx := context.Background()

	imgA := makePatternImage(256, 256, "checker")
	imgB := makePatternImage(256, 256, "solid")

	for _, ac := range acceleratorConfigs(t) {
		t.Run(ac.name, func(t *testing.T) {
			eng, err := litert.Open(ctx, modelPath, ac.opts...)
			if err != nil {
				t.Fatalf("Open(%s): %v", ac.name, err)
			}
			defer func() { _ = eng.Close() }()

			sig := eng.DefaultSignature()
			inSpec := sig.Inputs()[0]
			outSpec := sig.Outputs()[0]

			inA, _, err := preprocessForSpec(imgA, inSpec, vision.ResizeCenterCrop)
			if err != nil {
				t.Fatal(err)
			}
			inB, _, err := preprocessForSpec(imgB, inSpec, vision.ResizeCenterCrop)
			if err != nil {
				t.Fatal(err)
			}

			outA, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inA})
			if err != nil {
				t.Fatal(err)
			}
			outB, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inB})
			if err != nil {
				t.Fatal(err)
			}

			embA := tensor.L2Normalize(outA[outSpec.Name].([]float32))
			embB := tensor.L2Normalize(outB[outSpec.Name].([]float32))

			selfSim, err := tensor.CosineSimilarity(embA, embA)
			if err != nil || selfSim < 0.999 {
				t.Fatalf("self CosineSimilarity = %v, %v", selfSim, err)
			}
			crossSim, err := tensor.CosineSimilarity(embA, embB)
			if err != nil || crossSim >= 0.999 {
				t.Fatalf("expected distinct images cross similarity < 0.999, got %v (%v)", crossSim, err)
			}
			logEngineMetrics(t, "embedder/"+ac.name, eng)
		})
	}
}

func buildSyntheticStereoWAV(sampleRate int, pcmPayload []byte) []byte {
	var body bytes.Buffer
	body.WriteString("WAVEfmt ")
	_ = binary.Write(&body, binary.LittleEndian, uint32(16))
	_ = binary.Write(&body, binary.LittleEndian, uint16(1))
	_ = binary.Write(&body, binary.LittleEndian, uint16(2))
	_ = binary.Write(&body, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(&body, binary.LittleEndian, uint32(sampleRate*4))
	_ = binary.Write(&body, binary.LittleEndian, uint16(4))
	_ = binary.Write(&body, binary.LittleEndian, uint16(16))
	body.WriteString("data")
	_ = binary.Write(&body, binary.LittleEndian, uint32(len(pcmPayload)))
	body.Write(pcmPayload)

	var riff bytes.Buffer
	riff.WriteString("RIFF")
	_ = binary.Write(&riff, binary.LittleEndian, uint32(body.Len()))
	riff.Write(body.Bytes())
	return riff.Bytes()
}
