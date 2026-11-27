// Command classify-image preprocesses an image file, runs a .tflite image
// classification model, and prints the top-K predicted classes.
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"os"
	"strings"

	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/tensor"
	"github.com/vladimirvivien/litert-go/pkg/vision"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "classify-image: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to image classification .tflite model")
	imagePath := flag.String("image", "", "path to PNG or JPEG image")
	labelsPath := flag.String("labels", "", "optional path to newline-separated label file")
	topK := flag.Int("topk", 5, "number of top predictions to display")
	threads := flag.Int("threads", 4, "CPU thread count")
	libDir := flag.String("lib", "", "directory containing libLiteRt shared library")
	flag.Parse()

	if *modelPath == "" || *imagePath == "" {
		return fmt.Errorf("usage: classify-image -model model.tflite -image photo.jpg [-labels labels.txt]")
	}

	labels := loadLabels(*labelsPath)
	rawImg, err := os.ReadFile(*imagePath)
	if err != nil {
		return err
	}
	img, err := vision.DecodeBytes(rawImg)
	if err != nil {
		return err
	}

	opts := []litert.OpenOption{
		litert.WithCPUOptions(litert.CPUOptions{NumThreads: *threads}),
		litert.WithLogSeverity(litert.LogWarning),
	}
	if *libDir != "" {
		opts = append(opts, litert.WithLibDir(*libDir))
	} else {
		opts = append(opts, litert.WithAutoFetch())
	}

	ctx := context.Background()
	eng, err := litert.Open(ctx, *modelPath, opts...)
	if err != nil {
		return err
	}
	defer func() { _ = eng.Close() }()

	sig := eng.DefaultSignature()
	inSpec := sig.Inputs()[0]
	if len(inSpec.Type.Shape) < 4 {
		return fmt.Errorf("expected NHWC 4D input shape, got %v", inSpec.Type.Shape)
	}
	height, width := int(inSpec.Type.Shape[1]), int(inSpec.Type.Shape[2])
	inVal, err := preprocessInput(img, width, height, inSpec)
	if err != nil {
		return err
	}

	outputs, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inVal})
	if err != nil {
		return err
	}

	outSpec := sig.Outputs()[0]
	scores, err := decodeScores(outputs[outSpec.Name], outSpec)
	if err != nil {
		return err
	}
	for _, cls := range vision.ClassifyTopK(scores, labels, *topK) {
		fmt.Printf("[%d] %-30s %.4f\n", cls.Index, cls.Label, cls.Score)
	}
	return nil
}

func preprocessInput(img image.Image, w, h int, sp litert.TensorSpec) (any, error) {
	cfg := vision.Config{
		Width:      w,
		Height:     h,
		ResizeMode: vision.ResizeCenterCrop,
	}
	switch sp.Type.ElementType {
	case litert.ElementUInt8:
		data, _, err := vision.PreprocessUint8(img, cfg)
		return data, err
	case litert.ElementInt8:
		data, _, err := vision.PreprocessInt8(img, cfg)
		return data, err
	case litert.ElementFloat32:
		data, _, err := vision.PreprocessFloat32(img, cfg)
		return data, err
	default:
		return nil, fmt.Errorf("unsupported input element type %s", sp.Type.ElementType)
	}
}

func decodeScores(raw any, sp litert.TensorSpec) ([]float32, error) {
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
		return nil, fmt.Errorf("unexpected output slice type %T", raw)
	}
}

func loadLabels(path string) []string {
	if path == "" {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	lines := strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
	for i := range lines {
		lines[i] = strings.TrimSpace(lines[i])
	}
	return lines
}
