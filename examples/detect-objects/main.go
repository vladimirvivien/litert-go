// Command detect-objects runs an object detection .tflite model on an image
// using letterbox preprocessing, NMS filtering, and coordinate unletterboxing.
package main

import (
	"context"
	"flag"
	"fmt"
	"image/color"
	"os"
	"strings"

	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/tensor"
	"github.com/vladimirvivien/litert-go/pkg/vision"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "detect-objects: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to object detection .tflite model")
	imagePath := flag.String("image", "", "path to PNG or JPEG image")
	labelsPath := flag.String("labels", "", "optional path to newline-separated label file")
	threshold := flag.Float64("threshold", 0.45, "minimum detection score")
	iouThresh := flag.Float64("iou", 0.50, "NMS IoU threshold")
	libDir := flag.String("lib", "", "directory containing libLiteRt shared library")
	flag.Parse()

	if *modelPath == "" || *imagePath == "" {
		return fmt.Errorf("usage: detect-objects -model ssd.tflite -image photo.jpg [-labels labels.txt]")
	}

	rawImg, err := os.ReadFile(*imagePath)
	if err != nil {
		return err
	}
	img, err := vision.DecodeBytes(rawImg)
	if err != nil {
		return err
	}

	opts := []litert.OpenOption{litert.WithLogSeverity(litert.LogWarning)}
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

	cfg := vision.Config{
		Width:      int(inSpec.Type.Shape[2]),
		Height:     int(inSpec.Type.Shape[1]),
		ResizeMode: vision.ResizeLetterbox,
		PadColor:   color.RGBA{R: 114, G: 114, B: 114, A: 255},
	}

	var inVal any
	var lb vision.LetterboxInfo
	switch inSpec.Type.ElementType {
	case litert.ElementUInt8:
		inVal, lb, err = vision.PreprocessUint8(img, cfg)
	case litert.ElementInt8:
		inVal, lb, err = vision.PreprocessInt8(img, cfg)
	default:
		inVal, lb, err = vision.PreprocessFloat32(img, cfg)
	}
	if err != nil {
		return err
	}

	outputs, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inVal})
	if err != nil {
		return err
	}

	outSpecs := sig.Outputs()
	if len(outSpecs) < 3 {
		return fmt.Errorf("expected >= 3 detection output tensors, got %d", len(outSpecs))
	}
	boxes := toFloat32(outputs[outSpecs[0].Name])
	classes := toFloat32(outputs[outSpecs[1].Name])
	scores := toFloat32(outputs[outSpecs[2].Name])
	numDet := len(scores)
	if len(outSpecs) >= 4 {
		if cnt := toFloat32(outputs[outSpecs[3].Name]); len(cnt) > 0 {
			numDet = min(numDet, int(cnt[0]))
		}
	}

	dets, err := vision.DecodeDetections(boxes, classes, scores, numDet, float32(*threshold))
	if err != nil {
		return err
	}
	dets = tensor.NMS(dets, float32(*iouThresh))
	labels := loadLabels(*labelsPath)
	for _, d := range dets {
		box := vision.UnletterboxBox(d.Box, lb)
		label := fmt.Sprintf("class_%d", d.ClassID)
		if d.ClassID >= 0 && d.ClassID < len(labels) && labels[d.ClassID] != "" {
			label = labels[d.ClassID]
		}
		fmt.Printf("%-20s score=%.3f box=[%.3f, %.3f, %.3f, %.3f]\n",
			label, d.Score, box.YMin, box.XMin, box.YMax, box.XMax)
	}
	return nil
}

func toFloat32(raw any) []float32 {
	if s, ok := raw.([]float32); ok {
		return s
	}
	return nil
}

func loadLabels(path string) []string {
	if path == "" {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
}
