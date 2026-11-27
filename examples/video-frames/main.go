// Command video-frames demonstrates streaming raw RGB video frames through a
// pre-allocated litert.Runner for zero-allocation execution.
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"os"
	"time"

	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/vision"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "video-frames: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to 4D NHWC vision .tflite model")
	srcWidth := flag.Int("width", 640, "raw video frame width")
	srcHeight := flag.Int("height", 480, "raw video frame height")
	numFrames := flag.Int("frames", 30, "number of synthetic frames to process")
	libDir := flag.String("lib", "", "directory containing libLiteRt shared library")
	flag.Parse()

	if *modelPath == "" {
		return fmt.Errorf("usage: video-frames -model mobilenet.tflite [-width 640 -height 480 -frames 30]")
	}

	opts := []litert.OpenOption{
		litert.WithCPUOptions(litert.CPUOptions{NumThreads: 4}),
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

	runner, err := eng.NewRunner()
	if err != nil {
		return err
	}
	defer func() { _ = runner.Close() }()

	cfg := vision.Config{
		Width:         int(inSpec.Type.Shape[2]),
		Height:        int(inSpec.Type.Shape[1]),
		Interpolation: vision.InterpBilinear,
	}

	rawFrame := make([]byte, (*srcWidth)*(*srcHeight)*3)
	outSpec := sig.Outputs()[0]
	outElems, _ := outSpec.Type.NumElements()
	outFloat := make([]float32, outElems)

	start := time.Now()
	for frameIdx := range *numFrames {
		fillGradient(rawFrame, *srcWidth, *srcHeight, frameIdx)

		img, frameErr := vision.FromRGB(rawFrame, *srcWidth, *srcHeight, 0)
		if frameErr != nil {
			return fmt.Errorf("frame %d: %w", frameIdx, frameErr)
		}

		if wErr := writePreprocessed(runner.Input(0), img, cfg, inSpec.Type.ElementType); wErr != nil {
			return fmt.Errorf("write frame %d: %w", frameIdx, wErr)
		}
		if rErr := runner.Run(ctx); rErr != nil {
			return fmt.Errorf("run frame %d: %w", frameIdx, rErr)
		}
		if outSpec.Type.ElementType == litert.ElementFloat32 {
			if rdErr := litert.ReadSliceInto(runner.Output(0), outFloat); rdErr != nil {
				return rdErr
			}
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Processed %d frames in %v (%.1f FPS)\n", *numFrames, elapsed.Round(time.Millisecond), float64(*numFrames)/elapsed.Seconds())
	return nil
}

func fillGradient(rgb []byte, w, h, tick int) {
	offset := byte(tick * 7)
	for y := range h {
		row := rgb[y*w*3:]
		for x := range w {
			off := x * 3
			row[off] = byte(x) + offset
			row[off+1] = byte(y)
			row[off+2] = 128
		}
	}
}

func writePreprocessed(buf *litert.TensorBuffer, img image.Image, cfg vision.Config, dtype litert.ElementType) error {
	switch dtype {
	case litert.ElementUInt8:
		data, _, err := vision.PreprocessUint8(img, cfg)
		if err != nil {
			return err
		}
		return litert.WriteSlice(buf, data)
	case litert.ElementInt8:
		data, _, err := vision.PreprocessInt8(img, cfg)
		if err != nil {
			return err
		}
		return litert.WriteSlice(buf, data)
	case litert.ElementFloat32:
		data, _, err := vision.PreprocessFloat32(img, cfg)
		if err != nil {
			return err
		}
		return litert.WriteSlice(buf, data)
	default:
		return fmt.Errorf("unsupported input element type %s", dtype)
	}
}
