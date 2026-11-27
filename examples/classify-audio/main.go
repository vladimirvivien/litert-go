// Command classify-audio decodes a WAV recording, prepares mono 16 kHz audio
// or log-mel spectrogram features, runs a .tflite audio classifier, and prints
// the top-K predicted classes.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litert-go/pkg/audio"
	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/tensor"
	"github.com/vladimirvivien/litert-go/pkg/vision"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "classify-audio: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to audio .tflite model")
	wavPath := flag.String("wav", "", "path to RIFF/WAVE audio file")
	labelsPath := flag.String("labels", "", "optional path to class label file")
	topK := flag.Int("topk", 5, "number of top predictions")
	sampleRate := flag.Int("rate", 16000, "target sample rate (Hz)")
	libDir := flag.String("lib", "", "directory containing libLiteRt shared library")
	flag.Parse()

	if *modelPath == "" || *wavPath == "" {
		return fmt.Errorf("usage: classify-audio -model yamnet.tflite -wav sound.wav [-labels labels.txt]")
	}

	rawWAV, err := os.ReadFile(*wavPath)
	if err != nil {
		return err
	}
	pcm, err := audio.DecodeWAVBytes(rawWAV)
	if err != nil {
		return err
	}
	mono := audio.PreparePCM(pcm, *sampleRate)

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

	var features []float32
	if len(inSpec.Type.Shape) >= 2 && inSpec.Type.Shape[len(inSpec.Type.Shape)-1] == 64 {
		feats, _, melErr := audio.MelSpectrogram(mono, audio.DefaultYAMNetConfig())
		if melErr != nil {
			return melErr
		}
		features = feats
	} else {
		features = mono
	}

	if want, ok := inSpec.Type.NumElements(); ok && len(features) != want {
		padded := make([]float32, want)
		copy(padded, features)
		features = padded
	}

	outputs, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: features})
	if err != nil {
		return err
	}

	outSpec := sig.Outputs()[0]
	scores, _ := outputs[outSpec.Name].([]float32)
	labels := loadLabels(*labelsPath)
	for _, cls := range vision.ClassifyTopK(tensor.Softmax(scores), labels, *topK) {
		fmt.Printf("[%d] %-30s %.4f\n", cls.Index, cls.Label, cls.Score)
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
