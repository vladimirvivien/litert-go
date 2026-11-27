// Command embed runs an embedding or normalization .tflite model twice and
// prints L2-normalized vector statistics and pairwise cosine similarity.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/tensor"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "embed: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to float32 embedding .tflite model")
	libDir := flag.String("lib", "", "directory containing libLiteRt shared library")
	flag.Parse()

	if *modelPath == "" {
		return fmt.Errorf("usage: embed -model model.tflite [-lib dir]")
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
	outSpec := sig.Outputs()[0]
	n, ok := inSpec.Type.NumElements()
	if !ok || n == 0 {
		return fmt.Errorf("expected static float32 input shape, got %v", inSpec.Type.Shape)
	}

	vecA := make([]float32, n)
	vecB := make([]float32, n)
	for i := range n {
		vecA[i] = float32(i%13) * 0.1
		vecB[i] = float32((n-i)%13) * 0.1
	}

	outA, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: vecA})
	if err != nil {
		return err
	}
	outB, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: vecB})
	if err != nil {
		return err
	}

	embA := tensor.L2Normalize(outA[outSpec.Name].([]float32))
	embB := tensor.L2Normalize(outB[outSpec.Name].([]float32))
	sim, err := tensor.CosineSimilarity(embA, embB)
	if err != nil {
		return err
	}
	fmt.Printf("Embedding dim=%d, CosineSimilarity(A, B) = %.6f\n", len(embA), sim)
	return nil
}
