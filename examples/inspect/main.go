// Command inspect prints LiteRT runtime ABI, model signatures, tensor shapes,
// element types, quantization parameters, and optional metadata entries.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litert-go/pkg/litert"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "inspect: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to .tflite model file")
	libDir := flag.String("lib", "", "directory containing libLiteRt shared library")
	metaKey := flag.String("meta", "", "optional model metadata key to inspect")
	flag.Parse()

	if *modelPath == "" {
		return fmt.Errorf("usage: inspect -model path/to/model.tflite [-lib dir] [-meta key]")
	}

	opts := []litert.OpenOption{
		litert.WithLogSeverity(litert.LogWarning),
	}
	if *libDir != "" {
		opts = append(opts, litert.WithLibDir(*libDir))
	} else {
		opts = append(opts, litert.WithAutoFetch())
	}

	eng, err := litert.Open(context.Background(), *modelPath, opts...)
	if err != nil {
		return err
	}
	defer func() { _ = eng.Close() }()

	if abi, abiErr := litert.RuntimeABIVersion(); abiErr == nil {
		fmt.Printf("LiteRT ABI: %s\n", abi)
	}
	fmt.Printf("Model: %s\n", *modelPath)

	sigs := eng.Signatures()
	if len(sigs) == 0 {
		sigs = []*litert.Signature{eng.DefaultSignature()}
	}
	for _, sig := range sigs {
		fmt.Printf("\nSignature [%d] %q:\n", sig.Index(), sig.Key())
		for i, in := range sig.Inputs() {
			printTensor("  input", i, in)
		}
		for i, out := range sig.Outputs() {
			printTensor("  output", i, out)
		}
	}

	if *metaKey != "" {
		data, mErr := eng.Metadata(*metaKey)
		if mErr != nil {
			return fmt.Errorf("metadata %q: %w", *metaKey, mErr)
		}
		fmt.Printf("\nMetadata[%q] (%d bytes):\n%s\n", *metaKey, len(data), string(data))
	}
	return nil
}

func printTensor(kind string, idx int, sp litert.TensorSpec) {
	fmt.Printf("%s[%d] %q dtype=%s shape=%v", kind, idx, sp.Name, sp.Type.ElementType, sp.Type.Shape)
	switch sp.Quantization.Type {
	case litert.QuantPerTensor:
		fmt.Printf(" quant=(scale=%g, zp=%d)", sp.Quantization.Scale, sp.Quantization.ZeroPoint)
	case litert.QuantPerChannel:
		fmt.Printf(" quant=per_channel(dim=%d, channels=%d)", sp.Quantization.QuantizedDim, len(sp.Quantization.Scales))
	}
	fmt.Println()
}
