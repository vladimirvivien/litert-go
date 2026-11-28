# litert-go

[![CI](https://github.com/vladimirvivien/litert-go/actions/workflows/ci.yml/badge.svg)](https://github.com/vladimirvivien/litert-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/vladimirvivien/litert-go.svg)](https://pkg.go.dev/github.com/vladimirvivien/litert-go)

`litert-go` provides Go bindings for Google's **LiteRT** `CompiledModel` C API targeting general-purpose `.tflite` inference—image classification, object detection, video-frame pipelines, audio tagging and speech frontends, embeddings, and custom graphs—without CGO.

Generative `.litertlm` LLM orchestration lives separately in [litertlm-go](https://github.com/vladimirvivien/litertlm-go).

## Installation

Requires Go `1.26.2+`. Add the module (or `.../pkg/litert`) and tidy after adding package imports:

```sh
go get github.com/vladimirvivien/litert-go@latest
```

## Runtime Baseline

`litert-go` targets official **LiteRT `2.4.1+`** prebuilt shared libraries (`libLiteRt`) loaded dynamically at runtime via `jupiterrider/ffi` and `purego`. Supported desktop platforms include `linux/amd64`, `linux/arm64`, `darwin/arm64`, `windows/amd64`, and `windows/arm64`.

Runtime shared libraries can be resolved automatically with `litert.WithAutoFetch()`, prefetched via the `examples/libfetch` command, or pointed to explicitly via `LITERT_LIB` or `litert.WithLibDir(...)`.

## Quickstart

`litert.Open` loads the shared runtime, creates an `Environment`, opens and compiles a `.tflite` model, and exposes both one-shot named inference (`Engine.Run`) and reusable pre-allocated execution (`Engine.NewRunner`).

### One-Shot Classification

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/tensor"
	"github.com/vladimirvivien/litert-go/pkg/vision"
)

func classify(ctx context.Context, modelPath, imagePath string) error {
	rawImg, err := os.ReadFile(imagePath)
	if err != nil {
		return err
	}
	img, err := vision.DecodeBytes(rawImg)
	if err != nil {
		return err
	}

	eng, err := litert.Open(ctx, modelPath,
		litert.WithAutoFetch(),
		litert.WithAccelerator(litert.AccelCPU),
		litert.WithCPUOptions(litert.CPUOptions{NumThreads: 4}),
	)
	if err != nil {
		return err
	}
	defer func() { _ = eng.Close() }()

	sig := eng.DefaultSignature()
	inSpec := sig.Inputs()[0]
	outSpec := sig.Outputs()[0]

	pixels, _, err := vision.PreprocessUint8(img, vision.Config{
		Width:      int(inSpec.Type.Shape[2]),
		Height:     int(inSpec.Type.Shape[1]),
		ResizeMode: vision.ResizeCenterCrop,
	})
	if err != nil {
		return err
	}

	outputs, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: pixels})
	if err != nil {
		return err
	}

	probs, err := tensor.DequantizeUint8(outputs[outSpec.Name].([]uint8), tensor.QuantParams{
		Scale:     outSpec.Quantization.Scale,
		ZeroPoint: outSpec.Quantization.ZeroPoint,
	})
	if err != nil {
		return err
	}
	for _, top := range vision.ClassifyTopK(probs, nil, 5) {
		fmt.Printf("%s: %.4f\n", top.Label, top.Score)
	}
	return nil
}
```

### Zero-Allocation Streaming Loop

For repeated execution across video frames or sliding audio windows, `Engine.NewRunner` allocates input and output `TensorBuffer` handles once and reuses pre-pinned argument slices on every `runner.Run(ctx)` call:

```go
runner, err := eng.NewRunner()
if err != nil {
	return err
}
defer func() { _ = runner.Close() }()

out := make([]float32, outCount)
for _, frame := range frames {
	if err := litert.WriteSlice(runner.Input(0), frame); err != nil {
		return err
	}
	if err := runner.Run(ctx); err != nil {
		return err
	}
	if err := litert.ReadSliceInto(runner.Output(0), out); err != nil {
		return err
	}
	_ = tensor.Argmax(out)
}
```

For zero-copy host buffers without copying into managed runtime memory, allocate 64-byte-aligned backing storage with `litert.AllocAlignedBytes`, wrap it in `litert.NewHostBuffer`, and bind those buffers directly with `litert.NewRunner`.

## Package Map

| Package | Responsibility |
| :--- | :--- |
| `internal/capi` | Lazy FFI symbol binding, ABI layout codecs, memory pinning helpers |
| `pkg/loader` | Cross-platform shared library discovery (`LITERT_LIB`) and Windows DLL path configuration |
| `pkg/libfetch` | Streaming MD5/SHA-256 verified downloader for LiteRT `2.4.1` prebuilt runtimes and DXC |
| `pkg/litert` | `Engine`, `Environment`, `Options`, `Model`, `CompiledModel`, `TensorBuffer`, `Runner`, typed slices |
| `pkg/tensor` | Shape/stride checks, `uint8`/`int8` quantization, `Softmax`, `Argmax`, `TopK`, `L2Normalize`, `CosineSimilarity`, `NMS` |
| `pkg/vision` | JPEG/PNG decode, raw RGB/RGBA/YCbCr420 video adapters, resize/letterbox/center-crop, Top-K/detection/mask helpers |
| `pkg/audio` | RIFF/WAVE decoding (PCM8/16/24/32 and float32), mono downmix, linear resampling, configurable STFT + log-mel extraction |

## Examples

All sample programs compile with `go build ./...`:

| Command | Description |
| :--- | :--- |
| [`examples/libfetch`](./examples/libfetch/main.go) | Downloads verified LiteRT shared libraries for the current or target platform |
| [`examples/inspect`](./examples/inspect/main.go) | Prints runtime ABI version, signatures, tensor specs, quantization, and metadata |
| [`examples/classify-image`](./examples/classify-image/main.go) | Preprocesses a JPEG/PNG image and outputs Top-K predictions |
| [`examples/detect-objects`](./examples/detect-objects/main.go) | Runs letterboxed object detection with NMS and unletterboxed bounding boxes |
| [`examples/video-frames`](./examples/video-frames/main.go) | Demonstrates repeated streaming inference from raw RGB frames |
| [`examples/classify-audio`](./examples/classify-audio/main.go) | Decodes WAV audio, computes log-mel spectrograms, and classifies events |
| [`examples/embed`](./examples/embed/main.go) | Runs an embedding model twice and computes L2-normalized cosine similarity |
| [`examples/benchmark`](./examples/benchmark/main.go) | Measures warm-up + p50/mean/p95 inference latency and delegation metrics |

## Verification & Benchmarks

Run formatting, static analysis, unit tests with `-race`, and `golangci-lint` locally:

```sh
make all
```

Download LiteRT `2.4.1` shared libraries into a local directory and execute the live runtime integration tests and zero-allocation runner benchmarks:

```sh
go run ./examples/libfetch -dir .cache/litert-2.4.1 -version 2.4.1
LITERT_LIB="$PWD/.cache/litert-2.4.1" go test -race -v ./pkg/litert/...
LITERT_LIB="$PWD/.cache/litert-2.4.1" go test -run='^$' -bench=. -benchmem ./pkg/litert/...
```

## License

Licensed under the [Apache License 2.0](./LICENSE).
