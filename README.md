# litert-go

Go bindings for Google's **LiteRT** `CompiledModel` C API targeting general-purpose (non-LLM) `.tflite` inference—computer vision, video frames, audio classification and signal processing, dense embeddings, and custom graphs—without CGO.

Generative `.litertlm` LLM orchestration is provided separately by [litertlm-go](https://github.com/vladimirvivien/litertlm-go).

## Runtime Baseline

`litert-go` targets **LiteRT `2.4.1+`** prebuilt shared libraries (`libLiteRt`) loaded dynamically at runtime via `jupiterrider/ffi` and `purego`. Supported desktop platforms include `linux/amd64`, `linux/arm64`, `darwin/arm64`, `windows/amd64`, and `windows/arm64`.

## Package Layout

The module separates internal FFI symbol marshalling from public runtime and domain helper packages under `pkg/`:

```text
github.com/vladimirvivien/litert-go
├── internal/capi/     # Low-level FFI symbol resolution and marshalling
├── pkg/loader/        # Native shared-library discovery and OS DLL path setup
├── pkg/libfetch/      # Checksum-verified LiteRT runtime downloader
├── pkg/litert/        # Environment, Options, Model, CompiledModel, TensorBuffer, Runner
├── pkg/tensor/        # Quantization, Softmax, Argmax, TopK, L2Normalize, CosineSimilarity, NMS
├── pkg/vision/        # Image and raw video-frame preprocessing and postprocessing helpers
└── pkg/audio/         # WAV decoding, resampling, STFT, and mel-spectrogram helpers
```

## Development & Verification

Run the standard formatting, build, race-test, and lint checks locally with `make` or standard `go` commands:

```sh
make all
```

Individual targets match CI checks:

```sh
go fix ./...
go fmt ./...
go build ./...
go vet ./...
go test -race ./...
golangci-lint run ./...
```

## License

Licensed under the [Apache License 2.0](./LICENSE).
