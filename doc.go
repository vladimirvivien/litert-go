// Package litertgo is the module root for litert-go, Go bindings for the
// LiteRT CompiledModel C API targeting general-purpose .tflite inference without
// CGO.
//
// Core runtime, loader, and helper packages live under pkg/:
//
//   - pkg/litert: Engine, Environment, Options, Model, CompiledModel, TensorBuffer, Runner
//   - pkg/libfetch: Checksum-verified downloader for LiteRT prebuilt shared libraries
//   - pkg/loader: Dynamic shared-library discovery and OS DLL path setup
//   - pkg/tensor: Shape/stride helpers, quantization, Softmax, TopK, L2Normalize, NMS
//   - pkg/vision: Image and raw video-frame preprocessing and postprocessing helpers
//   - pkg/audio: RIFF/WAVE decoding, resampling, and configurable log-mel spectrograms
//
// See README.md for usage examples and supported platforms.
package litertgo
