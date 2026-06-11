# litert-go

A pure-Go (no CGO) library for on-device LLM inference, built directly on the
**LiteRT CompiledModel C API**: a `purego`-based binding plus a Go-authored
LLM runtime (chat templating, bucketed prefill, KV-cache sessions, sampling,
speculative decoding, vision/audio) on top. CPU and GPU (WebGPU) backends.

litert-go is a **library**: every capability is a programmatic API intended
for embedding in other projects. The `examples/` directory holds runnable
demonstrations; nothing lives only behind a flag.

**Module:** `github.com/vladimirvivien/litert-go` · **Go:** 1.26.2

## Getting the runtime libraries

The binding loads `libLiteRt` and a platform accelerator at runtime. The
`libfetch` package downloads them from the canonical LiteRT prebuilt releases
(plus, on Windows, the DirectX Shader Compiler the WebGPU accelerator needs):

```go
import "github.com/vladimirvivien/litert-go/libfetch"

dir, err := libfetch.Fetch(ctx) // current platform, validated release
```

or as a one-liner via the example CLI:

```
LITERT_LIB=$(go run github.com/vladimirvivien/litert-go/examples/libfetch)
```

Fetches are checksum-verified and idempotent. A self-built `libLiteRt` from
`google-ai-edge/litert` works too — point `LITERT_LIB` (or the lib-dir
argument) at its directory. The runtime and accelerator must come from the
same release: a runtime rejects an accelerator built from different sources.

## Library

```go
import (
	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/lm"
)

// Runtime libraries resolve from WithLibDir, then LITERT_LIB, then libfetch's
// default download location; WithFetch("2.1.5") opts in to downloading them.
eng, err := lm.Open(ctx, "gemma3-1b-it-int4.litertlm",
	lm.WithAccelerator(litert.AccelGPU),
	lm.WithMetrics(func(s lm.DecodeStats) { log.Printf("%.1f tok/s", s.TokensPerSecond()) }))
defer eng.Close()

out, _ := eng.Generate("What is the capital of France?", true /* chat */, lm.GenOptions{MaxTokens: 32})

// Streaming (Generate/Send have token-by-token variants):
_, _ = eng.GenerateStream("…", true, lm.GenOptions{MaxTokens: 64}, func(piece string) { fmt.Print(piece) })

// Multi-turn with a KV-reuse session; GenOptions.System steers the assistant:
conv, _ := eng.NewConversation(lm.GenOptions{MaxTokens: 64, Temp: 0.8, TopK: 40,
	System: "You are a terse assistant. Answer in one sentence."})
defer conv.Close()
reply, _ := conv.Send("My name is Vlad.")
reply, _ = conv.SendStream("What is my name?", func(piece string) { fmt.Print(piece) })

// Vision/audio (gemma-4): generate text from a prompt + image/clip; mark the
// position with <start_of_image> / <start_of_audio>:
img, _ := os.ReadFile("photo.jpg")
caption, _ := eng.GenerateFromImage("<start_of_image>Describe this image.", img, 0 /* budget */, lm.GenOptions{MaxTokens: 64})
pcm, _ := audio.DecodeWAV(wavBytes) // 16kHz mono
answer, _ := eng.GenerateFromAudio("<start_of_audio>What do you hear?", pcm, lm.GenOptions{MaxTokens: 64})
```

`GenOptions.Spec` enables MTP speculative decoding when the model supports it
(`eng.SupportsSpec()`). Both tokenizer families embedded in `.litertlm`
containers are supported: SentencePiece (including HF-converted raw-space
vocabs — Qwen3-0.6B, Phi-4) and HuggingFace byte-level BPE (`hftok`).

## Layout

```
litert/        no-CGO binding to the LiteRT C API (purego + jupiterrider/ffi)
  ffi.go         library loading, lazy symbol resolution, runtime.Pinner convention
  bindings.go    one lazyFun per bound C entry point
  litert.go      typed wrappers + enums: Environment, Options, Model, Signature,
                 CompiledModel, TensorBuffer
  run.go         Runner — repeated Run over a fixed buffer set, arguments pinned once
libfetch/      runtime-library acquisition from the LiteRT prebuilt releases
litertlm/      .litertlm container reader (minimal FlatBuffer parser)
  litertlm.go    Sections (+ model_type hints) / SectionTFLite / SectionBytes
  metadata.go    ReadMetadata — model family + max tokens (protobuf scan)
hftok/         pure-Go HuggingFace byte-level BPE tokenizer (Qwen, GPT-2 family)
vision/        pure-Go image preprocessing (decode, aspect-resize, patchify)
audio/         pure-Go audio preprocessing (WAV decode, mel spectrogram, FFT)
lm/            LLM runtime: Open / Generate / NewConversation / GenerateFrom{Image,Audio}
  engine.go      Engine, tokenizer abstraction, chat templating, token-input decode
  embed.go       embedding-input pipeline (gemma 3n/4: dual embedders, i8 KV)
  multimodal.go  shared vision/audio embedding splice (sentinel + markers)
  vision.go      gemma-4 vision: encoder (engine backend) + adapter (CPU)
  audio.go       gemma-4 audio: encoder + adapter
  spec.go        MTP speculative decoding (drafter + verify)
  sample.go      temperature / top-k / top-p sampling
examples/      runnable demos of the library API
  decode/        full pipeline CLI (-text / -repl, -chat, -image, -audio, -spec, sampling)
  libfetch/      runtime-library download CLI
  siginfo/       dump a model's sections and signature prefill shapes
```

The binding targets the canonical LiteRT C API in `google-ai-edge/litert`.
The 2.1.5 release's `LiteRtCreateModelFromBuffer` takes no leading
`environment` argument while newer runtimes add one; the binding detects the
variant at load time, so both work.

**Binding constraints:**

- The MSVC build of `libLiteRt` lays out `LiteRtRankedTensorType` with
  dimensions at offset 12 (the `rank`/`has_strides` bitfields are not packed);
  the Windows binding reads shapes accordingly.
- Every Go variable whose address reaches the C side must be pinned with
  `runtime.Pinner` for the duration of the call: the goroutine stack can move
  mid-call and the stack mover does not rewrite addresses laundered through
  `unsafe.Pointer`. Scope one `Pinner` per C call. See `litert/ffi.go`.
- `litert.Load` binds one library per process.

## Backends

- **CPU** (XNNPACK): every supported model.
- **GPU** (WebGPU — Direct3D 12 on Windows): token- and embedding-input
  models, vision, and audio. The engine applies the GPU configuration the
  C++ LiteRT-LM engine uses: a compact program/weight cache (warm starts in
  seconds), native-layout KV with double-buffered banks, async submission,
  and the single-buffer KV mode gemma-4-class (param_tensor) models require.
  The vision adapter always runs on CPU. Speculative decoding on GPU is not
  supported for param_tensor models.

## Examples

Decode text (greedy unless `-temp` is set; `-topk` / `-topp` / `-seed`
optional):

```
go run ./examples/decode -lib $LITERT_LIB -model model.litertlm -backend gpu \
    -chat -text "What is the capital of France?"
```

`-chat` wraps the prompt in the model's chat template (from the container's
`LlmMetadata`); `-repl` is interactive multi-turn chat over a KV-reuse
session; `-image` / `-audio` run the gemma-4 multimodal paths; `-spec`
enables MTP speculative decoding (prints tokens/verify-pass to stderr).

Dump a model's signatures and prefill bucket shapes:

```
go run ./examples/siginfo -lib $LITERT_LIB -model model.litertlm
```

## Limitations

- Context is bounded by the model's KV-cache size (e.g. 4096 for gemma3-1b).
  Prefill covers the prompt by chunking across the model's prefill buckets.
- Models are statically shaped; no KV-cache growth or tensor resize occurs.
- MTP speculative decoding is greedy-only and exact; on CPU the wide verify
  pass costs ~K× a decode step, so the speedup needs GPU.
- The 2.1.5 runtime exposes no log-severity control; C-side INFO lines reach
  stderr.
- Image and audio preprocessing are pure Go (gamma-space resize; hand-rolled
  FFT mel spectrogram); validated qualitatively. `DecodeWAV` handles 16 kHz
  mono PCM16/float32.

## Build & test

```
go build ./...
go vet ./...
go test ./...
```

Unit tests run without hardware. The `lm` package has env-gated integration
tests against real models:

```
LITERT_LIB=/abs/lib LITERT_LM_MODEL=/abs/model.litertlm go test ./lm
LITERT_LIB=/abs/lib LITERT_LM_EMBED_MODEL=/abs/gemma-4.litertlm go test ./lm
```

`TestModelMatrix` runs a four-check battery (greedy generate, stream ≡
generate, tokenize round-trip, multi-turn) over a directory of models:

```
LITERT_LIB=/abs/lib LITERT_LM_MODELS=/abs/models LITERT_LM_BACKEND=gpu go test ./lm -run TestModelMatrix
```

(`run-matrix.sh` drives it one process per model.) The `litert` ffi-call
benchmarks gate on `LITERT_LIB` + `LITERT_BENCH_MODEL`; the `litertlm`
reader benchmarks gate on `LITERTLM_BENCH_FILE`.
