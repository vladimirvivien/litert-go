# litert-go

Go bindings for **LiteRT**, Google's on-device AI runtime — pure Go, no
CGO. The `litert` package runs LiteRT models through the **CompiledModel C
API**; the `lm` package builds a complete LLM runtime on top (chat
templating, bucketed prefill, KV-cache sessions, sampling, speculative
decoding, vision/audio). CPU and GPU (WebGPU) backends.

LLM inference is the initial focus; the direction is the full family of
on-device AI/ML models in Go on the same binding.

The kernels stay inside LiteRT's shared libraries; everything above them —
the orchestration — is Go source that can be read, debugged, and extended
without a C++ toolchain. litert-go is a **library**: every capability is a
programmatic API for embedding in other projects. The `examples/` directory
holds runnable demonstrations.

**Module:** `github.com/vladimirvivien/litert-go` · **Go:** 1.26.2

## Getting the runtime libraries

litert-go implements no compute kernels: all inference executes inside
`libLiteRt` and a platform accelerator library, which the binding loads at
runtime. Putting those libraries on disk is therefore the first step. The
`libfetch` package downloads them from the canonical LiteRT prebuilt releases
(plus, on Windows, the DirectX Shader Compiler the WebGPU accelerator needs):

```go
import "github.com/vladimirvivien/litert-go/libfetch"

dir, err := libfetch.Fetch(ctx) // current platform, default release (2.1.5)
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

The `lm` package is the LLM runtime. An `Engine` owns one model — loaded
from a `.litertlm` container, compiled for one accelerator — and serves all
generation from it: single-shot, streaming (a callback per text piece),
multi-turn conversations that retain the KV cache across turns, and
image/audio prompts. Calls are synchronous and cancel through their context;
an Engine serves one call at a time.

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

out, _ := eng.Generate(ctx, "What is the capital of France?", true /* chat */, lm.GenOptions{MaxTokens: 32})

// Streaming (Generate/Send have token-by-token variants):
_, _ = eng.GenerateStream(ctx, "…", true, lm.GenOptions{MaxTokens: 64}, func(piece string) { fmt.Print(piece) })

// Multi-turn with a KV-reuse session; GenOptions.System steers the assistant:
conv, _ := eng.NewConversation(lm.GenOptions{MaxTokens: 64, Temp: 0.8, TopK: 40,
	System: "You are a terse assistant. Answer in one sentence."})
defer conv.Close()
reply, _ := conv.Send(ctx, "My name is Alice.")
reply, _ = conv.SendStream(ctx, "What is my name?", func(piece string) { fmt.Print(piece) })

// Vision/audio (gemma-4): generate text from a prompt + image/clip; mark the
// position with <start_of_image> / <start_of_audio>:
img, _ := os.ReadFile("photo.jpg")
caption, _ := eng.GenerateFromImage(ctx, "<start_of_image>Describe this image.", img, 0 /* budget */, lm.GenOptions{MaxTokens: 64})
pcm, _ := audio.DecodeWAV(wavBytes) // 16kHz mono
answer, _ := eng.GenerateFromAudio(ctx, "<start_of_audio>What do you hear?", pcm, lm.GenOptions{MaxTokens: 64})
```

Cancelling ctx stops generation between decode steps with `context.Canceled`.
`GenOptions.Spec` enables MTP speculative decoding when the model supports it
(`eng.SupportsSpec()`). Both tokenizer families embedded in `.litertlm`
containers are supported: SentencePiece (gemma, Qwen3-0.6B, Phi-4) and
HuggingFace byte-level BPE (`hftok`; Qwen3-4B).

## Layout

The packages layer bottom-up — the C binding, the container / tokenizer /
preprocessing support packages, and the `lm` runtime on top. Each is usable
on its own.

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
variant at load time.

**Binding constraints.** Crossing into C without CGO means the Go runtime
has no knowledge of the C side; these rules keep the boundary sound:

- The MSVC build of `libLiteRt` lays out `LiteRtRankedTensorType` with
  dimensions at offset 12 (the `rank`/`has_strides` bitfields are not packed);
  the Windows binding reads shapes accordingly.
- Every Go variable whose address reaches the C side must be pinned with
  `runtime.Pinner` for the duration of the call: the goroutine stack can move
  mid-call and the stack mover does not rewrite addresses laundered through
  `unsafe.Pointer`. Scope one `Pinner` per C call. See `litert/ffi.go`.
- `litert.Load` binds one library per process.

## Backends

A model compiles for exactly one accelerator, chosen at `Open` time
(`WithAccelerator`); the backend changes where kernels execute, not the API.

- **CPU** (XNNPACK): every supported model.
- **GPU** (WebGPU — Direct3D 12 on Windows): token- and embedding-input
  models, vision, and audio. Compiled GPU programs and weights persist in a
  cache directory (`WithGPUCacheDir` overrides the location): the first run
  on a model compiles its kernels and is slow; warm runs initialize in
  seconds. The vision adapter always runs on CPU. Speculative decoding on
  GPU returns `ErrSpecUnsupported` for gemma-4-class models.

## Examples

Each example is a thin CLI over the public API. Decode text (greedy unless
`-temp` is set; `-topk` / `-topp` / `-seed` optional):

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

Most limits follow from the substrate: `.litertlm` models ship fully static
graphs, and the runtime is a pinned prebuilt release.

- Context is bounded by the model's KV-cache size (e.g. 4096 for gemma3-1b).
  Prefill covers the prompt by chunking across the model's prefill buckets.
- Models are statically shaped; no KV-cache growth or tensor resize occurs.
- MTP speculative decoding is greedy-only and exact; on CPU the wide verify
  pass costs ~K× a decode step, so the speedup needs GPU.
- The 2.1.5 runtime exposes no log-severity control; C-side INFO lines reach
  stderr.
- Image and audio preprocessing are pure Go (gamma-space resize; FFT mel
  spectrogram); outputs are not bit-matched against a reference
  implementation. `DecodeWAV` handles 16 kHz mono PCM16/float32.

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
