# litert-go

A no-CGO Go binding for the **LiteRT CompiledModel C API**, and a runtime that
drives LLM inference (prefill/decode, KV cache, sampling, speculative decoding)
from Go. Exploratory — see the planning docs in
`~/DEV/project-planning/litert-go/`.

**Status:** Working text-in/text-out LLM inference — pure Go, no CGO. Against an
upstream-built `libLiteRt`, `cmd/decode` runs the full pipeline on CPU: text →
SentencePiece (pure-Go `eliben/go-sentencepiece`, loaded from the `.litertlm`'s
embedded tokenizer) → prefill → greedy decode → detokenize → text. Verified
correct on Gemma 3 1B:

```
$ decode -lib <libLiteRt dir> -model gemma3-1b-it-int4.litertlm -text "The capital of France is"
output: " Paris.\n..."
```

The models are statically shaped (fixed-context KV cache; `decode` →
`logits f32[1,1,vocab]`); no resize or KV-cache growth is needed.

Two constraints are documented in the code: the MSVC `LiteRtRankedTensorType`
struct layout (dimensions at offset 12), and the rule for handing Go pointers to
the C side across the ffi boundary — every Go variable whose address reaches C
must be pinned with `runtime.Pinner` for the duration of the call, because the
goroutine stack can move mid-call and the stack mover does not rewrite addresses
laundered through `unsafe.Pointer`. Each wrapper in `litert/litert.go` owns a
Pinner and pins its argument slots; see the convention note in `litert/ffi.go`.

**Module:** `github.com/vladimirvivien/litert-go` · **Go:** 1.26.2

## Layout

```
litert/        no-CGO binding to the LiteRT C API (purego + jupiterrider/ffi)
  ffi.go         library loading, lazy symbol resolution, helpers
  bindings.go    one lazyFun per bound C entry point (~30 symbols)
  litert.go      typed wrappers + enums: Environment, Options, Model,
                 Signature, CompiledModel, TensorBuffer
litertlm/      .litertlm container reader (minimal FlatBuffer parser)
  litertlm.go    Sections / SectionTFLite / SectionBytes — extract sections
  metadata.go    ReadMetadata — model family + max tokens (protobuf scan)
cmd/spike/     Phase 0 harness: extract, signature dump, compile, smoke run
```

The binding targets the canonical LiteRT C API in `google-ai-edge/litert`
(cloned at `~/DEV/litert`, headers under `litert/c/`, API v0.1.0) — **not** the
copy vendored inside LiteRT-LM, whose `LiteRtCreateModelFrom*` functions lack the
leading `environment` argument.

## Prerequisites

- **`libLiteRt`** shared library, built from `google-ai-edge/litert`:
  - Windows: `bazelisk --output_base=C:/bzlt build //litert/c:libLiteRt --config=windows`
    → `bazel-bin/litert/c/libLiteRt.dll`
  - Linux/macOS: `//litert/c:litert_runtime_c_api_so` → `libLiteRt.so` / `.dylib`

  Point to its directory with `LITERT_LIB` or the `-lib` flag.
- **A model** — pass a `.litertlm` container directly (the harness extracts the
  embedded TFLite section) or a raw `.tflite`, via `-model`.

## Running the spike

Dump the signature contract (no library compile, no GPU required):

```
spike -lib /abs/path/to/litert/lib -model /abs/path/to/model.tflite
```

This prints each signature (`prefill`, `decode`, …) with its input/output tensor
names, element types, and shapes — the Phase 0a discovery (including whether the
model consumes token IDs or embeddings).

Compile and report accelerator coverage:

```
spike -lib ... -model ... -backend cpu
spike -lib ... -model ... -backend gpu      # Phase 0b: reports fully-accelerated
```

Prove the call path end to end (zeroed inputs — the logits are meaningless):

```
spike -lib ... -model ... -backend cpu -smoke -sig decode
```

## What the spike validates

- **0a** — purego can load, introspect, compile, allocate buffers, and run a
  signature; the signature contract is readable.
- **0b** — whether a GPU backend fully owns the graph (`fully accelerated: true`)
  or falls back to CPU.

## Run path (Phase 1)

Introspection and compile are proven. The decode/prefill signatures use dynamic
dims (encoded as `0` in these models), so buffers can't be allocated until the
inputs are made concrete. `ResizeInput` (NonStrict) is bound, but a correct run
needs the model's real shape protocol — the executor's `dyn_shape_resolver`
(replace dynamic dims, value from prefill length / step) plus per-step KV-cache
growth (`ResizeKVCacheTensorBuffer`). The `-smoke` path resizes dynamic dims to 1
and reaches buffer allocation; the actual greedy decode with KV-cache wiring and
an oracle comparison is Phase 1.

## Not yet wired (post-scaffold)

- the tokenizer (feed/compare token IDs via `litertlm-go`)
- correct dynamic-shape resolution + KV-cache management for a real decode step
- OpenCL GPU-backend selection (needs the GPU opaque options bound)
- selecting a specific section in multi-section containers (Gemma 3n/4, MTP,
  adapters); the harness extracts the first `TFLiteModel` section

These are the executor components described in
`~/DEV/project-planning/litert-go/litert-go-proposal.md`.

## Build

```
go build ./...
go vet ./...
```

## Benchmarks

The `litertlm` reader/loader benchmarks are gated on a real container:

```
LITERTLM_BENCH_FILE=/abs/path/model.litertlm go test ./litertlm -bench . -benchmem
```

`BenchmarkSections` / `BenchmarkSectionTFLite` measure the in-memory parser;
`BenchmarkReadTFLite` measures the full load (file read + parse) with throughput
over file size.
