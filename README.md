# litert-go

A no-CGO Go binding for the **LiteRT CompiledModel C API**, with a Go-driven LLM
decode runtime (prefill, fixed-context KV cache, greedy decode) built on top.

`cmd/decode` runs the full pipeline on CPU: text → SentencePiece tokenizer
(pure-Go `eliben/go-sentencepiece`, loaded from the `.litertlm`'s embedded
tokenizer section) → prefill → greedy decode → detokenize → text.

```
decode -lib <libLiteRt dir> -model gemma3-1b-it-int4.litertlm -text "The capital of France is"
output: " Paris."
```

Models are statically shaped: the KV cache is the full fixed context, `decode`
produces `logits f32[1,1,vocab]`, and no resize or KV-cache growth occurs.

**Constraints in the binding:**

- The MSVC build of `libLiteRt` lays out `LiteRtRankedTensorType` with dimensions
  at offset 12 (the `rank`/`has_strides` bitfields are not packed); the Windows
  binding reads shapes accordingly.
- Every Go variable whose address reaches the C side must be pinned with
  `runtime.Pinner` for the duration of the call: the goroutine stack can move
  mid-call and the stack mover does not rewrite addresses laundered through
  `unsafe.Pointer`. Scope one `Pinner` per C call. See `litert/ffi.go`.

**Module:** `github.com/vladimirvivien/litert-go` · **Go:** 1.26.2

## Layout

```
litert/        no-CGO binding to the LiteRT C API (purego + jupiterrider/ffi)
  ffi.go         library loading, lazy symbol resolution, runtime.Pinner convention
  bindings.go    one lazyFun per bound C entry point
  litert.go      typed wrappers + enums: Environment, Options, Model, Signature,
                 CompiledModel, TensorBuffer
  run.go         Runner — repeated Run over a fixed buffer set, arguments pinned once
litertlm/      .litertlm container reader (minimal FlatBuffer parser)
  litertlm.go    Sections (+ model_type hints) / SectionTFLite (selects the
                 prefill/decode graph) / SectionBytes
  metadata.go    ReadMetadata — model family + max tokens (protobuf scan)
cmd/decode/    text → prefill → greedy decode → text
cmd/spike/     signature dump, compile, and smoke-run a single signature
cmd/repro/     ffi argument-pinning regression guard
```

The binding targets the canonical LiteRT C API in `google-ai-edge/litert`
(headers under `litert/c/`, API v0.1.0) — not the copy vendored inside LiteRT-LM,
whose `LiteRtCreateModelFrom*` functions lack the leading `environment` argument.

## Prerequisites

- **`libLiteRt`** shared library, built from `google-ai-edge/litert`:
  - Windows: `bazelisk build //litert/c:libLiteRt --config=windows` → `bazel-bin/litert/c/libLiteRt.dll`
  - Linux/macOS: `//litert/c:litert_runtime_c_api_so` → `libLiteRt.so` / `.dylib`

  Point to its directory with `LITERT_LIB` or the `-lib` flag.
- **A model** — a `.litertlm` container (the embedded TFLite section is extracted)
  or a raw `.tflite`, via `-model`.

## Usage

Decode text:

```
decode -lib /path/to/libLiteRt -model model.litertlm -text "The capital of France is" -n 16
```

`-text` uses the model's embedded SentencePiece tokenizer; `-prompt` takes
comma-separated token IDs instead. `-n` caps the number of generated tokens.

`-chat` wraps `-text` in the model's chat template, read from the container's
`LlmMetadata` (`prompt_templates` affixes plus `start_token` / `stop_tokens`),
and stops decoding on the model's turn-end token:

```
decode -lib /path/to/libLiteRt -model model.litertlm -chat -text "What is the capital of France?"
```

Containers that carry only a Jinja template (e.g. Gemma 4) fall back to the
documented fixed affixes for that family, keyed on `llm_model_type`.

Dump a model's signatures (names, element types, shapes):

```
spike -lib /path/to/libLiteRt -model model.tflite
```

Compile and report accelerator coverage, or smoke-run one signature with zeroed
inputs (`fully accelerated: true` means a backend owns the whole graph with no
CPU fallback):

```
spike -lib ... -model ... -backend cpu
spike -lib ... -model ... -backend gpu
spike -lib ... -model ... -backend cpu -smoke -sig decode
```

## Limitations

- CPU only. `-backend gpu` selects LiteRT's default GPU backend; forcing the
  OpenCL backend needs opaque GPU options that are not bound.
- Greedy decode only — no temperature/top-k/top-p sampling or speculative decoding.
- Token-ID-input models only. Multi-section containers select the
  `tf_lite_prefill_decode` section automatically (other sections — embedder, MTP
  drafter, vision/audio adapters — are identified by their `model_type` hint),
  but models whose main graph consumes embeddings (Gemma 3n/4: a separate
  `tf_lite_embedder` stage and i8 KV cache) are not yet wired.

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

The `litert` ffi-call benchmarks are gated on `LITERT_LIB` plus a model in
`LITERT_BENCH_MODEL`:

```
go test ./litert -bench . -benchmem
```
