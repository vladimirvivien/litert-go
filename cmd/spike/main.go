// Command spike is the litert-go Phase 0 harness.
//
// It validates that the LiteRT CompiledModel C API can be driven from pure Go
// (purego, no CGO). It always dumps a model's signature contract — the key
// Phase 0a discovery (tensor names, shapes, dtypes; token-ids vs embeddings).
// With -backend it compiles the model and reports whether the accelerator fully
// owns the graph (the Phase 0b signal). With -smoke it runs one inference with
// zeroed inputs to prove the call path end to end.
//
// The -model argument must be a raw .tflite extracted from a .litertlm.
// Correctness wiring (real tokens, KV cache, greedy loop, oracle comparison) is
// the post-scaffold step described in litert-go-proposal.md.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"unsafe"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

func main() {
	libDir := flag.String("lib", "", "directory or path of libLiteRt (or set LITERT_LIB)")
	modelPath := flag.String("model", "", "path to a .litertlm container or a raw .tflite model")
	backend := flag.String("backend", "none", "compile backend: none | cpu | gpu")
	smoke := flag.Bool("smoke", false, "run one inference with zeroed inputs (proves the call path; output is meaningless)")
	sigName := flag.String("sig", "decode", "signature key to smoke-run")
	section := flag.String("section", "", "TFLiteModel model_type to introspect (e.g. tf_lite_embedder); default is the prefill/decode graph")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "spike: -model is required")
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*libDir, *modelPath, *backend, *sigName, *section, *smoke); err != nil {
		fmt.Fprintln(os.Stderr, "spike:", err)
		os.Exit(1)
	}
}

func run(libDir, modelPath, backend, sigName, section string, smoke bool) error {
	if err := litert.Load(libDir); err != nil {
		return err
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		return err
	}
	defer env.Close()

	raw, err := os.ReadFile(modelPath)
	if err != nil {
		return err
	}
	if litertlm.IsContainer(raw) {
		secs, err := litertlm.Sections(raw)
		if err != nil {
			return err
		}
		fmt.Printf(".litertlm container, %d sections:\n", len(secs))
		nTFLite := 0
		marked := false
		for _, s := range secs {
			hint := ""
			if mt := s.Items[litertlm.HintModelType]; mt != "" {
				hint = "  model_type=" + mt
			}
			marker := ""
			if s.Type == litertlm.SectionTFLiteModel {
				nTFLite++
				mt := s.Items[litertlm.HintModelType]
				if !marked && (mt == "" || strings.EqualFold(mt, litertlm.TFLitePrefillDecode)) {
					marker = "  <- main"
					marked = true
				}
			}
			fmt.Printf("  %-18s [%d, %d) %d bytes%s%s\n", s.TypeName(), s.Begin, s.End, s.End-s.Begin, hint, marker)
		}
		if nTFLite > 1 {
			fmt.Printf("  %d TFLiteModel sections; SectionTFLite selects model_type=%s.\n", nTFLite, litertlm.TFLitePrefillDecode)
		}
		if md, err := litertlm.ReadMetadata(raw); err == nil {
			fmt.Printf("  model type: %s", md.ModelType)
			if md.MaxNumTokens > 0 {
				fmt.Printf(" (max %d tokens)", md.MaxNumTokens)
			}
			fmt.Println()
		}
		var tflite []byte
		if section != "" {
			tflite, err = litertlm.SectionTFLiteModelType(raw, section)
		} else {
			tflite, err = litertlm.SectionTFLite(raw)
		}
		if err != nil {
			return err
		}
		raw = tflite
	}
	// The C API references the buffer for the model's lifetime; keep raw alive
	// until after model.Close (this defer runs last).
	defer runtime.KeepAlive(raw)

	model, err := litert.OpenModelFromBuffer(env, raw)
	if err != nil {
		return err
	}
	defer model.Close()

	sigIndex, err := dumpSignatures(model)
	if err != nil {
		return err
	}

	if backend == "none" {
		return nil
	}

	accel, err := accelerator(backend)
	if err != nil {
		return err
	}
	opts, err := litert.NewOptions(accel)
	if err != nil {
		return err
	}
	defer opts.Close()

	fmt.Printf("\nCompiling for backend %q...\n", backend)
	compiled, err := litert.Compile(env, model, opts)
	if err != nil {
		return err
	}
	defer compiled.Close()

	full, err := compiled.FullyAccelerated()
	if err != nil {
		return err
	}
	fmt.Printf("fully accelerated: %v\n", full)
	if backend == "gpu" && !full {
		fmt.Println("  NOTE: not fully accelerated — ops fell back to CPU (Spike 0b: partial/unsupported).")
	}

	if !smoke {
		return nil
	}
	idx, ok := sigIndex[sigName]
	if !ok {
		return fmt.Errorf("signature %q not found for -smoke", sigName)
	}
	return smokeRun(env, compiled, model, idx, sigName)
}

func accelerator(backend string) (litert.HwAccelerator, error) {
	switch backend {
	case "cpu":
		return litert.AccelCPU, nil
	case "gpu":
		// Default GPU backend. Forcing OpenCL (Spike 0b) needs the GPU opaque
		// options, not yet bound — see litert.NewOptions.
		return litert.AccelGPU, nil
	default:
		return 0, fmt.Errorf("unknown backend %q (want none|cpu|gpu)", backend)
	}
}

// dumpSignatures prints every signature's input/output contract and returns a
// map from signature key to index.
func dumpSignatures(m litert.Model) (map[string]int, error) {
	n, err := m.NumSignatures()
	if err != nil {
		return nil, err
	}
	fmt.Printf("signatures: %d\n", n)
	index := make(map[string]int, n)
	for i := 0; i < n; i++ {
		sig, err := m.Signature(i)
		if err != nil {
			return nil, err
		}
		key, err := sig.Key()
		if err != nil {
			return nil, err
		}
		index[key] = i
		fmt.Printf("\n[%d] signature %q\n", i, key)

		nin, err := sig.NumInputs()
		if err != nil {
			return nil, err
		}
		for j := 0; j < nin; j++ {
			name, err := sig.InputName(j)
			if err != nil {
				return nil, err
			}
			tt, err := sig.InputType(name)
			if err != nil {
				return nil, err
			}
			fmt.Printf("  in  %-24s %s%v\n", name, tt.ElementType, tt.Shape)
		}

		nout, err := sig.NumOutputs()
		if err != nil {
			return nil, err
		}
		for j := 0; j < nout; j++ {
			name, err := sig.OutputName(j)
			if err != nil {
				return nil, err
			}
			tt, err := sig.OutputType(name)
			if err != nil {
				return nil, err
			}
			fmt.Printf("  out %-24s %s%v\n", name, tt.ElementType, tt.Shape)
		}
	}
	return index, nil
}

// smokeRun allocates input/output buffers per the compiled model's
// requirements, zero-fills the inputs, runs the signature once, and reports the
// first output buffer. Zeroed inputs make the logits meaningless — this proves
// the call path (compile, buffer alloc, run, read), not correctness.
func smokeRun(env litert.Environment, c litert.CompiledModel, m litert.Model, sigIdx int, sigName string) error {
	fmt.Printf("\nSmoke-running %q (zeroed inputs — output is meaningless)...\n", sigName)
	sig, err := m.Signature(sigIdx)
	if err != nil {
		return err
	}

	nin, err := sig.NumInputs()
	if err != nil {
		return err
	}
	var inputs []litert.TensorBuffer
	defer func() {
		for _, b := range inputs {
			b.Close()
		}
	}()
	// Make dynamic input dims concrete (resolver: dynamic -> 1), then allocate.
	for i := 0; i < nin; i++ {
		name, err := sig.InputName(i)
		if err != nil {
			return err
		}
		tt, err := sig.InputType(name)
		if err != nil {
			return err
		}
		if hasDynamic(tt.Shape) {
			if err := c.ResizeInput(sigIdx, i, resolveDyn(tt.Shape)); err != nil {
				return err
			}
		}
	}
	for i := 0; i < nin; i++ {
		name, err := sig.InputName(i)
		if err != nil {
			return err
		}
		tt, err := sig.InputType(name)
		if err != nil {
			return err
		}
		rt := litert.TensorType{ElementType: tt.ElementType, Shape: resolveDyn(tt.Shape)}
		size, bt, err := c.InputBufferInfo(sigIdx, i)
		if err != nil {
			return err
		}
		buf, err := litert.NewManagedBuffer(env, bt, rt, size)
		if err != nil {
			return err
		}
		if err := zeroBuffer(buf, size); err != nil {
			return err
		}
		inputs = append(inputs, buf)
	}

	nout, err := sig.NumOutputs()
	if err != nil {
		return err
	}
	var outputs []litert.TensorBuffer
	outSizes := make([]uint64, nout)
	outTypes := make([]litert.TensorType, nout)
	defer func() {
		for _, b := range outputs {
			b.Close()
		}
	}()
	for i := 0; i < nout; i++ {
		name, err := sig.OutputName(i)
		if err != nil {
			return err
		}
		tt, err := sig.OutputType(name)
		if err != nil {
			return err
		}
		rt := litert.TensorType{ElementType: tt.ElementType, Shape: resolveDyn(tt.Shape)}
		size, bt, err := c.OutputBufferInfo(sigIdx, i)
		if err != nil {
			return err
		}
		buf, err := litert.NewManagedBuffer(env, bt, rt, size)
		if err != nil {
			return err
		}
		outputs = append(outputs, buf)
		outSizes[i] = size
		outTypes[i] = rt
	}

	if err := c.Run(sigIdx, inputs, outputs); err != nil {
		return err
	}
	fmt.Println("run: ok")

	if nout == 0 {
		return nil
	}
	return reportOutput(outputs[0], outSizes[0], outTypes[0])
}

// resolveDyn replaces dynamic dims (<= 0) with 1, mirroring the executor's
// dynamic-shape resolver. Real decode uses model-specific values; 1 is enough
// to make the smoke run's buffers allocatable and exercise the call path.
func hasDynamic(shape []int32) bool {
	for _, d := range shape {
		if d <= 0 {
			return true
		}
	}
	return false
}

func resolveDyn(shape []int32) []int32 {
	out := make([]int32, len(shape))
	for i, d := range shape {
		if d <= 0 {
			out[i] = 1
		} else {
			out[i] = d
		}
	}
	return out
}

func zeroBuffer(b litert.TensorBuffer, size uint64) error {
	addr, err := b.Lock(litert.LockWrite)
	if err != nil {
		return err
	}
	clear(unsafe.Slice((*byte)(addr), size))
	return b.Unlock()
}

// reportOutput reads the first output buffer and, for float32 logits, prints
// the argmax — the hook the real greedy loop will replace.
func reportOutput(b litert.TensorBuffer, size uint64, tt litert.TensorType) error {
	addr, err := b.Lock(litert.LockRead)
	if err != nil {
		return err
	}
	defer b.Unlock()

	fmt.Printf("output[0]: %s%v, %d bytes\n", tt.ElementType, tt.Shape, size)
	if tt.ElementType != litert.ElementFloat32 {
		return nil
	}
	vals := unsafe.Slice((*float32)(addr), int(size/4))
	argmax, best := 0, vals[0]
	for i, v := range vals {
		if v > best {
			best, argmax = v, i
		}
	}
	fmt.Printf("argmax token id: %d (logit %.4f)\n", argmax, best)
	return nil
}
