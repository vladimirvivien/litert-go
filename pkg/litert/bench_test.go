package litert_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/vladimirvivien/litert-go/pkg/litert"
)

func TestEngineIntegration(t *testing.T) {
	requireLiveRuntime(t)

	ctx := context.Background()
	eng, err := litert.Open(ctx, testdataPath(t, "simple_l2_norm.tflite"),
		litert.WithAccelerator(litert.AccelCPU),
		litert.WithCPUOptions(litert.CPUOptions{NumThreads: 2}),
		litert.WithLogSeverity(litert.LogWarning),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	inSpec := eng.DefaultSignature().Inputs()[0]
	outSpec := eng.DefaultSignature().Outputs()[0]
	nElems, ok := inSpec.Type.NumElements()
	if !ok || nElems != 768 {
		t.Fatalf("unexpected input spec: %+v", inSpec)
	}
	inData := make([]float32, nElems)
	for i := range inData {
		inData[i] = 2.0
	}

	outputs, err := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inData})
	if err != nil {
		_ = eng.Close()
		t.Fatalf("Engine.Run: %v", err)
	}
	got, ok := outputs[outSpec.Name].([]float32)
	if !ok || len(got) != nElems {
		_ = eng.Close()
		t.Fatalf("unexpected output %T len=%d", outputs[outSpec.Name], len(got))
	}
	expected := float32(1.0 / math.Sqrt(float64(nElems)))
	if math.Abs(float64(got[0]-expected)) > 1e-4 {
		_ = eng.Close()
		t.Fatalf("L2 norm[0] = %v, want ~%v", got[0], expected)
	}

	// Reject unknown and missing inputs cleanly.
	if _, badErr := eng.Run(ctx, litert.NamedInputs{"bogus": inData}); !errors.Is(badErr, litert.ErrInvalidArgument) {
		_ = eng.Close()
		t.Fatalf("expected ErrInvalidArgument for unknown input, got %v", badErr)
	}
	if _, missErr := eng.Run(ctx, litert.NamedInputs{}); !errors.Is(missErr, litert.ErrInvalidArgument) {
		_ = eng.Close()
		t.Fatalf("expected ErrInvalidArgument for missing input, got %v", missErr)
	}

	// Idempotent close and post-close ErrClosed checks.
	if cErr := eng.Close(); cErr != nil {
		t.Fatalf("Engine.Close: %v", cErr)
	}
	if cErr := eng.Close(); cErr != nil {
		t.Fatalf("Engine.Close (second): %v", cErr)
	}
	if _, runErr := eng.Run(ctx, litert.NamedInputs{inSpec.Name: inData}); !errors.Is(runErr, litert.ErrClosed) {
		t.Fatalf("expected ErrClosed after Engine.Close, got %v", runErr)
	}

	// Multi-input / multi-output named signature dispatch via OpenBytes.
	revBytes, err := os.ReadFile(testdataPath(t, "reverse_signature_model.tflite"))
	if err != nil {
		t.Fatal(err)
	}
	revEng, err := litert.OpenBytes(ctx, revBytes)
	if err != nil {
		t.Fatalf("OpenBytes(reverse_signature_model): %v", err)
	}
	defer func() { _ = revEng.Close() }()

	xVals := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	yVals := []float32{2, 3, 4, 5, 6, 7, 8, 9}
	revOut, err := revEng.Run(ctx, litert.NamedInputs{"x": xVals, "y": yVals}, litert.WithSignature("serving_default"))
	if err != nil {
		t.Fatalf("revEng.Run: %v", err)
	}
	sums := revOut["sum"].([]float32)
	prods := revOut["prod"].([]float32)
	for i := range xVals {
		if sums[i] != xVals[i]+yVals[i] || prods[i] != xVals[i]*yVals[i] {
			t.Fatalf("idx %d: sum=%v prod=%v", i, sums[i], prods[i])
		}
	}

	// Quantized int16 typed I/O.
	qEng, err := litert.Open(ctx, testdataPath(t, "simple_quantized_ops.tflite"))
	if err != nil {
		t.Fatalf("Open(simple_quantized_ops): %v", err)
	}
	defer func() { _ = qEng.Close() }()
	qCount, _ := qEng.DefaultSignature().Inputs()[0].Type.NumElements()
	qx := make([]int16, qCount)
	qy := make([]int16, qCount)
	qx[0], qy[0] = 100, 50
	qOut, err := qEng.Run(ctx, litert.NamedInputs{"x": qx, "y": qy})
	if err != nil {
		t.Fatalf("qEng.Run: %v", err)
	}
	if qRes, isI16 := qOut["output_0"].([]int16); !isI16 || len(qRes) != qCount {
		t.Fatalf("unexpected quantized output %T len=%d", qOut["output_0"], len(qRes))
	}

	// Dynamic shape Engine.NewRunner with InputByName/OutputByName and WriteSlice/ReadSliceInto.
	dynEng, err := litert.Open(ctx, testdataPath(t, "simple_add_dynamic_shape.tflite"))
	if err != nil {
		t.Fatalf("Open(simple_add_dynamic_shape): %v", err)
	}
	defer func() { _ = dynEng.Close() }()

	dynRunner, err := dynEng.NewRunner(litert.WithInputShapes(map[string][]int32{
		"arg0": {2, 3},
		"arg1": {2, 3},
	}))
	if err != nil {
		t.Fatalf("dynEng.NewRunner: %v", err)
	}
	defer func() { _ = dynRunner.Close() }()

	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{10, 20, 30, 40, 50, 60}
	if wErr := litert.WriteSlice(dynRunner.InputByName("arg0"), a); wErr != nil {
		t.Fatalf("WriteSlice(arg0): %v", wErr)
	}
	if wErr := litert.WriteSlice(dynRunner.InputByName("arg1"), b); wErr != nil {
		t.Fatalf("WriteSlice(arg1): %v", wErr)
	}
	if rErr := dynRunner.Run(ctx); rErr != nil {
		t.Fatalf("dynRunner.Run: %v", rErr)
	}
	outSlice := make([]float32, 6)
	if rdErr := litert.ReadSliceInto(dynRunner.OutputByName("tfl.add"), outSlice); rdErr != nil {
		t.Fatalf("ReadSliceInto: %v", rdErr)
	}
	for i := range outSlice {
		if outSlice[i] != a[i]+b[i] {
			t.Fatalf("dyn add[%d] = %v, want %v", i, outSlice[i], a[i]+b[i])
		}
	}
}

func TestRunnerRunZeroAllocs(t *testing.T) {
	requireLiveRuntime(t)

	ctx := context.Background()
	eng, err := litert.Open(ctx, testdataPath(t, "simple_l2_norm.tflite"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = eng.Close() }()

	runner, err := eng.NewRunner()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = runner.Close() }()

	inData := make([]float32, 768)
	for i := range inData {
		inData[i] = 1.0
	}
	if wErr := litert.WriteSlice(runner.Input(0), inData); wErr != nil {
		t.Fatal(wErr)
	}
	if runErr := runner.Run(ctx); runErr != nil {
		t.Fatal(runErr)
	}

	var runErr error
	allocs := testing.AllocsPerRun(200, func() {
		runErr = runner.Run(ctx)
	})
	if runErr != nil {
		t.Fatalf("runner.Run during alloc test: %v", runErr)
	}
	if allocs != 0 {
		t.Fatalf("Runner.Run(context.Background()) allocated %.2f allocs/op, want 0", allocs)
	}
}

func TestLifecycleChurn(t *testing.T) {
	requireLiveRuntime(t)

	raw, err := os.ReadFile(testdataPath(t, "simple_l2_norm.tflite"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	inData := make([]float32, 768)
	for i := range inData {
		inData[i] = 1.0
	}
	outData := make([]float32, 768)

	const cycles = 500
	for i := range cycles {
		eng, openErr := litert.OpenBytes(ctx, raw)
		if openErr != nil {
			t.Fatalf("cycle %d OpenBytes: %v", i, openErr)
		}
		runner, rErr := eng.NewRunner()
		if rErr != nil {
			_ = eng.Close()
			t.Fatalf("cycle %d NewRunner: %v", i, rErr)
		}
		if wErr := litert.WriteSlice(runner.Input(0), inData); wErr != nil {
			_ = runner.Close()
			_ = eng.Close()
			t.Fatalf("cycle %d WriteSlice: %v", i, wErr)
		}
		if runErr := runner.Run(ctx); runErr != nil {
			_ = runner.Close()
			_ = eng.Close()
			t.Fatalf("cycle %d Run: %v", i, runErr)
		}
		if rdErr := litert.ReadSliceInto(runner.Output(0), outData); rdErr != nil {
			_ = runner.Close()
			_ = eng.Close()
			t.Fatalf("cycle %d ReadSliceInto: %v", i, rdErr)
		}
		_ = runner.Close()
		_ = eng.Close()
		if i%50 == 0 {
			runtime.GC()
		}
	}
}

func BenchmarkRunnerRun(b *testing.B) {
	if os.Getenv("LITERT_LIB") == "" {
		b.Skip("set LITERT_LIB to run live benchmarks")
	}
	if err := litert.Init(""); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	eng, err := litert.Open(ctx, filepath.Join("..", "..", "testdata", "simple_l2_norm.tflite"))
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = eng.Close() }()

	runner, err := eng.NewRunner()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = runner.Close() }()

	inData := make([]float32, 768)
	for i := range inData {
		inData[i] = 1.0
	}
	if err := litert.WriteSlice(runner.Input(0), inData); err != nil {
		b.Fatal(err)
	}
	if err := runner.Run(ctx); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := runner.Run(ctx); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHostBufferVsManaged(b *testing.B) {
	if os.Getenv("LITERT_LIB") == "" {
		b.Skip("set LITERT_LIB to run live benchmarks")
	}
	if err := litert.Init(""); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	eng, err := litert.Open(ctx, filepath.Join("..", "..", "testdata", "simple_l2_norm.tflite"))
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = eng.Close() }()

	inData := make([]float32, 768)
	outData := make([]float32, 768)
	for i := range inData {
		inData[i] = 1.0
	}

	b.Run("Managed", func(b *testing.B) {
		runner, rErr := eng.NewRunner()
		if rErr != nil {
			b.Fatal(rErr)
		}
		defer func() { _ = runner.Close() }()

		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			if err := litert.WriteSlice(runner.Input(0), inData); err != nil {
				b.Fatal(err)
			}
			if err := runner.Run(ctx); err != nil {
				b.Fatal(err)
			}
			if err := litert.ReadSliceInto(runner.Output(0), outData); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("HostZeroCopy", func(b *testing.B) {
		sig := eng.DefaultSignature()
		inBytes := litert.AllocAlignedBytes(768 * 4)
		outBytes := litert.AllocAlignedBytes(768 * 4)

		inBuf, hbErr := litert.NewHostBuffer(sig.Inputs()[0].Type, inBytes)
		if hbErr != nil {
			b.Fatal(hbErr)
		}
		defer func() { _ = inBuf.Close() }()

		outBuf, hbErr := litert.NewHostBuffer(sig.Outputs()[0].Type, outBytes)
		if hbErr != nil {
			b.Fatal(hbErr)
		}
		defer func() { _ = outBuf.Close() }()

		runner, rErr := litert.NewRunner(eng.CompiledModel(), sig.Index(), []*litert.TensorBuffer{inBuf}, []*litert.TensorBuffer{outBuf})
		if rErr != nil {
			b.Fatal(rErr)
		}
		defer func() { _ = runner.Close() }()

		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			if err := runner.Run(ctx); err != nil {
				b.Fatal(err)
			}
		}
	})
}
