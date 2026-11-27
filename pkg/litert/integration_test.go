package litert_test

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"unsafe"

	"github.com/vladimirvivien/litert-go/pkg/litert"
	"github.com/vladimirvivien/litert-go/pkg/loader"
)

func requireLiveRuntime(t *testing.T) {
	t.Helper()
	if os.Getenv(loader.EnvVar) == "" {
		t.Skipf("set %s to run live LiteRT integration tests", loader.EnvVar)
	}
	if err := litert.Init(""); err != nil {
		t.Fatalf("litert.Init: %v", err)
	}
}

func testdataPath(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join("..", "..", "testdata", name)
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("missing testdata fixture %q: %v", p, err)
	}
	return p
}

func TestLiveRuntimeABIVersionAndEnvironment(t *testing.T) {
	requireLiveRuntime(t)

	abi, err := litert.RuntimeABIVersion()
	if err != nil {
		t.Fatalf("RuntimeABIVersion: %v", err)
	}
	if abi == "" {
		t.Fatal("expected non-empty RuntimeABIVersion")
	}
	t.Logf("LiteRT ABI version: %s", abi)

	env, err := litert.NewEnvironment(
		litert.EnvIntOption(litert.EnvMinLoggerSeverity, int64(litert.LogWarning)),
		litert.EnvStringOption(litert.EnvCompilerCacheDir, t.TempDir()),
	)
	if err != nil {
		t.Fatalf("NewEnvironment: %v", err)
	}
	defer func() {
		if closeErr := env.Close(); closeErr != nil {
			t.Errorf("env.Close: %v", closeErr)
		}
	}()

	if _, fp16Err := env.SupportsFP16(); fp16Err != nil {
		t.Fatalf("SupportsFP16: %v", fp16Err)
	}
	if _, gpuErr := env.HasGPUEnvironment(); gpuErr != nil {
		t.Fatalf("HasGPUEnvironment: %v", gpuErr)
	}
}

func TestLiveIntrospectTestdataFixtures(t *testing.T) {
	requireLiveRuntime(t)

	env, err := litert.NewEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = env.Close() }()

	fixtures := []string{
		"simple_l2_norm.tflite",
		"simple_add_dynamic_shape.tflite",
		"simple_quantized_ops.tflite",
		"reverse_signature_model.tflite",
		"mul_quant.tflite",
	}

	for _, name := range fixtures {
		t.Run(name, func(t *testing.T) {
			m, openErr := litert.OpenModelFromFile(env, testdataPath(t, name))
			if openErr != nil {
				t.Fatalf("OpenModelFromFile(%s): %v", name, openErr)
			}
			defer func() { _ = m.Close() }()

			sig, sigErr := m.DefaultSignature()
			if sigErr != nil {
				t.Fatalf("DefaultSignature: %v", sigErr)
			}
			if len(sig.Inputs()) == 0 || len(sig.Outputs()) == 0 {
				t.Fatalf("expected non-empty I/O specs, got in=%d out=%d", len(sig.Inputs()), len(sig.Outputs()))
			}
			t.Logf("%s: sig=%q inputs=%+v outputs=%+v", name, sig.Key(), sig.Inputs(), sig.Outputs())
		})
	}
}

func TestLiveBufferLoadAndGCChurn(t *testing.T) {
	requireLiveRuntime(t)

	raw, err := os.ReadFile(testdataPath(t, "simple_l2_norm.tflite"))
	if err != nil {
		t.Fatal(err)
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = env.Close() }()

	m, err := litert.OpenModelFromBuffer(env, raw)
	if err != nil {
		t.Fatalf("OpenModelFromBuffer: %v", err)
	}
	defer func() { _ = m.Close() }()

	opts, err := litert.NewOptions(litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = opts.Close() }()

	if addErr := opts.AddCPUOptions(litert.CPUOptions{NumThreads: 2}); addErr != nil {
		t.Fatalf("AddCPUOptions: %v", addErr)
	}

	cm, err := litert.Compile(env, m, opts)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer func() { _ = cm.Close() }()

	sig, err := m.DefaultSignature()
	if err != nil {
		t.Fatal(err)
	}
	inSpec := sig.Inputs()[0]
	outSpec := sig.Outputs()[0]

	inBuf, err := cm.CreateInputBuffer(sig.Index(), inSpec, 0)
	if err != nil {
		t.Fatalf("CreateInputBuffer: %v", err)
	}
	defer func() { _ = inBuf.Close() }()

	outBuf, err := cm.CreateOutputBuffer(sig.Index(), outSpec, 0)
	if err != nil {
		t.Fatalf("CreateOutputBuffer: %v", err)
	}
	defer func() { _ = outBuf.Close() }()

	// Stress GC pinning during repeated signature/buffer inspection and execution.
	nElems, ok := inSpec.Type.NumElements()
	if !ok || nElems == 0 {
		t.Fatalf("unexpected input elements: %+v", inSpec.Type)
	}
	payload := make([]byte, nElems*4)
	for i := range nElems {
		binary.LittleEndian.PutUint32(payload[i*4:], math.Float32bits(3.0))
	}

	runner, err := litert.NewRunner(cm, sig.Index(), []*litert.TensorBuffer{inBuf}, []*litert.TensorBuffer{outBuf})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}
	defer func() { _ = runner.Close() }()

	for i := range 25 {
		if wErr := inBuf.WriteBytes(payload); wErr != nil {
			t.Fatalf("WriteBytes[%d]: %v", i, wErr)
		}
		if runErr := runner.Run(context.Background()); runErr != nil {
			t.Fatalf("runner.Run[%d]: %v", i, runErr)
		}
		runtime.GC()
		if _, sigErr := m.DefaultSignature(); sigErr != nil {
			t.Fatalf("DefaultSignature after GC[%d]: %v", i, sigErr)
		}
	}

	outRaw, err := outBuf.ReadAllBytes()
	if err != nil {
		t.Fatalf("ReadAllBytes: %v", err)
	}
	if len(outRaw) != nElems*4 {
		t.Fatalf("got %d output bytes, want %d", len(outRaw), nElems*4)
	}

	// Verify L2 norm result: all elements equal -> each output is 1/sqrt(nElems).
	expected := float32(1.0 / math.Sqrt(float64(nElems)))
	got0 := math.Float32frombits(binary.LittleEndian.Uint32(outRaw[0:4]))
	if math.Abs(float64(got0-expected)) > 1e-4 {
		t.Fatalf("L2 norm[0] = %v, want ~%v", got0, expected)
	}

	if _, durErr := cm.LastInferenceDuration(); durErr != nil {
		t.Fatalf("LastInferenceDuration: %v", durErr)
	}
	dm, err := cm.DelegationMetrics()
	if err != nil {
		t.Fatalf("DelegationMetrics: %v", err)
	}
	if dm.TotalNodeCount <= 0 {
		t.Fatalf("expected positive TotalNodeCount, got %d", dm.TotalNodeCount)
	}
}

func TestLiveDynamicResizeHostBufferAndAsync(t *testing.T) {
	requireLiveRuntime(t)

	env, err := litert.NewEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = env.Close() }()

	m, err := litert.OpenModelFromFile(env, testdataPath(t, "simple_add_dynamic_shape.tflite"))
	if err != nil {
		t.Fatalf("OpenModelFromFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	opts, err := litert.NewOptions(litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = opts.Close() }()

	cm, err := litert.Compile(env, m, opts)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer func() { _ = cm.Close() }()

	sig, err := m.DefaultSignature()
	if err != nil {
		t.Fatal(err)
	}

	// Resize dynamic input tensors to [2, 3].
	dims := []int32{2, 3}
	for i := range sig.Inputs() {
		if rErr := cm.ResizeInputNonStrict(sig.Index(), i, dims); rErr != nil {
			t.Fatalf("ResizeInputNonStrict(%d): %v", i, rErr)
		}
	}
	outLayouts, err := cm.OutputLayouts(sig.Index(), len(sig.Outputs()), true)
	if err != nil {
		t.Fatalf("OutputLayouts: %v", err)
	}
	if len(outLayouts) != 1 || len(outLayouts[0].Dimensions) != 2 || outLayouts[0].Dimensions[0] != 2 || outLayouts[0].Dimensions[1] != 3 {
		t.Fatalf("unexpected resized output layouts: %+v", outLayouts)
	}

	inLayout, err := cm.InputLayout(sig.Index(), 0)
	if err != nil {
		t.Fatalf("InputLayout: %v", err)
	}
	if len(inLayout.Dimensions) != 2 || inLayout.Dimensions[0] != 2 || inLayout.Dimensions[1] != 3 {
		t.Fatalf("unexpected input layout: %+v", inLayout)
	}

	// Allocate zero-copy 64-byte aligned host buffers for inputs and output.
	n := 6
	elemType := sig.Inputs()[0].Type.ElementType
	tt := litert.TensorType{ElementType: elemType, Shape: dims}
	byteLen := n * 4

	inBufs := make([]*litert.TensorBuffer, len(sig.Inputs()))
	for i := range inBufs {
		hostData := litert.AllocAlignedBytes(byteLen)
		for k := range n {
			binary.LittleEndian.PutUint32(hostData[k*4:], math.Float32bits(float32(i+1)*1.5))
		}
		hb, hbErr := litert.NewHostBuffer(tt, hostData)
		if hbErr != nil {
			t.Fatalf("NewHostBuffer(in %d): %v", i, hbErr)
		}
		defer func() { _ = hb.Close() }()
		inBufs[i] = hb
	}

	outHost := litert.AllocAlignedBytes(byteLen)
	outBuf, err := litert.NewHostBuffer(tt, outHost)
	if err != nil {
		t.Fatalf("NewHostBuffer(out): %v", err)
	}
	defer func() { _ = outBuf.Close() }()

	if _, runErr := cm.RunAsync(context.Background(), sig.Index(), inBufs, []*litert.TensorBuffer{outBuf}); runErr != nil {
		t.Fatalf("RunAsync: %v", runErr)
	}
	if waitErr := outBuf.Wait(-1); waitErr != nil {
		t.Fatalf("outBuf.Wait: %v", waitErr)
	}

	got := math.Float32frombits(binary.LittleEndian.Uint32(outHost[0:4]))
	if got != 4.5 {
		t.Fatalf("dynamic add output[0] = %v, want 4.5", got)
	}

	// Duplicate + verify cancellation error path when ctx is already cancelled.
	dup, err := outBuf.Duplicate()
	if err != nil {
		t.Fatalf("Duplicate: %v", err)
	}
	_ = dup.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if runErr := cm.Run(ctx, sig.Index(), inBufs, []*litert.TensorBuffer{outBuf}); runErr == nil {
		t.Fatal("expected cancellation error on cancelled context")
	}

	// Verify Lock/Unlock slice access works on duplicated/original host buffer.
	addr, err := outBuf.Lock(litert.LockRead)
	if err != nil {
		t.Fatalf("Lock: %v", err)
	}
	s := unsafe.Slice((*float32)(addr), n)
	if s[5] != 4.5 {
		_ = outBuf.Unlock()
		t.Fatalf("locked output[5] = %v, want 4.5", s[5])
	}
	if uErr := outBuf.Unlock(); uErr != nil {
		t.Fatalf("Unlock: %v", uErr)
	}
}
