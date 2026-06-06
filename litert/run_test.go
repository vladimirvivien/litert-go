package litert_test

import (
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/vladimirvivien/litert-go/litert"
)

// TestRunner validates litert.Runner against a well-formed elementwise model.
// Set LITERT_LIB and LITERT_RUNNER_MODEL (a .tflite with an "add" signature:
// output = a + b over equal-shaped f32 inputs). A second Run with new inputs
// exercises argument reuse across calls.
func TestRunner(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_RUNNER_MODEL")
	if lib == "" || model == "" {
		t.Skip("set LITERT_LIB and LITERT_RUNNER_MODEL")
	}
	if err := litert.Load(lib); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(model)
	if err != nil {
		t.Fatal(err)
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()
	m, err := litert.OpenModelFromBuffer(env, raw)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	defer runtime.KeepAlive(raw)

	nsig, _ := m.NumSignatures()
	var g litert.Signature
	gidx, found := 0, false
	for i := 0; i < nsig; i++ {
		s, _ := m.Signature(i)
		if key, _ := s.Key(); key == "add" {
			g, gidx, found = s, i, true
			break
		}
	}
	if !found {
		t.Skip(`model has no "add" signature`)
	}

	opts, err := litert.NewOptions(litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	cm, err := litert.Compile(env, m, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer cm.Close()

	n := 0 // element count, from the first input's shape
	nin, _ := g.NumInputs()
	inBufs := make([]litert.TensorBuffer, nin)
	for i := 0; i < nin; i++ {
		name, _ := g.InputName(i)
		tt, _ := g.InputType(name)
		size, bt, err := cm.InputBufferInfo(gidx, i)
		if err != nil {
			t.Fatal(err)
		}
		buf, err := litert.NewManagedBuffer(env, bt, tt, size)
		if err != nil {
			t.Fatal(err)
		}
		defer buf.Close()
		inBufs[i] = buf
		n = 1
		for _, d := range tt.Shape {
			n *= int(d)
		}
	}
	nout, _ := g.NumOutputs()
	outBufs := make([]litert.TensorBuffer, nout)
	for i := 0; i < nout; i++ {
		name, _ := g.OutputName(i)
		tt, _ := g.OutputType(name)
		size, bt, err := cm.OutputBufferInfo(gidx, i)
		if err != nil {
			t.Fatal(err)
		}
		buf, err := litert.NewManagedBuffer(env, bt, tt, size)
		if err != nil {
			t.Fatal(err)
		}
		defer buf.Close()
		outBufs[i] = buf
	}

	fill := func(b litert.TensorBuffer, v float32) {
		addr, err := b.Lock(litert.LockWrite)
		if err != nil {
			t.Fatal(err)
		}
		s := unsafe.Slice((*float32)(addr), n)
		for i := range s {
			s[i] = v
		}
		if err := b.Unlock(); err != nil {
			t.Fatal(err)
		}
	}
	first := func(b litert.TensorBuffer) float32 {
		addr, err := b.Lock(litert.LockRead)
		if err != nil {
			t.Fatal(err)
		}
		defer b.Unlock()
		return unsafe.Slice((*float32)(addr), n)[0]
	}

	r := litert.NewRunner(cm, gidx, inBufs, outBufs)
	defer r.Close()

	// Filling every input with v yields output = v+v for an add signature.
	for _, b := range inBufs {
		fill(b, 2)
	}
	if err := r.Run(); err != nil {
		t.Fatal(err)
	}
	if got := first(outBufs[0]); got != 4 {
		t.Fatalf("run 1: got %v, want 4", got)
	}

	for _, b := range inBufs {
		fill(b, 10)
	}
	if err := r.Run(); err != nil {
		t.Fatal(err)
	}
	if got := first(outBufs[0]); got != 20 {
		t.Fatalf("run 2 (reuse): got %v, want 20", got)
	}

	// Sustained reuse: the Runner holds its pinned arguments across many Run
	// calls. Loop hard (run under GOGC=1) to confirm the held pins stay valid.
	for i := 0; i < 500; i++ {
		v := float32(i%9) + 1
		for _, b := range inBufs {
			fill(b, v)
		}
		if err := r.Run(); err != nil {
			t.Fatalf("stress run %d: %v", i, err)
		}
		if got := first(outBufs[0]); got != 2*v {
			t.Fatalf("stress run %d: got %v, want %v", i, got, 2*v)
		}
	}
}
