package litert_test

import (
	"os"
	"runtime"
	"testing"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// benchSig loads the model named by LITERT_BENCH_MODEL (a .litertlm or .tflite)
// against the library in LITERT_LIB and returns its first signature. The
// benchmarks isolate the per-call ffi + runtime.Pinner overhead: every wrapper
// crosses into C, so these numbers are "C transition + argument pinning", not
// pinning alone.
func benchSig(b *testing.B) litert.Signature {
	b.Helper()
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_BENCH_MODEL")
	if lib == "" || model == "" {
		b.Skip("set LITERT_LIB and LITERT_BENCH_MODEL")
	}
	if err := litert.Load(lib); err != nil {
		b.Fatal(err)
	}
	raw, err := os.ReadFile(model)
	if err != nil {
		b.Fatal(err)
	}
	if litertlm.IsContainer(raw) {
		if raw, err = litertlm.SectionTFLite(raw); err != nil {
			b.Fatal(err)
		}
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		b.Fatal(err)
	}
	m, err := litert.OpenModelFromBuffer(env, raw)
	if err != nil {
		b.Fatal(err)
	}
	sig, err := m.Signature(0)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		m.Close()
		env.Close()
		runtime.KeepAlive(raw)
	})
	return sig
}

// BenchmarkNumInputs measures a 3-pin call returning an int (no result copy):
// the floor for "C transition + pinning".
func BenchmarkNumInputs(b *testing.B) {
	sig := benchSig(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sig.NumInputs(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkInputName measures a 4-pin call that also copies a C string out
// (goString) — pinning plus a result allocation.
func BenchmarkInputName(b *testing.B) {
	sig := benchSig(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sig.InputName(0); err != nil {
			b.Fatal(err)
		}
	}
}
