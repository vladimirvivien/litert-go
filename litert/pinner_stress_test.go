package litert_test

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// TestPinnerStress is a regression guard for the runtime.Pinner argument
// handling in the litert package: it drives LiteRtGetSignatureInputName (a C
// `const char** out` function) in a loop with a forced GC each iteration, and
// watches previously-read names plus pre-allocated sentinel strings for
// corruption from a stale pointer the C side wrote through after a stack or
// heap move. Set LITERT_LIB and LITERT_LM_MODEL (.litertlm or .tflite).
func TestPinnerStress(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	modelPath := os.Getenv("LITERT_LM_MODEL")
	if lib == "" || modelPath == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_MODEL")
	}
	if err := litert.Load(lib); err != nil {
		t.Fatal(err)
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	raw, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if litertlm.IsContainer(raw) {
		if raw, err = litertlm.SectionTFLite(raw); err != nil {
			t.Fatal(err)
		}
	}
	model, err := litert.OpenModelFromBuffer(env, raw)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	defer runtime.KeepAlive(raw)

	sig, err := model.Signature(0)
	if err != nil {
		t.Fatal(err)
	}
	nin, err := sig.NumInputs()
	if err != nil {
		t.Fatal(err)
	}

	const sentinel = "SSSSSSSSSSSSS" // 13 bytes, same length as the KV names
	sentinels := make([]string, 64)
	for i := range sentinels {
		sentinels[i] = strings.Clone(sentinel)
	}

	names := make([]string, 0, nin)
	for i := 0; i < nin; i++ {
		name, err := sig.InputName(i)
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, name)
		runtime.GC()

		for j, s := range sentinels {
			if s != sentinel {
				t.Fatalf("sentinel[%d] = %q after InputName(%d)", j, s, i)
			}
		}
		for j := 0; j < len(names)-1; j++ {
			if strings.ContainsRune(names[j], 0) {
				t.Fatalf("names[%d] = %q corrupted after InputName(%d) read %q", j, names[j], i, name)
			}
		}
	}
	t.Logf("read %d input names, no corruption", len(names))
}
