package litert

import (
	"fmt"
	"runtime"
	"testing"
	"unsafe"

	"github.com/vladimirvivien/litert-go/litertlm"
)

func TestModelLoadingSignatureBothDLLs(t *testing.T) {
	dlls := map[string]string{
		"REFERENCE DLL (include/litertlm/lib)": "C:\\Users\\vladi\\include\\litertlm\\lib",
		"NEW DLL (litert-go/lib)":              "C:\\Users\\vladi\\DEV\\litert-go\\lib",
	}

	modelPath := "C:\\Users\\vladi\\models\\gemma-4-E2B-it.litertlm"
	tfliteBytes, err := litertlm.ReadTFLite(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("Model TFLite bytes size: %d\n", len(tfliteBytes))

	for name, libDir := range dlls {
		t.Run(name, func(t *testing.T) {
			// Clean up loaded state (force load by resetting package variables)
			loaded = false

			if err := Load(libDir); err != nil {
				t.Fatal(err)
			}

			env, err := NewEnvironment()
			if err != nil {
				t.Fatal(err)
			}
			defer env.Close()

			var pin runtime.Pinner
			defer pin.Unpin()

			pin.Pin(&tfliteBytes[0])
			datap := unsafe.Pointer(&tfliteBytes[0])
			pin.Pin(&datap)
			n := uint64(len(tfliteBytes))
			pin.Pin(&n)

			takesEnv := modelBufferTakesEnv()
			fmt.Printf("[%s] modelBufferTakesEnv() returned: %t\n", name, takesEnv)

			// Test with takesEnv = true
			{
				var m uintptr
				pin.Pin(&m)
				mp := unsafe.Pointer(&m)
				pin.Pin(&mp)
				e := uintptr(env)
				pin.Pin(&e)

				st := invoke(&pin, createModelFromBufferEnvFunc,
					unsafe.Pointer(&e), unsafe.Pointer(&datap), unsafe.Pointer(&n), unsafe.Pointer(&mp))
				fmt.Printf("[%s] Loading WITH env: status = %d (ok: %t)\n", name, st, st.Ok())
				if st.Ok() {
					Model(m).Close()
				}
			}

			// Test with takesEnv = false
			{
				var m uintptr
				pin.Pin(&m)
				mp := unsafe.Pointer(&m)
				pin.Pin(&mp)

				st := invoke(&pin, createModelFromBufferFunc,
					unsafe.Pointer(&datap), unsafe.Pointer(&n), unsafe.Pointer(&mp))
				fmt.Printf("[%s] Loading WITHOUT env: status = %d (ok: %t)\n", name, st, st.Ok())
				if st.Ok() {
					Model(m).Close()
				}
			}
		})
	}
}
