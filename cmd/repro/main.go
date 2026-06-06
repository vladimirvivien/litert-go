// Command repro stresses the ffi output-pointer path that is prone to memory
// corruption: load a model, then call LiteRtGetSignatureInputName (a C
// `const char** out` function) in a loop, forcing a GC each iteration with -gc.
// It watches previously-read names and a set of pre-allocated sentinel Go
// strings for corruption — a regression guard for the runtime.Pinner argument
// handling in the litert package.
//
//	repro -lib <libLiteRt dir> -model <.litertlm|.tflite> [-gc] [-sig N]
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

const sentinel = "SSSSSSSSSSSSS" // 13 bytes, same length as the KV names

func main() {
	lib := flag.String("lib", "", "libLiteRt directory (or LITERT_LIB)")
	modelPath := flag.String("model", "", ".litertlm container or .tflite")
	gc := flag.Bool("gc", false, "force runtime.GC() each iteration")
	sigIdx := flag.Int("sig", 0, "signature index to read")
	flag.Parse()
	if err := run(*lib, *modelPath, *sigIdx, *gc); err != nil {
		fmt.Fprintln(os.Stderr, "repro:", err)
		os.Exit(1)
	}
}

func run(lib, modelPath string, sigIdx int, gc bool) error {
	if err := litert.Load(lib); err != nil {
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
		if raw, err = litertlm.SectionTFLite(raw); err != nil {
			return err
		}
	}
	model, err := litert.OpenModelFromBuffer(env, raw)
	if err != nil {
		return err
	}
	defer model.Close()
	defer runtime.KeepAlive(raw)

	sig, err := model.Signature(sigIdx)
	if err != nil {
		return err
	}
	nin, err := sig.NumInputs()
	if err != nil {
		return err
	}

	// Sentinels: fresh Go-heap strings that must never change.
	sentinels := make([]string, 64)
	for i := range sentinels {
		sentinels[i] = strings.Clone(sentinel)
	}

	names := make([]string, 0, nin)
	for i := 0; i < nin; i++ {
		name, err := sig.InputName(i)
		if err != nil {
			return err
		}
		names = append(names, name)
		if gc {
			runtime.GC()
		}

		for j, s := range sentinels {
			if s != sentinel {
				fmt.Printf("FAIL: sentinel[%d]=%q after InputName(%d)\n", j, s, i)
				return nil
			}
		}
		for j := 0; j < len(names)-1; j++ {
			if strings.ContainsRune(names[j], 0) {
				fmt.Printf("FAIL: names[%d]=%q corrupted after InputName(%d) read %q\n", j, names[j], i, name)
				return nil
			}
		}
	}
	fmt.Printf("PASS: read %d input names from sig %d, no corruption (gc=%v)\n", len(names), sigIdx, gc)
	return nil
}
