// Command siginfo dumps a model's signatures and the shapes of their prefill
// input tensors — used to scope prefill bucketing/chunking.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/litertlm"
)

func main() {
	lib := flag.String("lib", os.Getenv("LITERT_LIB"), "dir with libLiteRt")
	model := flag.String("model", "", ".litertlm or .tflite")
	flag.Parse()

	if err := litert.Load(*lib); err != nil {
		panic(err)
	}
	env, err := litert.NewEnvironment()
	if err != nil {
		panic(err)
	}
	defer env.Close()

	raw, err := os.ReadFile(*model)
	if err != nil {
		panic(err)
	}
	tflite := raw
	if litertlm.IsContainer(raw) {
		if tflite, err = litertlm.SectionTFLite(raw); err != nil {
			panic(err)
		}
	}
	m, err := litert.OpenModelFromBuffer(env, tflite)
	if err != nil {
		panic(err)
	}
	defer m.Close()

	n, _ := m.NumSignatures()
	fmt.Printf("%d signatures:\n", n)
	for i := 0; i < n; i++ {
		s, _ := m.Signature(i)
		key, _ := s.Key()
		nin, _ := s.NumInputs()
		var names []string
		for j := 0; j < nin; j++ {
			name, _ := s.InputName(j)
			names = append(names, name)
		}
		fmt.Printf("  [%d] %-16s inputs: %s\n", i, key, strings.Join(names, ", "))
		for _, probe := range []string{"tokens", "embeddings", "input_pos", "mask"} {
			if tt, err := s.InputType(probe); err == nil {
				fmt.Printf("        %-12s %v\n", probe, tt.Shape)
			}
		}
	}
}
