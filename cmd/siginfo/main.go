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
	section := flag.String("section", "", "inspect the TFLiteModel section with this model_type hint (e.g. tf_lite_vision_encoder)")
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
		secs, err := litertlm.Sections(raw)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%d sections:\n", len(secs))
		for _, s := range secs {
			fmt.Printf("  %-18s items=%v\n", s.TypeName(), s.Items)
		}
		if md, merr := litertlm.ReadMetadata(raw); merr == nil {
			fmt.Printf("metadata: model_type=%s start=%+v stops=%+v hasPrompts=%v\n",
				md.ModelType, md.StartToken, md.StopTokens, md.HasPrompts)
			if tpl, ok := md.Templates(); ok {
				fmt.Printf("  user:   prefix=%q suffix=%q\n", tpl.User.Prefix, tpl.User.Suffix)
				fmt.Printf("  model:  prefix=%q suffix=%q\n", tpl.Model.Prefix, tpl.Model.Suffix)
				fmt.Printf("  system: prefix=%q suffix=%q\n", tpl.System.Prefix, tpl.System.Suffix)
			} else {
				fmt.Println("  (no chat template)")
			}
		}
		if *section != "" {
			if tflite, err = litertlm.SectionTFLiteModelType(raw, *section); err != nil {
				panic(err)
			}
		} else if tflite, err = litertlm.SectionTFLite(raw); err != nil {
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
		fmt.Printf("  [%d] %-16s (%d inputs)\n", i, key, nin)
		isKV := func(s string) bool { return strings.HasPrefix(s, "kv_cache_") }
		for _, name := range names {
			if isKV(name) {
				continue
			}
			if tt, err := s.InputType(name); err == nil {
				dyn := ""
				for _, d := range tt.Shape {
					if d <= 0 {
						dyn = "  <-- DYNAMIC"
					}
				}
				fmt.Printf("        in  %-22s %v %v%s\n", name, tt.ElementType, tt.Shape, dyn)
			}
		}
		nout, _ := s.NumOutputs()
		for j := 0; j < nout; j++ {
			name, _ := s.OutputName(j)
			if isKV(name) {
				continue
			}
			if tt, err := s.OutputType(name); err == nil {
				fmt.Printf("        out %-22s %v %v\n", name, tt.ElementType, tt.Shape)
			}
		}
	}
}
