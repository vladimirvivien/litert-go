package lm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vladimirvivien/litert-go/litert"
)

// TestModelMatrix runs the model-matrix battery against every .litertlm model
// under LITERT_LM_MODELS (a directory, or a single .litertlm file). Per model:
// greedy generate (chat when the model has a template, raw completion
// otherwise), stream ≡ generate, tokenize round-trip, and multi-turn
// conversation. The greedy output and timing are logged so runs on different
// backends can be diffed.
//
//	LITERT_LIB         runtime library directory
//	LITERT_LM_MODELS   directory of .litertlm files, or one file
//	LITERT_LM_BACKEND  cpu (default) or gpu
func TestModelMatrix(t *testing.T) {
	runModelMatrix(t)
}

// envAccel selects the test backend from LITERT_LM_BACKEND (cpu default).
func envAccel() litert.HwAccelerator {
	if strings.EqualFold(os.Getenv("LITERT_LM_BACKEND"), "gpu") {
		return litert.AccelGPU
	}
	return litert.AccelCPU
}

func runModelMatrix(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	models := os.Getenv("LITERT_LM_MODELS")
	if lib == "" || models == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_MODELS (a .litertlm file or a directory of them)")
	}
	accel := envAccel()

	var files []string
	if fi, err := os.Stat(models); err == nil && fi.IsDir() {
		files, err = filepath.Glob(filepath.Join(models, "*.litertlm"))
		if err != nil || len(files) == 0 {
			t.Fatalf("no .litertlm files under %s", models)
		}
	} else {
		files = []string{models}
	}

	for _, f := range files {
		t.Run(filepath.Base(f), func(t *testing.T) {
			if reason, broken := upstreamBroken[filepath.Base(f)]; broken {
				t.Skip(reason)
			}
			matrixModel(t, lib, f, accel)
		})
	}
}

// upstreamBroken lists zoo models the C++ LiteRT-LM engine cannot run
// either — failures on these are parity, not litert-go defects.
var upstreamBroken = map[string]string{
	"gemma-4-12B-it.litertlm": "fails on the C++ engine too (engine_create error as of v0.13.1); litert-go fails at prefill",
}

func matrixModel(t *testing.T, lib, modelPath string, accel litert.HwAccelerator) {
	t.Helper()
	opts := []Option{WithLibDir(lib), WithAccelerator(accel)}
	if os.Getenv("LITERT_DECODE_STATS") != "" {
		opts = append(opts, WithMetrics(func(s DecodeStats) {
			if s.Tokens == 0 {
				return
			}
			fmt.Fprintf(os.Stderr, "decode: %d tokens in %v (%.1f tok/s)\n",
				s.Tokens, s.Decode.Round(time.Millisecond), s.TokensPerSecond())
		}))
	}
	eng, err := Open(context.Background(), modelPath, opts...)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(eng.Close)

	chat := eng.HasChatTemplate()
	prompt := "The capital of France is"
	if chat {
		prompt = "Name one primary color."
	}
	o := GenOptions{MaxTokens: 16}

	t.Run("generate", func(t *testing.T) {
		t0 := time.Now()
		out, err := eng.Generate(context.Background(), prompt, chat, o)
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		if out == "" {
			t.Fatal("empty greedy output")
		}
		t.Logf("greedy (%v, chat=%v): %q", time.Since(t0).Round(time.Millisecond), chat, out)

		var sb strings.Builder
		ret, err := eng.GenerateStream(context.Background(), prompt, chat, o, func(p string) { sb.WriteString(p) })
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if ret != out || sb.String() != out {
			t.Fatalf("stream %q / pieces %q != generate %q", ret, sb.String(), out)
		}
	})

	t.Run("tokenize-roundtrip", func(t *testing.T) {
		if eng.tok == nil {
			t.Skip("no tokenizer")
		}
		const s = "The quick brown fox jumps over 12 lazy dogs."
		ids := eng.tok.Encode(s)
		if len(ids) == 0 {
			t.Fatal("Encode returned no tokens")
		}
		ints := make([]int, len(ids))
		for i, id := range ids {
			ints[i] = int(id)
		}
		got := eng.tok.Decode(ints)
		if strings.TrimSpace(got) != s {
			t.Fatalf("round-trip mismatch:\n  in:  %q\n  out: %q", s, got)
		}
	})

	t.Run("multi-turn", func(t *testing.T) {
		if !chat {
			t.Skip("no chat template")
		}
		conv, err := eng.NewConversation(GenOptions{MaxTokens: 24})
		if err != nil {
			t.Fatalf("conversation: %v", err)
		}
		defer conv.Close()
		r1, err := conv.Send(context.Background(), "My name is Vlad. Remember it.")
		if err != nil {
			t.Fatalf("turn 1: %v", err)
		}
		if r1 == "" {
			t.Fatal("empty turn 1")
		}
		r2, err := conv.Send(context.Background(), "What is my name?")
		if err != nil {
			t.Fatalf("turn 2: %v", err)
		}
		if r2 == "" {
			t.Fatal("empty turn 2")
		}
		recall := "ok"
		if !strings.Contains(r2, "Vlad") {
			recall = "missed (model-dependent)"
		}
		t.Logf("turn 2 (%s): %q", recall, r2)
	})
}
