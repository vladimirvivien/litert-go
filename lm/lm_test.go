package lm_test

import (
	"os"
	"strings"
	"testing"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/lm"
)

// openEngine loads the token-input chat model named by LITERT_LM_MODEL using the
// library in LITERT_LIB. The tests assert model-agnostic invariants (determinism,
// stream/Session equivalence), not specific text, so any small instruction-tuned
// token-input model works.
func openEngine(t *testing.T) *lm.Engine {
	t.Helper()
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_LM_MODEL")
	if lib == "" || model == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_MODEL (a token-input chat .litertlm)")
	}
	eng, err := lm.Open(lib, model, litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(eng.Close)
	return eng
}

// Greedy decoding (Temp 0) must be deterministic.
func TestGenerateGreedyDeterministic(t *testing.T) {
	eng := openEngine(t)
	o := lm.GenOptions{MaxTokens: 16}
	a, err := eng.Generate("The capital of France is", false, o)
	if err != nil {
		t.Fatal(err)
	}
	if a == "" {
		t.Fatal("empty greedy output")
	}
	b, err := eng.Generate("The capital of France is", false, o)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("greedy not deterministic:\n  %q\n  %q", a, b)
	}
}

// Streamed pieces must concatenate to exactly the non-streamed output.
func TestStreamMatchesGenerate(t *testing.T) {
	eng := openEngine(t)
	o := lm.GenOptions{MaxTokens: 16}
	full, err := eng.Generate("The capital of France is", false, o)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	ret, err := eng.GenerateStream("The capital of France is", false, o, func(p string) { sb.WriteString(p) })
	if err != nil {
		t.Fatal(err)
	}
	if ret != full {
		t.Fatalf("GenerateStream return %q != Generate %q", ret, full)
	}
	if sb.String() != full {
		t.Fatalf("streamed pieces %q != Generate %q", sb.String(), full)
	}
}

// A KV-reuse Session must produce exactly the same reply as Generate — the
// proof that ingesting through decode equals batched prefill.
func TestSessionMatchesGenerate(t *testing.T) {
	eng := openEngine(t)
	if !eng.HasChatTemplate() {
		t.Skip("model has no chat template")
	}
	o := lm.GenOptions{MaxTokens: 16}
	want, err := eng.Generate("Name one primary color.", true, o)
	if err != nil {
		t.Fatal(err)
	}
	sess, err := eng.NewSession(o)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	got, err := sess.Send("Name one primary color.")
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("Session %q != Generate %q (KV-reuse must be exact)", got, want)
	}
}

// The same seed must give the same sampled output.
func TestSeededSamplingDeterministic(t *testing.T) {
	eng := openEngine(t)
	if !eng.HasChatTemplate() {
		t.Skip("model has no chat template")
	}
	o := lm.GenOptions{MaxTokens: 16, Temp: 0.8, TopK: 40, Seed: 7}
	a, err := eng.Generate("Write a short sentence about the sea.", true, o)
	if err != nil {
		t.Fatal(err)
	}
	b, err := eng.Generate("Write a short sentence about the sea.", true, o)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("same seed not deterministic:\n  %q\n  %q", a, b)
	}
}

// Multi-turn via NewConversation: both turns produce output and retain context.
func TestMultiTurn(t *testing.T) {
	eng := openEngine(t)
	if !eng.HasChatTemplate() {
		t.Skip("model has no chat template")
	}
	conv, err := eng.NewConversation(lm.GenOptions{MaxTokens: 32})
	if err != nil {
		t.Fatal(err)
	}
	defer conv.Close()
	if r1, err := conv.Send("My name is Vlad. Remember it."); err != nil {
		t.Fatal(err)
	} else if r1 == "" {
		t.Fatal("empty turn 1")
	}
	r2, err := conv.Send("What is my name?")
	if err != nil {
		t.Fatal(err)
	}
	if r2 == "" {
		t.Fatal("empty turn 2")
	}
	if !strings.Contains(r2, "Vlad") {
		t.Logf("note: turn-2 reply did not echo the name (model-dependent): %q", r2)
	}
}
