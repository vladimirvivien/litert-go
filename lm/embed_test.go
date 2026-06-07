package lm_test

import (
	"os"
	"strings"
	"testing"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/lm"
)

// openEmbedEngine loads the embedding-input chat model named by
// LITERT_LM_EMBED_MODEL (gemma 3n/4) using the library in LITERT_LIB. These
// tests assert that the embedSession's prefill-at-offset ingest matches the
// reference paths (Generate for turn 1, a re-prefill Chat across turns).
func openEmbedEngine(t *testing.T) *lm.Engine {
	t.Helper()
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_LM_EMBED_MODEL")
	if lib == "" || model == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_EMBED_MODEL (an embedding-input chat .litertlm)")
	}
	eng, err := lm.Open(lib, model, litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(eng.Close)
	if !eng.HasChatTemplate() {
		t.Skip("embedding model has no chat template")
	}
	return eng
}

// Generate must be re-runnable on one Engine: the embedder stages are compiled
// once and reused, so a second call neither fails nor changes output.
func TestEmbedGenerateReentrant(t *testing.T) {
	eng := openEmbedEngine(t)
	o := lm.GenOptions{MaxTokens: 16}
	a, err := eng.Generate("Name one primary color.", true, o)
	if err != nil {
		t.Fatal(err)
	}
	b, err := eng.Generate("Name one primary color.", true, o)
	if err != nil {
		t.Fatalf("second Generate failed (embedders must be reusable): %v", err)
	}
	if a != b {
		t.Fatalf("Generate not deterministic across calls:\n  %q\n  %q", a, b)
	}
}

// A re-prefill Chat must run multiple turns on an embedding model (each Send is
// a fresh decodeEmbeddingInput over the re-rendered history).
func TestEmbedChatMultiTurn(t *testing.T) {
	eng := openEmbedEngine(t)
	chat, err := eng.NewChat(lm.GenOptions{MaxTokens: 24})
	if err != nil {
		t.Fatal(err)
	}
	defer chat.Close()
	if r1, err := chat.Send("My name is Vlad. Remember it."); err != nil {
		t.Fatalf("turn 1: %v", err)
	} else if r1 == "" {
		t.Fatal("empty turn 1")
	}
	r2, err := chat.Send("What is my name?")
	if err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if r2 == "" {
		t.Fatal("empty turn 2")
	}
}

// Turn 1 of an embedding-input Session ingests the prompt with a
// prefill-at-offset of 0 — the same prefill Generate runs — so its reply must
// equal Generate's exactly.
func TestEmbedSessionMatchesGenerate(t *testing.T) {
	eng := openEmbedEngine(t)
	o := lm.GenOptions{MaxTokens: 16}
	want, err := eng.Generate("Name one primary color.", true, o)
	if err != nil {
		t.Fatal(err)
	}
	conv, err := eng.NewConversation(o)
	if err != nil {
		t.Fatal(err)
	}
	defer conv.Close()
	got, err := conv.Send("Name one primary color.")
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("embedding Session turn 1 %q != Generate %q (prefill-at-offset-0 must be exact)", got, want)
	}
}

// Across turns, the embedding-input Session ingests each new turn with a
// prefill-at-offset into the retained KV cache. Both turns must produce output
// and the second must keep the first turn's context.
func TestEmbedSessionMultiTurn(t *testing.T) {
	eng := openEmbedEngine(t)
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
		t.Fatalf("turn-2 reply lost context (expected the name): %q", r2)
	}
}
