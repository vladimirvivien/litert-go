package hftok_test

import (
	"os"
	"testing"

	"github.com/vladimirvivien/litert-go/hftok"
	"github.com/vladimirvivien/litert-go/litertlm"
)

// loadTok loads the HF tokenizer from the .litertlm at LITERT_HF_MODEL.
func loadTok(t *testing.T) *hftok.Tokenizer {
	t.Helper()
	path := os.Getenv("LITERT_HF_MODEL")
	if path == "" {
		t.Skip("set LITERT_HF_MODEL to a .litertlm with an HF tokenizer section")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sec, err := litertlm.SectionBytes(raw, litertlm.SectionHFTokenizerZlib)
	if err != nil {
		t.Fatalf("no HF tokenizer section: %v", err)
	}
	tok, err := hftok.LoadSection(sec)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// Byte-level BPE round-trips any UTF-8 text exactly.
func TestRoundTrip(t *testing.T) {
	tok := loadTok(t)
	cases := []string{
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog.",
		"  leading and  internal   spaces\tand\ttabs",
		"line one\nline two\n\nparagraph",
		"unicode: café, naïve, 日本語, emoji 🎉",
		"code: func main() { fmt.Println(\"hi\") }",
		"numbers 1234567890 and 3.14159",
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
	}
	for _, s := range cases {
		ids := tok.Encode(s)
		if len(ids) == 0 && s != "" {
			t.Errorf("empty encoding of %q", s)
		}
		got := tok.Decode(toInts(ids))
		if got != s {
			t.Errorf("round-trip mismatch:\n  in:  %q\n  out: %q\n  ids: %v", s, got, ids)
		}
	}
}

// Special tokens encode to their single known IDs.
func TestSpecialTokens(t *testing.T) {
	tok := loadTok(t)
	cases := map[string]int32{
		"<|endoftext|>": 151643,
		"<|im_start|>":  151644,
		"<|im_end|>":    151645,
	}
	for content, want := range cases {
		ids := tok.Encode(content)
		if len(ids) != 1 || ids[0] != want {
			t.Errorf("special %q: got %v, want [%d]", content, ids, want)
		}
	}
}

// A special token embedded in text is isolated as one ID, surrounding text
// tokenized normally.
func TestSpecialInline(t *testing.T) {
	tok := loadTok(t)
	ids := tok.Encode("hi<|im_end|>bye")
	var seen bool
	for _, id := range ids {
		if id == 151645 {
			seen = true
		}
	}
	if !seen {
		t.Fatalf("embedded <|im_end|> (151645) not found in %v", ids)
	}
	if got := tok.Decode(toInts(ids)); got != "hi<|im_end|>bye" {
		t.Fatalf("round-trip with special: %q", got)
	}
}

func toInts(ids []int32) []int {
	out := make([]int, len(ids))
	for i, v := range ids {
		out[i] = int(v)
	}
	return out
}
