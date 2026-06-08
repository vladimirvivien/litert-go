package lm

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/vision"
)

// The vision encoder+adapter pipeline must run on a preprocessed image and yield
// adapter embeddings of shape [tokens, 1536] with finite values.
func TestVisionEncodeSmoke(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_LM_EMBED_MODEL")
	if lib == "" || model == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_EMBED_MODEL (a gemma-4 .litertlm with vision)")
	}
	eng, err := Open(lib, model, litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(eng.Close)

	v, err := eng.ensureVision()
	if err != nil {
		t.Skipf("model has no vision sections: %v", err)
	}

	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	for y := 0; y < 480; y++ {
		for x := 0; x < 640; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x ^ y), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	p, err := vision.Preprocess(buf.Bytes(), 70)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("patches=%d tokens=%d", p.Count, p.Tokens)

	mm, tokens, err := v.encode(eng.env, p)
	if err != nil {
		t.Fatal(err)
	}
	if tokens == 0 {
		t.Fatal("zero visual tokens")
	}
	const embDim = 1536
	if len(mm) != tokens*embDim {
		t.Fatalf("mm len %d != tokens*embDim %d (%d*%d)", len(mm), tokens*embDim, tokens, embDim)
	}
	for i, f := range mm {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatalf("non-finite embedding at %d: %v", i, f)
		}
	}
	t.Logf("mm_embedding: %d tokens x %d", tokens, embDim)
}

// End-to-end: a solid-color image plus a "what color" prompt should yield a
// color-aware reply. Qualitative (no C++ image oracle); the color match is
// logged, the run must succeed and be non-empty.
func TestGenerateFromImageSolidColor(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_LM_EMBED_MODEL")
	if lib == "" || model == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_EMBED_MODEL (a gemma-4 .litertlm with vision)")
	}
	eng, err := Open(lib, model, litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(eng.Close)
	if _, err := eng.ensureVision(); err != nil {
		t.Skipf("model has no vision sections: %v", err)
	}

	img := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 220, G: 20, B: 20, A: 255}) // solid red
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}

	out, err := eng.GenerateFromImage(
		"<start_of_image>What is the dominant color in this image? Answer in one word.",
		buf.Bytes(), 70, GenOptions{MaxTokens: 16})
	if err != nil {
		t.Fatal(err)
	}
	if out == "" {
		t.Fatal("empty reply")
	}
	t.Logf("reply: %q", out)
	if !strings.Contains(strings.ToLower(out), "red") {
		t.Fatalf("reply did not identify the red image: %q", out)
	}
}
