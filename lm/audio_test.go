package lm

import (
	"math"
	"os"
	"testing"

	"github.com/vladimirvivien/litert-go/audio"
	"github.com/vladimirvivien/litert-go/litert"
)

// The audio encoder+adapter pipeline must run on a mel spectrogram and yield
// adapter embeddings of shape [tokens, 1536] with finite values.
func TestAudioEncodeSmoke(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_LM_EMBED_MODEL")
	if lib == "" || model == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_EMBED_MODEL (a gemma-4 .litertlm with audio)")
	}
	eng, err := Open(lib, model, litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(eng.Close)

	ap, err := eng.ensureAudio()
	if err != nil {
		t.Skipf("model has no audio sections: %v", err)
	}

	// ~2s of a 440 Hz tone.
	pcm := make([]float32, audio.SampleRate*2)
	for i := range pcm {
		pcm[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/audio.SampleRate))
	}
	mel := audio.Preprocess(pcm)
	t.Logf("mel frames=%d", mel.Frames)

	emb, tokens, err := ap.encode(eng.env, mel)
	if err != nil {
		t.Fatal(err)
	}
	if tokens == 0 {
		t.Fatal("zero audio tokens")
	}
	const embDim = 1536
	if len(emb) != tokens*embDim {
		t.Fatalf("emb len %d != tokens*embDim %d (%d*%d)", len(emb), tokens*embDim, tokens, embDim)
	}
	for i, f := range emb {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatalf("non-finite embedding at %d: %v", i, f)
		}
	}
	t.Logf("audio embedding: %d tokens x %d", tokens, embDim)
}

// End-to-end: GenerateFromAudio must splice the audio embeddings and produce a
// non-empty reply. Qualitative understanding needs a real clip (a tone is not
// semantically recognizable); here we assert it runs and replies.
func TestGenerateFromAudioRuns(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_LM_EMBED_MODEL")
	if lib == "" || model == "" {
		t.Skip("set LITERT_LIB and LITERT_LM_EMBED_MODEL (a gemma-4 .litertlm with audio)")
	}
	eng, err := Open(lib, model, litert.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(eng.Close)
	if _, err := eng.ensureAudio(); err != nil {
		t.Skipf("model has no audio sections: %v", err)
	}

	pcm := make([]float32, audio.SampleRate*2) // 2s 440 Hz tone
	for i := range pcm {
		pcm[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/audio.SampleRate))
	}
	out, err := eng.GenerateFromAudio("<start_of_audio>Describe what you hear in one sentence.", pcm, GenOptions{MaxTokens: 32})
	if err != nil {
		t.Fatal(err)
	}
	if out == "" {
		t.Fatal("empty reply")
	}
	t.Logf("reply: %q", out)
}
