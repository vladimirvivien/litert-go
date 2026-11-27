package audio_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"sync"
	"testing"

	"github.com/vladimirvivien/litert-go/pkg/audio"
)

func buildWAV(format uint16, sampleRate, channels, bits int, payload []byte, extraChunk bool) []byte {
	var body bytes.Buffer
	body.WriteString("WAVE")

	if extraChunk {
		body.WriteString("JUNK")
		_ = binary.Write(&body, binary.LittleEndian, uint32(3))
		body.Write([]byte{1, 2, 3, 0}) // odd chunk + 1 pad byte
	}

	body.WriteString("fmt ")
	_ = binary.Write(&body, binary.LittleEndian, uint32(16))
	_ = binary.Write(&body, binary.LittleEndian, format)
	_ = binary.Write(&body, binary.LittleEndian, uint16(channels))
	_ = binary.Write(&body, binary.LittleEndian, uint32(sampleRate))
	byteRate := uint32(sampleRate * channels * (bits / 8))
	blockAlign := uint16(channels * (bits / 8))
	_ = binary.Write(&body, binary.LittleEndian, byteRate)
	_ = binary.Write(&body, binary.LittleEndian, blockAlign)
	_ = binary.Write(&body, binary.LittleEndian, uint16(bits))

	body.WriteString("data")
	_ = binary.Write(&body, binary.LittleEndian, uint32(len(payload)))
	body.Write(payload)

	var riff bytes.Buffer
	riff.WriteString("RIFF")
	_ = binary.Write(&riff, binary.LittleEndian, uint32(body.Len()))
	riff.Write(body.Bytes())
	return riff.Bytes()
}

func TestDecodeWAVFormatsAndResample(t *testing.T) {
	// PCM8 unsigned.
	p8, err := audio.DecodeWAVBytes(buildWAV(1, 8000, 1, 8, []byte{0, 128, 255}, false))
	if err != nil {
		t.Fatal(err)
	}
	if p8.Samples[1] != 0 || p8.Samples[0] != -1.0 {
		t.Fatalf("PCM8 = %v", p8.Samples)
	}

	// PCM16 stereo with odd JUNK chunk.
	var p16Buf bytes.Buffer
	for _, s := range []int16{16384, -16384, 32767, 0} {
		_ = binary.Write(&p16Buf, binary.LittleEndian, s)
	}
	p16, err := audio.DecodeWAV(bytes.NewReader(buildWAV(1, 8000, 2, 16, p16Buf.Bytes(), true)))
	if err != nil {
		t.Fatal(err)
	}
	mono := audio.PreparePCM(p16, 16000)
	if len(mono) != 4 {
		t.Fatalf("resampled len = %d, want 4", len(mono))
	}
	if math.Abs(float64(mono[0])) > 1e-5 {
		t.Fatalf("expected balanced stereo downmix to ~0, got %v", mono[0])
	}

	// PCM24 signed LE (-1.0, 0.0, ~+0.5).
	p24Bytes := []byte{
		0x00, 0x00, 0x80, // -8388608 -> -1.0
		0x00, 0x00, 0x00, // 0
		0x00, 0x00, 0x40, // +4194304 -> +0.5
	}
	p24, err := audio.DecodeWAVBytes(buildWAV(1, 16000, 1, 24, p24Bytes, false))
	if err != nil {
		t.Fatal(err)
	}
	if p24.Samples[0] != -1.0 || p24.Samples[1] != 0 || p24.Samples[2] != 0.5 {
		t.Fatalf("PCM24 = %v", p24.Samples)
	}

	// IEEE float32.
	var f32Buf bytes.Buffer
	_ = binary.Write(&f32Buf, binary.LittleEndian, float32(0.25))
	_ = binary.Write(&f32Buf, binary.LittleEndian, float32(-0.75))
	pf, err := audio.DecodeWAVBytes(buildWAV(3, 16000, 1, 32, f32Buf.Bytes(), false))
	if err != nil {
		t.Fatal(err)
	}
	if pf.Samples[0] != 0.25 || pf.Samples[1] != -0.75 {
		t.Fatalf("IEEE f32 = %v", pf.Samples)
	}

	if _, badErr := audio.DecodeWAVBytes([]byte("NOT_RIFF")); badErr == nil {
		t.Fatal("expected invalid header error")
	}
}

func TestMelSpectrogramPeakAndConcurrency(t *testing.T) {
	cfg := audio.DefaultYAMNetConfig()
	ex, err := audio.NewMelExtractor(cfg)
	if err != nil {
		t.Fatal(err)
	}

	makeSine := func(freqHz float64) []float32 {
		samples := make([]float32, 3200) // 200 ms at 16 kHz
		for i := range samples {
			samples[i] = float32(math.Sin(2.0 * math.Pi * freqHz * float64(i) / float64(cfg.SampleRate)))
		}
		return samples
	}

	peakMelBin := func(feats []float32, frames int) int {
		sums := make([]float64, cfg.NumMelBins)
		for f := range frames {
			for m := range cfg.NumMelBins {
				sums[m] += float64(feats[f*cfg.NumMelBins+m])
			}
		}
		best := 0
		for m := 1; m < cfg.NumMelBins; m++ {
			if sums[m] > sums[best] {
				best = m
			}
		}
		return best
	}

	lowFeats, lowFrames := ex.Extract(makeSine(440))
	highFeats, highFrames, err := audio.MelSpectrogram(makeSine(2000), audio.DefaultWhisperConfig())
	if err != nil {
		t.Fatal(err)
	}
	if lowFrames == 0 || highFrames == 0 || len(highFeats) != highFrames*80 {
		t.Fatalf("unexpected frame counts low=%d high=%d", lowFrames, highFrames)
	}

	yamHighFeats, yamHighFrames := ex.Extract(makeSine(2000))
	lowPeak := peakMelBin(lowFeats, lowFrames)
	highPeak := peakMelBin(yamHighFeats, yamHighFrames)
	if lowPeak >= highPeak {
		t.Fatalf("expected 440 Hz peak (%d) < 2000 Hz peak (%d)", lowPeak, highPeak)
	}

	// Verify concurrent Extract calls are race-free.
	s440 := makeSine(440)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			got, gotFrames := ex.Extract(s440)
			if gotFrames != lowFrames || len(got) != len(lowFeats) {
				t.Errorf("concurrent extract mismatch frames=%d len=%d", gotFrames, len(got))
			}
		})
	}
	wg.Wait()
}
