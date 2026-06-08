package audio

import (
	"math"
	"testing"
)

// naiveDFT is the O(n^2) reference: X[k] = sum_n x[n] exp(-2πi kn/N).
func naiveDFT(re []float64) (outRe, outIm []float64) {
	n := len(re)
	outRe = make([]float64, n)
	outIm = make([]float64, n)
	for k := 0; k < n; k++ {
		for t := 0; t < n; t++ {
			ang := -2.0 * math.Pi * float64(k) * float64(t) / float64(n)
			outRe[k] += re[t] * math.Cos(ang)
			outIm[k] += re[t] * math.Sin(ang)
		}
	}
	return outRe, outIm
}

// The hand-rolled FFT must match the reference DFT.
func TestFFTMatchesDFT(t *testing.T) {
	for _, n := range []int{8, 64, 512} {
		re := make([]float64, n)
		im := make([]float64, n)
		for i := range re {
			re[i] = math.Sin(float64(i)*0.3) + 0.5*math.Cos(float64(i)*0.07)
		}
		wantRe, wantIm := naiveDFT(re)
		fft(re, im)
		for k := 0; k < n; k++ {
			if math.Abs(re[k]-wantRe[k]) > 1e-6 || math.Abs(im[k]-wantIm[k]) > 1e-6 {
				t.Fatalf("n=%d bin %d: fft (%v,%v) != dft (%v,%v)", n, k, re[k], im[k], wantRe[k], wantIm[k])
			}
		}
	}
}

// Framing: prepend FrameLength-HopLength zeros, hop by HopLength, keep the final
// partial frame. First frame is [160 zeros, first 160 samples].
func TestFraming(t *testing.T) {
	pcm := make([]float32, 16000) // 1s
	for i := range pcm {
		pcm[i] = float32(i)
	}
	frames := frame(pcm)
	// effective length = 160 + 16000 = 16160; frames at hop 160 while start<16160.
	wantFrames := (16160 + HopLength - 1) / HopLength
	if len(frames) != wantFrames {
		t.Fatalf("frames = %d, want %d", len(frames), wantFrames)
	}
	// First frame: 160 zeros then pcm[0..159].
	for j := 0; j < FrameLength-HopLength; j++ {
		if frames[0][j] != 0 {
			t.Fatalf("frame0[%d] = %v, want 0 (semicausal pad)", j, frames[0][j])
		}
	}
	if frames[0][FrameLength-HopLength] != pcm[0] {
		t.Fatalf("frame0 first real sample = %v, want %v", frames[0][FrameLength-HopLength], pcm[0])
	}
}

// The Hann window is periodic and symmetric-ish: w[0]=0, peaks at the middle.
func TestHannWindow(t *testing.T) {
	w := hannWindow(FrameLength)
	if math.Abs(w[0]) > 1e-9 {
		t.Errorf("w[0] = %v, want 0", w[0])
	}
	mid := w[FrameLength/2]
	if math.Abs(mid-1.0) > 1e-9 {
		t.Errorf("w[mid] = %v, want 1", mid)
	}
}

// The mel filterbank weights must put each in-band FFT bin's energy entirely
// across two adjacent channels (weight + (1-weight) = 1), and span all channels.
func TestMelFilterbank(t *testing.T) {
	fb := filterbank
	if fb.start != 1 {
		t.Errorf("start = %d, want 1 (skip DC)", fb.start)
	}
	// Every in-band bin maps to a valid channel and weight in [0,1].
	for i := fb.start; i <= fb.end; i++ {
		if fb.weights[i] < -1e-9 || fb.weights[i] > 1.0+1e-9 {
			t.Errorf("weight[%d] = %v out of [0,1]", i, fb.weights[i])
		}
		ch := fb.bandMapper[i]
		if ch < -1 || ch >= NumMelBins {
			t.Errorf("bandMapper[%d] = %d out of range", i, ch)
		}
	}
}

// A pure tone should put nearly all mel energy in the channel covering its
// frequency. Verify the peak mel bin tracks the tone frequency monotonically.
func TestPureTonePeak(t *testing.T) {
	peakBin := func(freq float64) int {
		pcm := make([]float32, SampleRate) // 1s
		for i := range pcm {
			pcm[i] = float32(0.8 * math.Sin(2*math.Pi*freq*float64(i)/SampleRate))
		}
		ms := Preprocess(pcm)
		// Use a middle frame (steady state); find the max mel bin.
		fi := ms.Frames / 2
		best, bi := float32(math.Inf(-1)), 0
		for j := 0; j < NumMelBins; j++ {
			if v := ms.Mel[fi*NumMelBins+j]; v > best {
				best, bi = v, j
			}
		}
		return bi
	}
	low := peakBin(300)
	high := peakBin(3000)
	if !(low < high) {
		t.Fatalf("peak mel bin not monotonic in frequency: 300Hz->%d, 3000Hz->%d", low, high)
	}
}

// Preprocess output shape is consistent.
func TestPreprocessShape(t *testing.T) {
	pcm := make([]float32, SampleRate/2) // 0.5s
	ms := Preprocess(pcm)
	if ms.Frames == 0 {
		t.Fatal("no frames")
	}
	if len(ms.Mel) != ms.Frames*NumMelBins {
		t.Fatalf("Mel len %d != Frames*NumMelBins %d", len(ms.Mel), ms.Frames*NumMelBins)
	}
}
