// Package audio turns 16 kHz mono PCM into the log-mel spectrogram the gemma-4
// audio encoder consumes. The pipeline mirrors LiteRT-LM's
// AudioPreprocessorMiniaudio / MelFilterbank: semicausal framing, a periodic
// Hann window, center-padding to the FFT length, a real FFT power spectrum, a
// triangular mel filterbank, and a log with a floor.
package audio

import "math"

const (
	SampleRate  = 16000
	FrameLength = 320
	HopLength   = 160
	FFTLength   = 512
	FFTBins     = FFTLength/2 + 1 // 257
	NumMelBins  = 128
	MelLowHz    = 0.0
	MelHighHz   = 8000.0
	MelFloor    = 1e-3
)

// MelSpectrogram is a log-mel spectrogram: Frames rows of NumMelBins, row-major.
type MelSpectrogram struct {
	Mel    []float32 // [Frames*NumMelBins]
	Frames int
}

var filterbank = newMelFilterbank()

// Preprocess computes the log-mel spectrogram of 16 kHz mono PCM samples.
func Preprocess(pcm []float32) *MelSpectrogram {
	frames := frame(pcm)
	window := hannWindow(FrameLength)
	out := &MelSpectrogram{Frames: len(frames), Mel: make([]float32, len(frames)*NumMelBins)}

	buf := make([]float64, FFTLength)
	re := make([]float64, FFTLength)
	im := make([]float64, FFTLength)
	power := make([]float64, FFTBins)
	mel := make([]float64, NumMelBins)
	for fi, fr := range frames {
		// Window, then center-pad to the FFT length.
		clear(buf)
		const padLeft = (FFTLength - FrameLength) / 2
		for j := 0; j < FrameLength; j++ {
			buf[padLeft+j] = float64(fr[j]) * window[j]
		}
		copy(re, buf)
		clear(im)
		fft(re, im)
		for k := 0; k < FFTBins; k++ {
			power[k] = re[k]*re[k] + im[k]*im[k]
		}
		filterbank.toMel(power, mel)
		for j := 0; j < NumMelBins; j++ {
			out.Mel[fi*NumMelBins+j] = float32(math.Log(mel[j] + MelFloor))
		}
	}
	return out
}

// frame splits pcm into FrameLength windows hopped by HopLength. Semicausal
// padding prepends FrameLength-HopLength zeros (center framing); the final
// partial window is zero-padded and kept.
func frame(pcm []float32) [][]float32 {
	prepad := FrameLength - HopLength
	eff := make([]float32, prepad+len(pcm))
	copy(eff[prepad:], pcm)

	var frames [][]float32
	for start := 0; start < len(eff); start += HopLength {
		f := make([]float32, FrameLength)
		copy(f, eff[start:min(start+FrameLength, len(eff))])
		frames = append(frames, f)
	}
	return frames
}

// hannWindow returns the periodic Hann window of length n:
// w[i] = 0.5 - 0.5*cos(2*pi*i/n).
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	arg := 2.0 * math.Pi / float64(n)
	for i := 0; i < n; i++ {
		w[i] = 0.5 - 0.5*math.Cos(arg*float64(i))
	}
	return w
}

// fft is an in-place iterative radix-2 Cooley-Tukey forward FFT; len(re) must be
// a power of two. Unnormalized, matching kiss_fftr's magnitudes.
func fft(re, im []float64) {
	n := len(re)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := -2.0 * math.Pi / float64(length)
		wr, wi := math.Cos(ang), math.Sin(ang)
		half := length / 2
		for i := 0; i < n; i += length {
			cr, ci := 1.0, 0.0
			for k := 0; k < half; k++ {
				tr := re[i+k+half]*cr - im[i+k+half]*ci
				ti := re[i+k+half]*ci + im[i+k+half]*cr
				re[i+k+half] = re[i+k] - tr
				im[i+k+half] = im[i+k] - ti
				re[i+k] += tr
				im[i+k] += ti
				cr, ci = cr*wr-ci*wi, cr*wi+ci*wr
			}
		}
	}
}

// melFilterbank holds the triangular FFT-bin -> mel-channel weights.
type melFilterbank struct {
	start, end int
	bandMapper []int     // FFT bin -> right-triangle channel (-2 if unused)
	weights    []float64 // FFT bin -> right-triangle weight
}

func freqToMel(f float64) float64 { return 1127.0 * math.Log(1.0+f/700.0) }

func newMelFilterbank() *melFilterbank {
	melLow := freqToMel(MelLowHz)
	melHi := freqToMel(MelHighHz)
	melSpacing := (melHi - melLow) / float64(NumMelBins+1)
	hzPerBin := SampleRate / (2.0 * float64(FFTBins-1))

	fb := &melFilterbank{
		start:      int(1.5 + MelLowHz/hzPerBin),
		end:        int(MelHighHz / hzPerBin),
		bandMapper: make([]int, FFTBins),
		weights:    make([]float64, FFTBins),
	}
	for i := 0; i < FFTBins; i++ {
		if i < fb.start || i > fb.end {
			fb.bandMapper[i] = -2
			continue
		}
		melPos := (freqToMel(float64(i)*hzPerBin)-melLow)/melSpacing - 1
		ch := int(math.Ceil(melPos)) - 1
		fb.bandMapper[i] = ch
		fb.weights[i] = 1.0 - (melPos - float64(ch))
	}
	return fb
}

// toMel integrates the power spectrum into mel channels via the triangular
// filters (each FFT bin feeds the right slope of one channel and the left slope
// of the next).
func (fb *melFilterbank) toMel(power, mel []float64) {
	clear(mel)
	for i := fb.start; i <= fb.end; i++ {
		spec := math.Sqrt(power[i])
		weighted := spec * fb.weights[i]
		ch := fb.bandMapper[i]
		if ch >= 0 {
			mel[ch] += weighted
		}
		ch++
		if ch < NumMelBins {
			mel[ch] += spec - weighted
		}
	}
}
