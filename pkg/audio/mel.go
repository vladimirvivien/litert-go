package audio

import (
	"fmt"
	"math"
)

// MelConfig configures sliding-window STFT and log-mel spectrogram extraction.
type MelConfig struct {
	SampleRate     int
	FrameLength    int
	HopLength      int
	FFTLength      int
	NumMelBins     int
	MelLowHz       float64
	MelHighHz      float64
	LogFloor       float64
	PeriodicWindow bool
}

// DefaultYAMNetConfig returns the standard 16 kHz / 64-mel configuration
// (25 ms frame, 10 ms hop, 512-point FFT, 125–7500 Hz).
func DefaultYAMNetConfig() MelConfig {
	return MelConfig{
		SampleRate:     16000,
		FrameLength:    400,
		HopLength:      160,
		FFTLength:      512,
		NumMelBins:     64,
		MelLowHz:       125.0,
		MelHighHz:      7500.0,
		LogFloor:       0.001,
		PeriodicWindow: true,
	}
}

// DefaultWhisperConfig returns a 16 kHz / 80-mel configuration (512-point radix-2 FFT).
func DefaultWhisperConfig() MelConfig {
	return MelConfig{
		SampleRate:     16000,
		FrameLength:    400,
		HopLength:      160,
		FFTLength:      512,
		NumMelBins:     80,
		MelLowHz:       0.0,
		MelHighHz:      8000.0,
		LogFloor:       1e-10,
		PeriodicWindow: true,
	}
}

// MelExtractor computes row-major [numFrames x NumMelBins] log-mel spectrograms
// using a precomputed Hann window, radix-2 FFT, and triangular mel filterbank.
// Extract is safe for concurrent use across goroutines.
type MelExtractor struct {
	cfg        MelConfig
	numBins    int
	window     []float64
	filterbank []float64
}

// NewMelExtractor validates cfg and precomputes the Hann window and mel filterbank.
func NewMelExtractor(cfg MelConfig) (*MelExtractor, error) {
	if cfg.SampleRate <= 0 {
		return nil, fmt.Errorf("audio: SampleRate must be positive, got %d", cfg.SampleRate)
	}
	if cfg.FrameLength <= 0 || cfg.HopLength <= 0 {
		return nil, fmt.Errorf("audio: FrameLength (%d) and HopLength (%d) must be positive", cfg.FrameLength, cfg.HopLength)
	}
	if cfg.FFTLength < cfg.FrameLength || (cfg.FFTLength&(cfg.FFTLength-1)) != 0 {
		return nil, fmt.Errorf("audio: FFTLength (%d) must be a power of 2 >= FrameLength (%d)", cfg.FFTLength, cfg.FrameLength)
	}
	if cfg.NumMelBins <= 0 {
		return nil, fmt.Errorf("audio: NumMelBins must be positive, got %d", cfg.NumMelBins)
	}
	if cfg.MelHighHz <= 0 {
		cfg.MelHighHz = float64(cfg.SampleRate) / 2.0
	}
	if cfg.MelLowHz < 0 || cfg.MelHighHz <= cfg.MelLowHz {
		return nil, fmt.Errorf("audio: invalid mel frequency range [%v, %v]", cfg.MelLowHz, cfg.MelHighHz)
	}
	if cfg.LogFloor <= 0 {
		cfg.LogFloor = 1e-6
	}

	window := make([]float64, cfg.FrameLength)
	if cfg.FrameLength == 1 {
		window[0] = 1.0
	} else {
		denom := float64(cfg.FrameLength - 1)
		if cfg.PeriodicWindow {
			denom = float64(cfg.FrameLength)
		}
		for i := range cfg.FrameLength {
			window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/denom))
		}
	}

	numBins := cfg.FFTLength/2 + 1
	filterbank := buildMelFilterbank(cfg.NumMelBins, numBins, cfg.SampleRate, cfg.MelLowHz, cfg.MelHighHz)

	return &MelExtractor{
		cfg:        cfg,
		numBins:    numBins,
		window:     window,
		filterbank: filterbank,
	}, nil
}

// Config returns the normalized MelConfig used by the extractor.
func (m *MelExtractor) Config() MelConfig { return m.cfg }

// Extract computes the log-mel spectrogram for mono PCM samples. It returns
// a flat row-major [numFrames * NumMelBins] slice and the frame count.
func (m *MelExtractor) Extract(samples []float32) ([]float32, int) {
	if len(samples) < m.cfg.FrameLength {
		return nil, 0
	}
	numFrames := 1 + (len(samples)-m.cfg.FrameLength)/m.cfg.HopLength
	out := make([]float32, numFrames*m.cfg.NumMelBins)

	re := make([]float64, m.cfg.FFTLength)
	im := make([]float64, m.cfg.FFTLength)
	mag := make([]float64, m.numBins)

	for f := range numFrames {
		start := f * m.cfg.HopLength
		for i := range m.cfg.FrameLength {
			re[i] = float64(samples[start+i]) * m.window[i]
		}
		clear(re[m.cfg.FrameLength:])
		clear(im)

		fftRadix2(re, im)
		for b := range m.numBins {
			mag[b] = math.Hypot(re[b], im[b])
		}

		dstFrame := out[f*m.cfg.NumMelBins:]
		for melIdx := range m.cfg.NumMelBins {
			fbRow := m.filterbank[melIdx*m.numBins:]
			var energy float64
			for b := range m.numBins {
				energy += fbRow[b] * mag[b]
			}
			dstFrame[melIdx] = float32(math.Log(energy + m.cfg.LogFloor))
		}
	}
	return out, numFrames
}

// MelSpectrogram validates cfg and computes a log-mel spectrogram in one call.
func MelSpectrogram(samples []float32, cfg MelConfig) ([]float32, int, error) {
	ex, err := NewMelExtractor(cfg)
	if err != nil {
		return nil, 0, err
	}
	feats, frames := ex.Extract(samples)
	return feats, frames, nil
}

func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

func buildMelFilterbank(numMels, numBins, sampleRate int, lowHz, highHz float64) []float64 {
	lowMel := hzToMel(lowHz)
	highMel := hzToMel(highHz)
	edgeHz := make([]float64, numMels+2)
	for i := range edgeHz {
		m := lowMel + (highMel-lowMel)*float64(i)/float64(numMels+1)
		edgeHz[i] = melToHz(m)
	}

	fftLen := (numBins - 1) * 2
	binHz := float64(sampleRate) / float64(fftLen)

	fb := make([]float64, numMels*numBins)
	for m := range numMels {
		left, center, right := edgeHz[m], edgeHz[m+1], edgeHz[m+2]
		row := fb[m*numBins:]
		for b := range numBins {
			f := float64(b) * binHz
			switch {
			case f >= left && f <= center && center > left:
				row[b] = (f - left) / (center - left)
			case f > center && f <= right && right > center:
				row[b] = (right - f) / (right - center)
			}
		}
	}
	return fb
}

func fftRadix2(re, im []float64) {
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

	for stageLen := 2; stageLen <= n; stageLen <<= 1 {
		half := stageLen >> 1
		angle := -2.0 * math.Pi / float64(stageLen)
		wStepRe := math.Cos(angle)
		wStepIm := math.Sin(angle)
		for base := 0; base < n; base += stageLen {
			wRe, wIm := 1.0, 0.0
			for k := range half {
				even := base + k
				odd := even + half
				tr := wRe*re[odd] - wIm*im[odd]
				ti := wRe*im[odd] + wIm*re[odd]

				re[odd] = re[even] - tr
				im[odd] = im[even] - ti
				re[even] += tr
				im[even] += ti

				wRe, wIm = wRe*wStepRe-wIm*wStepIm, wRe*wStepIm+wIm*wStepRe
			}
		}
	}
}
