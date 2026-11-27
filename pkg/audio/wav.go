// Package audio provides standard-library WAV decoding, PCM resampling,
// framing, STFT, and configurable log-mel spectrogram extraction for audio
// .tflite models.
package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"
)

const (
	wavFormatPCM       = 1
	wavFormatIEEEFloat = 3
	wavFormatExt       = 0xFFFE
)

// PCM holds decoded normalized audio samples in [-1.0, 1.0].
type PCM struct {
	Samples    []float32
	SampleRate int
	Channels   int
}

// DecodeWAV reads and decodes a RIFF/WAVE stream into normalized PCM samples.
func DecodeWAV(r io.Reader) (PCM, error) {
	if r == nil {
		return PCM{}, fmt.Errorf("audio: nil reader")
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return PCM{}, fmt.Errorf("audio: read WAV: %w", err)
	}
	return DecodeWAVBytes(data)
}

// DecodeWAVBytes decodes a RIFF/WAVE byte buffer supporting 8/16/24/32-bit
// integer PCM and 32-bit IEEE float waveforms.
func DecodeWAVBytes(data []byte) (PCM, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return PCM{}, fmt.Errorf("audio: invalid RIFF/WAVE header")
	}

	var (
		hasFmt        bool
		hasData       bool
		audioFormat   uint16
		channels      int
		sampleRate    int
		bitsPerSample int
		pcmBytes      []byte
	)

	off := 12
	for off+8 <= len(data) {
		chunkID := string(data[off : off+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		payloadStart := off + 8
		if chunkSize < 0 || payloadStart+chunkSize > len(data) {
			return PCM{}, fmt.Errorf("audio: truncated WAV chunk %q", chunkID)
		}
		payload := data[payloadStart : payloadStart+chunkSize]

		switch chunkID {
		case "fmt ":
			if len(payload) < 16 {
				return PCM{}, fmt.Errorf("audio: fmt chunk too short (%d bytes)", len(payload))
			}
			audioFormat = binary.LittleEndian.Uint16(payload[0:2])
			channels = int(binary.LittleEndian.Uint16(payload[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(payload[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(payload[14:16]))
			if audioFormat == wavFormatExt && len(payload) >= 40 {
				audioFormat = binary.LittleEndian.Uint16(payload[24:26])
			}
			if channels <= 0 || sampleRate <= 0 {
				return PCM{}, fmt.Errorf("audio: invalid WAV format (channels=%d, sampleRate=%d)", channels, sampleRate)
			}
			hasFmt = true

		case "data":
			pcmBytes = payload
			hasData = true
		}

		next := payloadStart + chunkSize
		if chunkSize%2 == 1 && next < len(data) {
			next++
		}
		off = next
	}

	if !hasFmt || !hasData {
		return PCM{}, fmt.Errorf("audio: missing fmt (%v) or data (%v) chunk", hasFmt, hasData)
	}

	samples, err := decodePCMSamples(pcmBytes, audioFormat, bitsPerSample)
	if err != nil {
		return PCM{}, err
	}
	return PCM{
		Samples:    samples,
		SampleRate: sampleRate,
		Channels:   channels,
	}, nil
}

func decodePCMSamples(data []byte, format uint16, bits int) ([]float32, error) {
	switch format {
	case wavFormatPCM:
		switch bits {
		case 8:
			out := make([]float32, len(data))
			for i, b := range data {
				out[i] = (float32(b) - 128.0) / 128.0
			}
			return out, nil
		case 16:
			n := len(data) / 2
			out := make([]float32, n)
			for i := range n {
				v := int16(binary.LittleEndian.Uint16(data[i*2:]))
				out[i] = float32(v) / 32768.0
			}
			return out, nil
		case 24:
			n := len(data) / 3
			out := make([]float32, n)
			for i := range n {
				off := i * 3
				v := int32(data[off]) | int32(data[off+1])<<8 | int32(data[off+2])<<16
				if v&0x800000 != 0 {
					v |= ^0xffffff
				}
				out[i] = float32(v) / 8388608.0
			}
			return out, nil
		case 32:
			n := len(data) / 4
			out := make([]float32, n)
			for i := range n {
				v := int32(binary.LittleEndian.Uint32(data[i*4:]))
				out[i] = float32(float64(v) / 2147483648.0)
			}
			return out, nil
		}
	case wavFormatIEEEFloat:
		if bits == 32 {
			n := len(data) / 4
			out := make([]float32, n)
			for i := range n {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
			}
			return out, nil
		}
	}
	return nil, fmt.Errorf("audio: unsupported WAV format=%d bits=%d", format, bits)
}

// ToMono mixes interleaved multi-channel PCM frames into a single mono slice.
func ToMono(samples []float32, channels int) []float32 {
	if len(samples) == 0 {
		return nil
	}
	if channels <= 1 {
		return slices.Clone(samples)
	}
	frames := len(samples) / channels
	out := make([]float32, frames)
	inv := 1.0 / float64(channels)
	for i := range frames {
		var sum float64
		base := i * channels
		for c := range channels {
			sum += float64(samples[base+c])
		}
		out[i] = float32(sum * inv)
	}
	return out
}

// ResampleLinear linearly resamples mono samples from srcRate Hz to dstRate Hz.
func ResampleLinear(samples []float32, srcRate, dstRate int) []float32 {
	if len(samples) == 0 || srcRate <= 0 || dstRate <= 0 {
		return nil
	}
	if srcRate == dstRate {
		return slices.Clone(samples)
	}
	outLen := max(1, int(math.Round(float64(len(samples))*float64(dstRate)/float64(srcRate))))
	out := make([]float32, outLen)
	ratio := float64(srcRate) / float64(dstRate)
	last := len(samples) - 1
	for i := range outLen {
		pos := float64(i) * ratio
		idx := int(pos)
		if idx >= last {
			out[i] = samples[last]
			continue
		}
		frac := pos - float64(idx)
		out[i] = float32(float64(samples[idx])*(1.0-frac) + float64(samples[idx+1])*frac)
	}
	return out
}

// PreparePCM converts p to mono and linearly resamples to targetSampleRate Hz.
// Passing targetSampleRate <= 0 skips resampling.
func PreparePCM(p PCM, targetSampleRate int) []float32 {
	mono := ToMono(p.Samples, p.Channels)
	if targetSampleRate <= 0 {
		return mono
	}
	return ResampleLinear(mono, p.SampleRate, targetSampleRate)
}
