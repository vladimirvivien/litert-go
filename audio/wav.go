package audio

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DecodeWAV reads a 16 kHz mono WAV (PCM16 or IEEE-float32) into PCM samples for
// Preprocess. It is intentionally minimal: other sample rates, channel counts,
// or bit depths return an error (resample/downmix upstream).
func DecodeWAV(data []byte) ([]float32, error) {
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("audio: not a RIFF/WAVE file")
	}

	var format, channels, bits uint16
	var rate uint32
	var pcm []byte
	haveFmt := false
	for p := 12; p+8 <= len(data); {
		id := string(data[p : p+4])
		size := int(binary.LittleEndian.Uint32(data[p+4 : p+8]))
		body := p + 8
		if body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("audio: short fmt chunk")
			}
			format = binary.LittleEndian.Uint16(data[body : body+2])
			channels = binary.LittleEndian.Uint16(data[body+2 : body+4])
			rate = binary.LittleEndian.Uint32(data[body+4 : body+8])
			bits = binary.LittleEndian.Uint16(data[body+14 : body+16])
			haveFmt = true
		case "data":
			pcm = data[body : body+size]
		}
		p = body + size
		if size%2 == 1 {
			p++ // chunks are word-aligned
		}
	}
	if !haveFmt || pcm == nil {
		return nil, fmt.Errorf("audio: missing fmt or data chunk")
	}
	if channels != 1 {
		return nil, fmt.Errorf("audio: only mono supported, got %d channels", channels)
	}
	if rate != SampleRate {
		return nil, fmt.Errorf("audio: only %d Hz supported, got %d", SampleRate, rate)
	}

	switch {
	case format == 1 && bits == 16:
		out := make([]float32, len(pcm)/2)
		for i := range out {
			out[i] = float32(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / 32768
		}
		return out, nil
	case format == 3 && bits == 32:
		out := make([]float32, len(pcm)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(pcm[i*4:]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("audio: unsupported format=%d bits=%d (want PCM16 or float32)", format, bits)
	}
}
