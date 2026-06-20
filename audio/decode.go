package audio

import (
	"bytes"
	"fmt"
	"io"
	"strings"

	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/flac"
	"github.com/gopxl/beep/v2/mp3"
	"github.com/gopxl/beep/v2/vorbis"
	"github.com/gopxl/beep/v2/wav"
)

type readSeekCloser struct {
	io.ReadSeeker
}

func (rsc readSeekCloser) Close() error {
	return nil
}

// DecodeAudio decodes raw audio bytes of various formats (.wav, .mp3, .flac, .ogg),
// resamples the audio to 16 kHz if necessary, downmixes to mono, and returns the
// raw float32 PCM samples ready for Preprocess.
func DecodeAudio(data []byte, formatHint string) ([]float32, error) {
	var streamer beep.StreamSeekCloser
	var format beep.Format
	var err error

	r := bytes.NewReader(data)
	rsc := readSeekCloser{r}
	hint := strings.ToLower(formatHint)

	switch {
	case hint == "wav" || strings.HasSuffix(hint, "wav") || strings.HasSuffix(hint, "wave") || strings.Contains(hint, "audio/wav"):
		streamer, format, err = wav.Decode(r)
	case hint == "mp3" || strings.HasSuffix(hint, "mp3") || strings.HasSuffix(hint, "mpeg") || strings.Contains(hint, "audio/mpeg") || strings.Contains(hint, "audio/mp3"):
		streamer, format, err = mp3.Decode(rsc)
	case hint == "flac" || strings.HasSuffix(hint, "flac") || strings.Contains(hint, "audio/flac"):
		streamer, format, err = flac.Decode(rsc)
	case hint == "ogg" || strings.HasSuffix(hint, "ogg") || strings.HasSuffix(hint, "vorbis") || strings.Contains(hint, "audio/ogg") || strings.Contains(hint, "audio/vorbis"):
		streamer, format, err = vorbis.Decode(rsc)
	default:
		// Attempt format auto-detection by trying to decode in sequence
		streamer, format, err = wav.Decode(r)
		if err != nil {
			r.Seek(0, io.SeekStart)
			streamer, format, err = mp3.Decode(rsc)
			if err != nil {
				r.Seek(0, io.SeekStart)
				streamer, format, err = flac.Decode(rsc)
				if err != nil {
					r.Seek(0, io.SeekStart)
					streamer, format, err = vorbis.Decode(rsc)
					if err != nil {
						return nil, fmt.Errorf("audio: unsupported or unrecognized format (hint: %q)", formatHint)
					}
				}
			}
		}
	}
	if err != nil {
		return nil, err
	}
	defer streamer.Close()

	// Target: 16 kHz sample rate
	targetRate := beep.SampleRate(SampleRate)
	var finalStreamer beep.Streamer = streamer

	if format.SampleRate != targetRate {
		// Quality 3 (Cubic) provides a high-quality balance of performance & preservation
		finalStreamer = beep.Resample(3, format.SampleRate, targetRate, streamer)
	}

	// Read and downmix to mono float32
	var pcm []float32
	buf := make([][2]float64, 512)
	for {
		n, ok := finalStreamer.Stream(buf)
		if n == 0 || !ok {
			break
		}
		for i := 0; i < n; i++ {
			// Downmix stereo to mono by averaging channels
			val := float32((buf[i][0] + buf[i][1]) / 2.0)
			pcm = append(pcm, val)
		}
	}

	if err := finalStreamer.Err(); err != nil {
		return nil, fmt.Errorf("audio: decode/resample error: %w", err)
	}

	return pcm, nil
}
