// Command decode generates text from a .litertlm (or raw .tflite) model on the
// LiteRT runtime, using the lm package. It supports a single completion (-text),
// raw prompt token IDs (-prompt), or interactive multi-turn chat (-repl), with
// chat templating, sampling, and MTP speculative decoding.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/vladimirvivien/litert-go/audio"
	"github.com/vladimirvivien/litert-go/litert"
	"github.com/vladimirvivien/litert-go/lm"
)

func main() {
	libDir := flag.String("lib", "", "directory or path of libLiteRt (or set LITERT_LIB)")
	modelPath := flag.String("model", "", "path to a .litertlm container or raw .tflite")
	text := flag.String("text", "", "text prompt (uses the model's embedded SentencePiece tokenizer)")
	promptCSV := flag.String("prompt", "", "prompt token IDs, comma-separated (alternative to -text)")
	ngen := flag.Int("n", 16, "max number of tokens to generate")
	chat := flag.Bool("chat", false, "wrap -text in the model's chat template (from container metadata)")
	backend := flag.String("backend", "cpu", "compile backend: cpu | gpu")
	spec := flag.Bool("spec", false, "MTP speculative decoding (needs a verify signature + mtp_drafter section)")
	temp := flag.Float64("temp", 0, "sampling temperature (0 = greedy)")
	topK := flag.Int("topk", 0, "top-k sampling (0 = off)")
	topP := flag.Float64("topp", 0, "top-p / nucleus sampling (0 = off)")
	seed := flag.Int64("seed", 0, "sampling RNG seed")
	repl := flag.Bool("repl", false, "interactive multi-turn chat: read user turns from stdin")
	system := flag.String("system", "", "system prompt (chat/-repl only)")
	image := flag.String("image", "", "image file; -text must contain <start_of_image> (gemma-4 vision)")
	audioFile := flag.String("audio", "", "16kHz mono WAV; -text must contain <start_of_audio> (gemma-4 audio)")
	flag.Parse()

	if *modelPath == "" || (!*repl && *text == "" && *promptCSV == "") {
		fmt.Fprintln(os.Stderr, "decode: -model and one of -text/-prompt (or -repl) are required")
		flag.Usage()
		os.Exit(2)
	}
	accel, err := parseBackend(*backend)
	if err != nil {
		fail(err)
	}

	eng, err := lm.Open(*libDir, *modelPath, accel)
	if err != nil {
		fail(err)
	}
	defer eng.Close()

	o := lm.GenOptions{
		MaxTokens: *ngen,
		Temp:      float32(*temp),
		TopK:      *topK,
		TopP:      float32(*topP),
		Seed:      *seed,
		Spec:      *spec,
		System:    *system,
	}

	switch {
	case *repl:
		err = runRepl(eng, o)
	case *image != "":
		var data []byte
		if data, err = os.ReadFile(*image); err == nil {
			var out string
			if out, err = eng.GenerateFromImage(*text, data, 0, o); err == nil {
				fmt.Printf("prompt: %q\noutput: %s\n", *text, out)
			}
		}
	case *audioFile != "":
		var data []byte
		if data, err = os.ReadFile(*audioFile); err == nil {
			var pcm []float32
			if pcm, err = audio.DecodeWAV(data); err == nil {
				var out string
				if out, err = eng.GenerateFromAudio(*text, pcm, o); err == nil {
					fmt.Printf("prompt: %q\noutput: %s\n", *text, out)
				}
			}
		}
	case *promptCSV != "":
		var ids []int32
		if ids, err = parseIDs(*promptCSV); err == nil {
			var gen []int
			if gen, err = eng.GenerateIDs(ids, o); err == nil {
				fmt.Printf("prompt=%v\noutput tokens=%v\n", ids, gen)
			}
		}
	default:
		fmt.Printf("prompt: %q\noutput: ", *text)
		_, err = eng.GenerateStream(*text, *chat, o, func(p string) { fmt.Print(p) })
		fmt.Println()
	}
	if err != nil {
		fail(err)
	}
}

// runRepl is an interactive multi-turn chat loop: each stdin line is a user
// turn; the reply is printed and kept for context.
func runRepl(eng *lm.Engine, o lm.GenOptions) error {
	chat, err := eng.NewConversation(o)
	if err != nil {
		return err
	}
	defer chat.Close()
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	fmt.Fprint(os.Stderr, "> ")
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			fmt.Fprint(os.Stderr, "> ")
			continue
		}
		if _, err := chat.SendStream(line, func(p string) { fmt.Print(p) }); err != nil {
			return err
		}
		fmt.Println()
		fmt.Fprint(os.Stderr, "> ")
	}
	return sc.Err()
}

func parseBackend(s string) (litert.HwAccelerator, error) {
	switch s {
	case "cpu":
		return litert.AccelCPU, nil
	case "gpu":
		return litert.AccelGPU, nil
	default:
		return 0, fmt.Errorf("unknown backend %q (cpu|gpu)", s)
	}
}

func parseIDs(csv string) ([]int32, error) {
	parts := strings.Split(csv, ",")
	ids := make([]int32, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, fmt.Errorf("bad token id %q: %w", p, err)
		}
		ids = append(ids, int32(n))
	}
	if len(ids) < 2 {
		return nil, fmt.Errorf("need at least 2 prompt tokens")
	}
	return ids, nil
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "decode:", err)
	os.Exit(1)
}
