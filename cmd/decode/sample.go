package main

import (
	"math"
	"math/rand"
	"sort"
	"unsafe"

	"github.com/vladimirvivien/litert-go/litert"
)

// sampler turns a logits row into a token id. With temperature <= 0 it is
// greedy (argmax); otherwise it applies temperature, optional top-k and
// top-p (nucleus) filtering, then draws from the resulting distribution.
type sampler struct {
	temp float32
	topK int
	topP float32
	rng  *rand.Rand
}

func newSampler(temp float32, topK int, topP float32, seed int64) *sampler {
	if seed == 0 {
		seed = 1 // a fixed non-zero default keeps runs reproducible
	}
	return &sampler{temp: temp, topK: topK, topP: topP, rng: rand.New(rand.NewSource(seed))}
}

func (s *sampler) greedy() bool { return s.temp <= 0 }

// sample reads a vocab-sized logits row from b and returns a token id.
func (s *sampler) sample(b litert.TensorBuffer, vocab int) (int32, error) {
	if s.greedy() {
		return argmaxF32(b, vocab)
	}
	addr, err := b.Lock(litert.LockRead)
	if err != nil {
		return 0, err
	}
	defer b.Unlock()
	logits := unsafe.Slice((*float32)(addr), vocab)
	return s.sampleLogits(logits), nil
}

func (s *sampler) sampleLogits(logits []float32) int32 {
	type cand struct {
		id    int
		logit float32
	}
	cands := make([]cand, len(logits))
	for i, v := range logits {
		cands[i] = cand{i, v / s.temp}
	}
	if s.topK > 0 && s.topK < len(cands) {
		sort.Slice(cands, func(a, b int) bool { return cands[a].logit > cands[b].logit })
		cands = cands[:s.topK]
	}

	maxL := cands[0].logit
	for _, c := range cands {
		if c.logit > maxL {
			maxL = c.logit
		}
	}
	probs := make([]float64, len(cands))
	var sum float64
	for i, c := range cands {
		p := math.Exp(float64(c.logit - maxL))
		probs[i] = p
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}

	order := make([]int, len(cands))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })

	// Nucleus: keep the smallest prefix whose cumulative probability reaches topP.
	cut := len(order)
	if s.topP > 0 && s.topP < 1 {
		var cum float64
		for k, i := range order {
			cum += probs[i]
			if cum >= float64(s.topP) {
				cut = k + 1
				break
			}
		}
	}

	var kept float64
	for _, i := range order[:cut] {
		kept += probs[i]
	}
	r := s.rng.Float64() * kept
	var acc float64
	for _, i := range order[:cut] {
		acc += probs[i]
		if r < acc {
			return int32(cands[i].id)
		}
	}
	return int32(cands[order[0]].id)
}
