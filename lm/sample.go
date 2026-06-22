package lm

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

func (s *sampler) sampleAddr(addr unsafe.Pointer, vocab int) int32 {
	if s.greedy() {
		return argmaxF32Addr(addr, vocab)
	}
	logits := unsafe.Slice((*float32)(addr), vocab)
	return s.sampleLogits(logits)
}

func argmaxF32Addr(addr unsafe.Pointer, n int) int32 {
	vals := unsafe.Slice((*float32)(addr), n)
	best, idx := vals[0], 0
	for i, v := range vals {
		if v > best {
			best, idx = v, i
		}
	}
	return int32(idx)
}

type cand struct {
	id    int
	logit float32
}

func heapUp(h []cand, i int) {
	for {
		parent := (i - 1) / 2
		if parent == i || h[i].logit >= h[parent].logit {
			break
		}
		h[i], h[parent] = h[parent], h[i]
		i = parent
	}
}

func heapDown(h []cand, i int) {
	n := len(h)
	for {
		left := 2*i + 1
		if left >= n || left < 0 {
			break
		}
		smallest := left
		if right := left + 1; right < n && h[right].logit < h[left].logit {
			smallest = right
		}
		if h[smallest].logit >= h[i].logit {
			break
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
}

func (s *sampler) topKCandidates(logits []float32, k int, h []cand) []cand {
	if k <= 0 || k > len(logits) {
		k = 200 // reasonable fallback limit to avoid sorting all 256,000 elements
	}
	if cap(h) < k+1 {
		h = make([]cand, 0, k+1)
	} else {
		h = h[:0]
	}
	for i, v := range logits {
		if len(h) < k {
			h = append(h, cand{id: i, logit: v})
			heapUp(h, len(h)-1)
		} else if v > h[0].logit {
			h[0] = cand{id: i, logit: v}
			heapDown(h, 0)
		}
	}
	invTemp := 1.0 / s.temp
	for i := range h {
		h[i].logit *= invTemp
	}
	sort.Slice(h, func(i, j int) bool {
		return h[i].logit > h[j].logit
	})
	return h
}

func (s *sampler) sampleLogits(logits []float32) int32 {
	var candsBuf [256]cand
	var cands []cand
	if s.topK > 0 {
		cands = s.topKCandidates(logits, s.topK, candsBuf[:0])
	} else {
		cands = s.topKCandidates(logits, 200, candsBuf[:0])
	}

	maxL := cands[0].logit

	var probsBuf [256]float64
	var probs []float64
	if len(cands) <= len(probsBuf) {
		probs = probsBuf[:len(cands)]
	} else {
		probs = make([]float64, len(cands))
	}

	var sum float64
	for i, c := range cands {
		p := math.Exp(float64(c.logit - maxL))
		probs[i] = p
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Nucleus: keep the smallest prefix whose cumulative probability reaches topP.
	cut := len(cands)
	if s.topP > 0 && s.topP < 1 {
		var cum float64
		for i := 0; i < len(cands); i++ {
			cum += probs[i]
			if cum >= float64(s.topP) {
				cut = i + 1
				break
			}
		}
	}

	var kept float64
	for i := 0; i < cut; i++ {
		kept += probs[i]
	}
	r := s.rng.Float64() * kept
	var acc float64
	for i := 0; i < cut; i++ {
		acc += probs[i]
		if r < acc {
			return int32(cands[i].id)
		}
	}
	return int32(cands[0].id)
}

