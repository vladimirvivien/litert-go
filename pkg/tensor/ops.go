package tensor

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

// ScoredIndex pairs a tensor element index with its score.
type ScoredIndex struct {
	Index int
	Score float32
}

// Softmax returns a numerically stable softmax probability distribution over logits.
func Softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}
	out := slices.Clone(logits)
	SoftmaxInPlace(out)
	return out
}

// SoftmaxInPlace applies numerically stable softmax directly to v.
func SoftmaxInPlace(v []float32) {
	if len(v) == 0 {
		return
	}
	maxVal := v[0]
	for _, x := range v[1:] {
		if x > maxVal {
			maxVal = x
		}
	}
	var sum float64
	for i, x := range v {
		e := math.Exp(float64(x - maxVal))
		v[i] = float32(e)
		sum += e
	}
	if sum == 0 {
		inv := float32(1.0 / float64(len(v)))
		for i := range v {
			v[i] = inv
		}
		return
	}
	inv := 1.0 / sum
	for i := range v {
		v[i] = float32(float64(v[i]) * inv)
	}
}

// Argmax returns the index of the largest element in v, or -1 if v is empty.
// Ties select the lowest index.
func Argmax(v []float32) int {
	if len(v) == 0 {
		return -1
	}
	bestIdx := 0
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			bestIdx = i
		}
	}
	return bestIdx
}

// TopK returns up to k highest-scoring elements ordered by descending score,
// breaking ties by lowest index.
func TopK(v []float32, k int) []ScoredIndex {
	if k <= 0 || len(v) == 0 {
		return nil
	}
	if k > len(v) {
		k = len(v)
	}
	items := make([]ScoredIndex, len(v))
	for i, s := range v {
		items[i] = ScoredIndex{Index: i, Score: s}
	}
	slices.SortFunc(items, func(a, b ScoredIndex) int {
		if a.Score != b.Score {
			return cmp.Compare(b.Score, a.Score)
		}
		return cmp.Compare(a.Index, b.Index)
	})
	return items[:k]
}

// L2Normalize returns a unit-L2-normalized copy of v (or zeroes if v has zero norm).
func L2Normalize(v []float32) []float32 {
	if len(v) == 0 {
		return nil
	}
	out := slices.Clone(v)
	L2NormalizeInPlace(out)
	return out
}

// L2NormalizeInPlace divides v by its L2 norm in place. Zero-norm vectors are unchanged.
func L2NormalizeInPlace(v []float32) {
	if len(v) == 0 {
		return
	}
	var sumSq float64
	for _, x := range v {
		fx := float64(x)
		sumSq += fx * fx
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 || math.IsNaN(norm) {
		return
	}
	inv := 1.0 / norm
	for i, x := range v {
		v[i] = float32(float64(x) * inv)
	}
}

// CosineSimilarity computes the cosine similarity between equal-length vectors a and b.
func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("tensor: mismatched vector lengths %d and %d", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, fmt.Errorf("tensor: empty vectors")
	}
	var dot, normA, normB float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0, nil
	}
	return float32(dot / denom), nil
}

// Box stores normalized or pixel bounding-box coordinates [YMin, XMin, YMax, XMax].
type Box struct {
	YMin float32
	XMin float32
	YMax float32
	XMax float32
}

// Detection pairs a bounding box with its predicted class ID and confidence score.
type Detection struct {
	Box     Box
	ClassID int
	Score   float32
}

// IoU computes intersection-over-union for two boxes.
func IoU(a, b Box) float32 {
	yMin := max(a.YMin, b.YMin)
	xMin := max(a.XMin, b.XMin)
	yMax := min(a.YMax, b.YMax)
	xMax := min(a.XMax, b.XMax)

	if yMax <= yMin || xMax <= xMin {
		return 0
	}
	inter := float64(yMax-yMin) * float64(xMax-xMin)
	areaA := boxArea(a)
	areaB := boxArea(b)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return float32(inter / union)
}

func boxArea(b Box) float64 {
	h := float64(b.YMax) - float64(b.YMin)
	w := float64(b.XMax) - float64(b.XMin)
	if h <= 0 || w <= 0 {
		return 0
	}
	return h * w
}

// NMS performs per-class non-maximum suppression and returns detections
// ordered by descending confidence score.
func NMS(dets []Detection, iouThreshold float32) []Detection {
	return nms(dets, iouThreshold, true)
}

// NMSClassAgnostic performs class-agnostic non-maximum suppression and returns
// detections ordered by descending confidence score.
func NMSClassAgnostic(dets []Detection, iouThreshold float32) []Detection {
	return nms(dets, iouThreshold, false)
}

func nms(dets []Detection, iouThreshold float32, perClass bool) []Detection {
	if len(dets) == 0 {
		return nil
	}
	sorted := slices.Clone(dets)
	slices.SortFunc(sorted, func(a, b Detection) int {
		if a.Score != b.Score {
			return cmp.Compare(b.Score, a.Score)
		}
		return cmp.Compare(a.ClassID, b.ClassID)
	})

	kept := make([]Detection, 0, len(sorted))
	for _, cand := range sorted {
		suppressed := false
		for _, prev := range kept {
			if perClass && cand.ClassID != prev.ClassID {
				continue
			}
			if IoU(cand.Box, prev.Box) > iouThreshold {
				suppressed = true
				break
			}
		}
		if !suppressed {
			kept = append(kept, cand)
		}
	}
	return kept
}
