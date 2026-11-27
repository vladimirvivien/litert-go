package tensor_test

import (
	"math"
	"slices"
	"testing"

	"github.com/vladimirvivien/litert-go/pkg/tensor"
)

func TestShapeHelpers(t *testing.T) {
	n, err := tensor.NumElements([]int32{1, 224, 224, 3})
	if err != nil || n != 150528 {
		t.Fatalf("NumElements = %d, %v; want 150528", n, err)
	}
	if nScalar, sErr := tensor.NumElements(nil); sErr != nil || nScalar != 1 {
		t.Fatalf("scalar NumElements = %d, %v; want 1", nScalar, sErr)
	}
	if nZero, zErr := tensor.NumElements([]int32{2, 0, 5}); zErr != nil || nZero != 0 {
		t.Fatalf("zero-dim NumElements = %d, %v; want 0", nZero, zErr)
	}
	if _, negErr := tensor.NumElements([]int32{1, -1}); negErr == nil {
		t.Fatal("expected error for negative dim")
	}

	strides, err := tensor.Strides([]int32{2, 3, 4})
	if err != nil || !slices.Equal(strides, []int{12, 4, 1}) {
		t.Fatalf("Strides = %v, %v; want [12 4 1]", strides, err)
	}
	if vErr := tensor.ValidateShape(24, []int32{2, 3, 4}); vErr != nil {
		t.Fatalf("ValidateShape failed: %v", vErr)
	}
	if vErr := tensor.ValidateShape(23, []int32{2, 3, 4}); vErr == nil {
		t.Fatal("expected ValidateShape mismatch error")
	}
}

func TestQuantizeDequantize(t *testing.T) {
	qu := tensor.QuantParams{Scale: 0.5, ZeroPoint: 128}
	src := []float32{-100, -1.0, 0.0, 1.25, 100}
	u8, err := tensor.QuantizeUint8(src, qu)
	if err != nil {
		t.Fatal(err)
	}
	wantU8 := []uint8{0, 126, 128, 130, 255}
	if !slices.Equal(u8, wantU8) {
		t.Fatalf("QuantizeUint8 = %v, want %v", u8, wantU8)
	}
	deqU8, err := tensor.DequantizeUint8(u8, qu)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(deqU8[2]-0)) > 1e-6 || math.Abs(float64(deqU8[1]-(-1.0))) > 1e-6 {
		t.Fatalf("DequantizeUint8 = %v", deqU8)
	}

	qi := tensor.QuantParams{Scale: 0.25, ZeroPoint: -10}
	i8, err := tensor.QuantizeInt8([]float32{-100, -1.0, 0, 100}, qi)
	if err != nil {
		t.Fatal(err)
	}
	wantI8 := []int8{-128, -14, -10, 127}
	if !slices.Equal(i8, wantI8) {
		t.Fatalf("QuantizeInt8 = %v, want %v", i8, wantI8)
	}
	deqI8, err := tensor.DequantizeInt8(i8, qi)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(deqI8[2])) > 1e-6 {
		t.Fatalf("DequantizeInt8[2] = %v, want 0", deqI8[2])
	}

	if _, badErr := tensor.QuantizeUint8(src, tensor.QuantParams{Scale: 0}); badErr == nil {
		t.Fatal("expected error for zero scale")
	}
	if intoErr := tensor.DequantizeInt8Into(make([]float32, 1), i8, qi); intoErr == nil {
		t.Fatal("expected length mismatch error")
	}
}

func TestOpsAndNMS(t *testing.T) {
	// Softmax stability on large logits.
	probs := tensor.Softmax([]float32{1000, 1001, 999})
	var sum float64
	for _, p := range probs {
		sum += float64(p)
	}
	if math.Abs(sum-1.0) > 1e-5 || tensor.Argmax(probs) != 1 {
		t.Fatalf("Softmax/Argmax unexpected: probs=%v sum=%v", probs, sum)
	}

	// Tie-breaking for Argmax and TopK.
	scores := []float32{0.5, 0.9, 0.9, 0.2}
	if got := tensor.Argmax(scores); got != 1 {
		t.Fatalf("Argmax tie-break = %d, want 1", got)
	}
	top2 := tensor.TopK(scores, 2)
	wantTop2 := []tensor.ScoredIndex{{Index: 1, Score: 0.9}, {Index: 2, Score: 0.9}}
	if !slices.Equal(top2, wantTop2) {
		t.Fatalf("TopK = %+v, want %+v", top2, wantTop2)
	}
	if tensor.Argmax(nil) != -1 || tensor.TopK(scores, 0) != nil {
		t.Fatal("empty edge cases failed")
	}

	// L2Normalize & CosineSimilarity.
	normed := tensor.L2Normalize([]float32{3, 4})
	if math.Abs(float64(normed[0]-0.6)) > 1e-6 || math.Abs(float64(normed[1]-0.8)) > 1e-6 {
		t.Fatalf("L2Normalize = %v, want [0.6 0.8]", normed)
	}
	zero := tensor.L2Normalize([]float32{0, 0})
	if !slices.Equal(zero, []float32{0, 0}) {
		t.Fatalf("zero norm vector mutated: %v", zero)
	}
	cos, err := tensor.CosineSimilarity([]float32{1, 0}, []float32{0, 1})
	if err != nil || cos != 0 {
		t.Fatalf("CosineSimilarity orthogonal = %v, %v", cos, err)
	}
	cosSame, err := tensor.CosineSimilarity([]float32{2, 0}, []float32{5, 0})
	if err != nil || math.Abs(float64(cosSame-1.0)) > 1e-6 {
		t.Fatalf("CosineSimilarity parallel = %v, %v", cosSame, err)
	}
	if _, misErr := tensor.CosineSimilarity([]float32{1}, []float32{1, 2}); misErr == nil {
		t.Fatal("expected error for mismatched vector lengths")
	}

	// IoU & NMS (per-class vs class-agnostic).
	b1 := tensor.Box{YMin: 0, XMin: 0, YMax: 10, XMax: 10}
	b2 := tensor.Box{YMin: 1, XMin: 1, YMax: 10, XMax: 10}
	if iou := tensor.IoU(b1, b2); iou <= 0.7 || iou >= 0.9 {
		t.Fatalf("unexpected IoU %v", iou)
	}
	dets := []tensor.Detection{
		{Box: b1, ClassID: 0, Score: 0.9},
		{Box: b2, ClassID: 0, Score: 0.7}, // suppressed by b1
		{Box: b2, ClassID: 1, Score: 0.8}, // different class -> kept by per-class NMS
	}
	perClass := tensor.NMS(dets, 0.5)
	if len(perClass) != 2 || perClass[0].Score != 0.9 || perClass[1].ClassID != 1 {
		t.Fatalf("per-class NMS = %+v", perClass)
	}
	agnostic := tensor.NMSClassAgnostic(dets, 0.5)
	if len(agnostic) != 1 || agnostic[0].Score != 0.9 {
		t.Fatalf("class-agnostic NMS = %+v", agnostic)
	}
}
