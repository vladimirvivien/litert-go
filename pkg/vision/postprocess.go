package vision

import (
	"fmt"

	"github.com/vladimirvivien/litert-go/pkg/tensor"
)

// Classification pairs a predicted class index, display label, and score.
type Classification struct {
	Index int
	Label string
	Score float32
}

// ClassifyTopK selects the top-k highest-scoring predictions and attaches
// labels from labels when in bounds.
func ClassifyTopK(scores []float32, labels []string, k int) []Classification {
	top := tensor.TopK(scores, k)
	if len(top) == 0 {
		return nil
	}
	out := make([]Classification, len(top))
	for i, item := range top {
		label := fmt.Sprintf("class_%d", item.Index)
		if item.Index >= 0 && item.Index < len(labels) && labels[item.Index] != "" {
			label = labels[item.Index]
		}
		out[i] = Classification{
			Index: item.Index,
			Label: label,
			Score: item.Score,
		}
	}
	return out
}

// DecodeDetections parses TFLite object detection output arrays ([YMin, XMin,
// YMax, XMax], class IDs, scores) filtering predictions below scoreThreshold.
func DecodeDetections(boxes, classes, scores []float32, numDetections int, scoreThreshold float32) ([]tensor.Detection, error) {
	if numDetections < 0 {
		return nil, fmt.Errorf("vision: negative numDetections %d", numDetections)
	}
	if numDetections == 0 {
		return nil, nil
	}
	if len(boxes) < numDetections*4 || len(classes) < numDetections || len(scores) < numDetections {
		return nil, fmt.Errorf("vision: detection outputs too short for %d detections (boxes=%d, classes=%d, scores=%d)",
			numDetections, len(boxes), len(classes), len(scores))
	}
	var out []tensor.Detection
	for i := range numDetections {
		if scores[i] < scoreThreshold {
			continue
		}
		base := i * 4
		out = append(out, tensor.Detection{
			Box: tensor.Box{
				YMin: boxes[base],
				XMin: boxes[base+1],
				YMax: boxes[base+2],
				XMax: boxes[base+3],
			},
			ClassID: int(classes[i]),
			Score:   scores[i],
		})
	}
	return out, nil
}

// UnletterboxBox maps a normalized [0, 1] bounding box from letterboxed input
// coordinates back into normalized [0, 1] source-image coordinates.
func UnletterboxBox(box tensor.Box, info LetterboxInfo) tensor.Box {
	if info.Scale <= 0 || info.SrcWidth <= 0 || info.SrcHeight <= 0 || info.DstWidth <= 0 || info.DstHeight <= 0 {
		return box
	}
	unscaleX := func(normX float32) float32 {
		px := float64(normX)*float64(info.DstWidth) - float64(info.PadX)
		srcNorm := px / (float64(info.Scale) * float64(info.SrcWidth))
		return clampUnit(srcNorm)
	}
	unscaleY := func(normY float32) float32 {
		px := float64(normY)*float64(info.DstHeight) - float64(info.PadY)
		srcNorm := px / (float64(info.Scale) * float64(info.SrcHeight))
		return clampUnit(srcNorm)
	}
	return tensor.Box{
		YMin: unscaleY(box.YMin),
		XMin: unscaleX(box.XMin),
		YMax: unscaleY(box.YMax),
		XMax: unscaleX(box.XMax),
	}
}

func clampUnit(v float64) float32 {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 1
	}
	return float32(v)
}

// ArgmaxMask converts dense per-pixel segmentation logits [H * W * numClasses]
// into a flat [H * W] class-index mask.
func ArgmaxMask(logits []float32, height, width, numClasses int) ([]uint8, error) {
	if height <= 0 || width <= 0 || numClasses <= 0 || numClasses > 256 {
		return nil, fmt.Errorf("vision: invalid segmentation mask dimensions (%dx%d, classes=%d)", height, width, numClasses)
	}
	pixels := height * width
	if len(logits) != pixels*numClasses {
		return nil, fmt.Errorf("vision: logits length %d != %d*%d*%d", len(logits), height, width, numClasses)
	}
	mask := make([]uint8, pixels)
	for i := range pixels {
		off := i * numClasses
		mask[i] = uint8(tensor.Argmax(logits[off : off+numClasses]))
	}
	return mask, nil
}
