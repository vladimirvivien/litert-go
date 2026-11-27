// Package tensor provides standard-library helpers for shape and stride
// calculation, affine quantization/dequantization, and common postprocessing
// operations (Softmax, TopK, L2 normalization, cosine similarity, IoU, and NMS).
package tensor

import (
	"fmt"
	"math"
)

// NumElements returns the total scalar element count for a non-negative shape.
// An empty shape (scalar tensor) has 1 element.
func NumElements(shape []int32) (int, error) {
	n := 1
	for i, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("tensor: negative dimension %d at axis %d", d, i)
		}
		if d == 0 {
			return 0, nil
		}
		if n > math.MaxInt/int(d) {
			return 0, fmt.Errorf("tensor: element count overflow on axis %d", i)
		}
		n *= int(d)
	}
	return n, nil
}

// Strides computes row-major element strides for a non-negative shape.
func Strides(shape []int32) ([]int, error) {
	if _, err := NumElements(shape); err != nil {
		return nil, err
	}
	if len(shape) == 0 {
		return []int{}, nil
	}
	strides := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = acc
		d := int(shape[i])
		if d > 0 {
			if acc > math.MaxInt/d {
				return nil, fmt.Errorf("tensor: stride overflow on axis %d", i)
			}
			acc *= d
		}
	}
	return strides, nil
}

// ValidateShape verifies that actualLen matches the total element count of shape.
func ValidateShape(actualLen int, shape []int32) error {
	if actualLen < 0 {
		return fmt.Errorf("tensor: negative length %d", actualLen)
	}
	want, err := NumElements(shape)
	if err != nil {
		return err
	}
	if actualLen != want {
		return fmt.Errorf("tensor: length %d does not match shape %v (%d elements)", actualLen, shape, want)
	}
	return nil
}
