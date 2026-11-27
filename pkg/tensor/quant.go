package tensor

import (
	"fmt"
	"math"
)

// QuantParams holds affine quantization parameters matching LiteRT/TFLite
// real = (quantized - ZeroPoint) * Scale.
type QuantParams struct {
	Scale     float32
	ZeroPoint int64
}

func validateQuantParams(q QuantParams) error {
	s := float64(q.Scale)
	if s <= 0 || math.IsNaN(s) || math.IsInf(s, 0) {
		return fmt.Errorf("tensor: invalid quantization scale %v", q.Scale)
	}
	return nil
}

func quantizeValue(v, invScale float64, zp int64, minVal, maxVal float64) float64 {
	q := math.RoundToEven(v*invScale) + float64(zp)
	if q < minVal {
		return minVal
	}
	if q > maxVal {
		return maxVal
	}
	return q
}

// QuantizeUint8 quantizes src into a newly allocated uint8 slice.
func QuantizeUint8(src []float32, q QuantParams) ([]uint8, error) {
	dst := make([]uint8, len(src))
	if err := QuantizeUint8Into(dst, src, q); err != nil {
		return nil, err
	}
	return dst, nil
}

// QuantizeUint8Into quantizes src into dst using q.
func QuantizeUint8Into(dst []uint8, src []float32, q QuantParams) error {
	if err := validateQuantParams(q); err != nil {
		return err
	}
	if len(dst) != len(src) {
		return fmt.Errorf("tensor: dst length %d != src length %d", len(dst), len(src))
	}
	invScale := 1.0 / float64(q.Scale)
	for i, v := range src {
		dst[i] = uint8(quantizeValue(float64(v), invScale, q.ZeroPoint, 0, 255))
	}
	return nil
}

// DequantizeUint8 dequantizes src into a newly allocated float32 slice.
func DequantizeUint8(src []uint8, q QuantParams) ([]float32, error) {
	dst := make([]float32, len(src))
	if err := DequantizeUint8Into(dst, src, q); err != nil {
		return nil, err
	}
	return dst, nil
}

// DequantizeUint8Into dequantizes src into dst using q.
func DequantizeUint8Into(dst []float32, src []uint8, q QuantParams) error {
	if err := validateQuantParams(q); err != nil {
		return err
	}
	if len(dst) != len(src) {
		return fmt.Errorf("tensor: dst length %d != src length %d", len(dst), len(src))
	}
	scale := float64(q.Scale)
	for i, v := range src {
		dst[i] = float32(float64(int64(v)-q.ZeroPoint) * scale)
	}
	return nil
}

// QuantizeInt8 quantizes src into a newly allocated int8 slice.
func QuantizeInt8(src []float32, q QuantParams) ([]int8, error) {
	dst := make([]int8, len(src))
	if err := QuantizeInt8Into(dst, src, q); err != nil {
		return nil, err
	}
	return dst, nil
}

// QuantizeInt8Into quantizes src into dst using q.
func QuantizeInt8Into(dst []int8, src []float32, q QuantParams) error {
	if err := validateQuantParams(q); err != nil {
		return err
	}
	if len(dst) != len(src) {
		return fmt.Errorf("tensor: dst length %d != src length %d", len(dst), len(src))
	}
	invScale := 1.0 / float64(q.Scale)
	for i, v := range src {
		dst[i] = int8(quantizeValue(float64(v), invScale, q.ZeroPoint, -128, 127))
	}
	return nil
}

// DequantizeInt8 dequantizes src into a newly allocated float32 slice.
func DequantizeInt8(src []int8, q QuantParams) ([]float32, error) {
	dst := make([]float32, len(src))
	if err := DequantizeInt8Into(dst, src, q); err != nil {
		return nil, err
	}
	return dst, nil
}

// DequantizeInt8Into dequantizes src into dst using q.
func DequantizeInt8Into(dst []float32, src []int8, q QuantParams) error {
	if err := validateQuantParams(q); err != nil {
		return err
	}
	if len(dst) != len(src) {
		return fmt.Errorf("tensor: dst length %d != src length %d", len(dst), len(src))
	}
	scale := float64(q.Scale)
	for i, v := range src {
		dst[i] = float32(float64(int64(v)-q.ZeroPoint) * scale)
	}
	return nil
}
