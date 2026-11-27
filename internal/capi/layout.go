package capi

import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

// ABI constants matching LiteRT 2.4.1 headers across Linux, macOS, and Windows.
const (
	// MaxRank is LITERT_TENSOR_MAX_RANK.
	MaxRank = 8
	// LayoutSize is sizeof(LiteRtLayout) in bytes.
	LayoutSize = 68
	// RankedTensorTypeSize is sizeof(LiteRtRankedTensorType) in bytes.
	RankedTensorTypeSize = 72
	// EnvOptionSize is sizeof(LiteRtEnvOption) in bytes.
	EnvOptionSize = 24
	// QuantizationPerTensorSize is sizeof(LiteRtQuantizationPerTensor) in bytes.
	QuantizationPerTensorSize = 16
	// QuantizationPerChannelSize is sizeof(LiteRtQuantizationPerChannel) in bytes.
	QuantizationPerChannelSize = 32
)

// LiteRtAnyType discriminators from litert_any.h.
const (
	AnyTypeNone    int32 = 0
	AnyTypeBool    int32 = 1
	AnyTypeInt     int32 = 2
	AnyTypeReal    int32 = 3
	AnyTypeString  int32 = 8
	AnyTypeVoidPtr int32 = 9
)

// Layout mirrors LiteRtLayout (68 bytes).
type Layout struct {
	Rank       int
	HasStrides bool
	Dimensions []int32
	Strides    []uint32
}

// DecodeLayout decodes a 68-byte LiteRtLayout buffer.
func DecodeLayout(raw []byte) (Layout, error) {
	if len(raw) < LayoutSize {
		return Layout{}, fmt.Errorf("capi: layout buffer too short (%d < %d)", len(raw), LayoutSize)
	}
	header := binary.LittleEndian.Uint32(raw[0:4])
	rank := int(header & 0x7f)
	if rank > MaxRank {
		return Layout{}, fmt.Errorf("capi: tensor rank %d exceeds max %d", rank, MaxRank)
	}
	hasStrides := (header & 0x80) != 0

	dims := make([]int32, rank)
	for i := range rank {
		off := 4 + 4*i
		dims[i] = int32(binary.LittleEndian.Uint32(raw[off : off+4]))
	}
	var strides []uint32
	if hasStrides {
		strides = make([]uint32, rank)
		for i := range rank {
			off := 36 + 4*i
			strides[i] = binary.LittleEndian.Uint32(raw[off : off+4])
		}
	}
	return Layout{
		Rank:       rank,
		HasStrides: hasStrides,
		Dimensions: dims,
		Strides:    strides,
	}, nil
}

// EncodeLayout encodes a Layout into a 68-byte LiteRtLayout buffer.
func EncodeLayout(l Layout) ([]byte, error) {
	if len(l.Dimensions) > MaxRank {
		return nil, fmt.Errorf("capi: tensor rank %d exceeds max %d", len(l.Dimensions), MaxRank)
	}
	if l.HasStrides && len(l.Strides) != len(l.Dimensions) {
		return nil, fmt.Errorf("capi: strides length %d does not match rank %d", len(l.Strides), len(l.Dimensions))
	}
	raw := make([]byte, LayoutSize)
	header := uint32(len(l.Dimensions))
	if l.HasStrides {
		header |= 0x80
	}
	binary.LittleEndian.PutUint32(raw[0:4], header)
	for i, d := range l.Dimensions {
		binary.LittleEndian.PutUint32(raw[4+4*i:8+4*i], uint32(d))
	}
	if l.HasStrides {
		for i, s := range l.Strides {
			binary.LittleEndian.PutUint32(raw[36+4*i:40+4*i], s)
		}
	}
	return raw, nil
}

// DecodeRankedTensorType decodes a 72-byte LiteRtRankedTensorType buffer.
func DecodeRankedTensorType(raw []byte) (elementType int32, layout Layout, err error) {
	if len(raw) < RankedTensorTypeSize {
		return 0, Layout{}, fmt.Errorf("capi: ranked tensor type buffer too short (%d < %d)", len(raw), RankedTensorTypeSize)
	}
	elementType = int32(binary.LittleEndian.Uint32(raw[0:4]))
	layout, err = DecodeLayout(raw[4:RankedTensorTypeSize])
	if err != nil {
		return 0, Layout{}, err
	}
	return elementType, layout, nil
}

// EncodeRankedTensorType encodes an element type and dimensions into a 72-byte
// LiteRtRankedTensorType buffer.
func EncodeRankedTensorType(elementType int32, dims []int32) ([]byte, error) {
	layoutRaw, err := EncodeLayout(Layout{
		Rank:       len(dims),
		Dimensions: dims,
	})
	if err != nil {
		return nil, err
	}
	raw := make([]byte, RankedTensorTypeSize)
	binary.LittleEndian.PutUint32(raw[0:4], uint32(elementType))
	copy(raw[4:RankedTensorTypeSize], layoutRaw)
	return raw, nil
}

// EnvOptionValue carries a single serialized LiteRtEnvOption entry. String
// pointers must already point to pinned NUL-terminated bytes retained by the caller.
type EnvOptionValue struct {
	Tag     int32
	AnyType int32
	IntVal  int64
	PtrVal  unsafe.Pointer
}

// EncodeEnvOptions writes options into a contiguous C LiteRtEnvOption array.
func EncodeEnvOptions(opts []EnvOptionValue) ([]byte, error) {
	if len(opts) == 0 {
		return nil, nil
	}
	buf := make([]byte, len(opts)*EnvOptionSize)
	for i, o := range opts {
		base := i * EnvOptionSize
		binary.LittleEndian.PutUint32(buf[base:base+4], uint32(o.Tag))
		binary.LittleEndian.PutUint32(buf[base+8:base+12], uint32(o.AnyType))
		switch o.AnyType {
		case AnyTypeInt, AnyTypeBool:
			binary.LittleEndian.PutUint64(buf[base+16:base+24], uint64(o.IntVal))
		case AnyTypeString, AnyTypeVoidPtr:
			binary.LittleEndian.PutUint64(buf[base+16:base+24], uint64(uintptr(o.PtrVal)))
		default:
			return nil, fmt.Errorf("capi: unsupported env option AnyType %d", o.AnyType)
		}
	}
	return buf, nil
}

// DecodePerTensorQuantization decodes LiteRtQuantizationPerTensor (16 bytes).
func DecodePerTensorQuantization(raw []byte) (scale float32, zeroPoint int64, err error) {
	if len(raw) < QuantizationPerTensorSize {
		return 0, 0, fmt.Errorf("capi: per-tensor quantization buffer too short (%d < %d)", len(raw), QuantizationPerTensorSize)
	}
	scaleBits := binary.LittleEndian.Uint32(raw[0:4])
	scale = *(*float32)(unsafe.Pointer(&scaleBits))
	zeroPoint = int64(binary.LittleEndian.Uint64(raw[8:16]))
	return scale, zeroPoint, nil
}

// RawPerChannelQuantization matches the C ABI layout of
// LiteRtQuantizationPerChannel (32 bytes on 64-bit targets).
type RawPerChannelQuantization struct {
	QuantizedDimension int32
	_                  int32
	NumChannels        uint64
	Scales             unsafe.Pointer
	ZeroPoints         unsafe.Pointer
}

// CopyPerChannelQuantization copies C-owned per-channel quantization arrays
// into Go-owned slices.
func CopyPerChannelQuantization(raw RawPerChannelQuantization) (dim int32, scales []float32, zeroPoints []int64) {
	n := int(raw.NumChannels)
	if n <= 0 {
		return raw.QuantizedDimension, nil, nil
	}
	if raw.Scales != nil {
		scales = make([]float32, n)
		copy(scales, unsafe.Slice((*float32)(raw.Scales), n))
	}
	if raw.ZeroPoints != nil {
		zeroPoints = make([]int64, n)
		copy(zeroPoints, unsafe.Slice((*int64)(raw.ZeroPoints), n))
	}
	return raw.QuantizedDimension, scales, zeroPoints
}
