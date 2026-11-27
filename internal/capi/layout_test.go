package capi

import (
	"encoding/binary"
	"math"
	"runtime"
	"testing"
	"unsafe"
)

func TestLayoutCodec(t *testing.T) {
	raw, err := EncodeRankedTensorType(1, []int32{1, 224, 224, 3})
	if err != nil {
		t.Fatalf("EncodeRankedTensorType: %v", err)
	}
	if len(raw) != RankedTensorTypeSize {
		t.Fatalf("len(raw) = %d, want %d", len(raw), RankedTensorTypeSize)
	}
	// Verify fixed 2.4.1 offsets: elem@0, header@4, dims@8.
	if got := int32(binary.LittleEndian.Uint32(raw[0:4])); got != 1 {
		t.Fatalf("element_type = %d, want 1", got)
	}
	if got := int32(binary.LittleEndian.Uint32(raw[8:12])); got != 1 {
		t.Fatalf("dim[0] at offset 8 = %d, want 1", got)
	}
	if got := int32(binary.LittleEndian.Uint32(raw[12:16])); got != 224 {
		t.Fatalf("dim[1] at offset 12 = %d, want 224", got)
	}

	elem, layout, err := DecodeRankedTensorType(raw)
	if err != nil {
		t.Fatalf("DecodeRankedTensorType: %v", err)
	}
	if elem != 1 || layout.Rank != 4 || layout.HasStrides {
		t.Fatalf("unexpected decoded metadata: elem=%d layout=%+v", elem, layout)
	}
	wantDims := []int32{1, 224, 224, 3}
	for i, w := range wantDims {
		if layout.Dimensions[i] != w {
			t.Fatalf("dim[%d] = %d, want %d", i, layout.Dimensions[i], w)
		}
	}
}

func TestLayoutStridesAndDynamicDims(t *testing.T) {
	raw, err := EncodeLayout(Layout{
		HasStrides: true,
		Dimensions: []int32{-1, 16},
		Strides:    []uint32{16, 1},
	})
	if err != nil {
		t.Fatalf("EncodeLayout: %v", err)
	}
	if len(raw) != LayoutSize {
		t.Fatalf("len(raw) = %d, want %d", len(raw), LayoutSize)
	}
	got, err := DecodeLayout(raw)
	if err != nil {
		t.Fatalf("DecodeLayout: %v", err)
	}
	if got.Rank != 2 || !got.HasStrides || got.Dimensions[0] != -1 || got.Strides[0] != 16 {
		t.Fatalf("unexpected layout: %+v", got)
	}
}

func TestLayoutBoundsErrors(t *testing.T) {
	if _, err := DecodeLayout(make([]byte, LayoutSize-1)); err == nil {
		t.Fatal("expected error for short layout")
	}
	if _, _, err := DecodeRankedTensorType(make([]byte, RankedTensorTypeSize-1)); err == nil {
		t.Fatal("expected error for short ranked tensor type")
	}
	badRank := make([]byte, LayoutSize)
	binary.LittleEndian.PutUint32(badRank[0:4], 9)
	if _, err := DecodeLayout(badRank); err == nil {
		t.Fatal("expected error for rank > 8")
	}
	if _, err := EncodeRankedTensorType(1, make([]int32, MaxRank+1)); err == nil {
		t.Fatal("expected error encoding rank > 8")
	}
	if _, err := EncodeLayout(Layout{HasStrides: true, Dimensions: []int32{1, 2}, Strides: []uint32{1}}); err == nil {
		t.Fatal("expected error on mismatched strides length")
	}
}

func TestEncodeEnvOptions(t *testing.T) {
	cs := CBytes("/tmp/cache")
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&cs[0])

	buf, err := EncodeEnvOptions([]EnvOptionValue{
		{Tag: 18, AnyType: AnyTypeString, PtrVal: unsafe.Pointer(&cs[0])},
		{Tag: 25, AnyType: AnyTypeInt, IntVal: 3},
	})
	if err != nil {
		t.Fatalf("EncodeEnvOptions: %v", err)
	}
	if len(buf) != 2*EnvOptionSize {
		t.Fatalf("len = %d, want %d", len(buf), 2*EnvOptionSize)
	}
	if got := int32(binary.LittleEndian.Uint32(buf[0:4])); got != 18 {
		t.Fatalf("tag[0] = %d, want 18", got)
	}
	if got := int32(binary.LittleEndian.Uint32(buf[8:12])); got != AnyTypeString {
		t.Fatalf("type[0] = %d, want %d", got, AnyTypeString)
	}
	if got := binary.LittleEndian.Uint64(buf[16:24]); got != uint64(uintptr(unsafe.Pointer(&cs[0]))) {
		t.Fatalf("ptr[0] = %#x, want %#x", got, uintptr(unsafe.Pointer(&cs[0])))
	}
	if got := int32(binary.LittleEndian.Uint32(buf[24:28])); got != 25 {
		t.Fatalf("tag[1] = %d, want 25", got)
	}
	if got := int64(binary.LittleEndian.Uint64(buf[40:48])); got != 3 {
		t.Fatalf("int[1] = %d, want 3", got)
	}
	if _, err := EncodeEnvOptions([]EnvOptionValue{{Tag: 1, AnyType: 99}}); err == nil {
		t.Fatal("expected error for unknown AnyType")
	}
}

func TestQuantizationDecoders(t *testing.T) {
	if got := unsafe.Sizeof(RawPerChannelQuantization{}); got != QuantizationPerChannelSize {
		t.Fatalf("sizeof(RawPerChannelQuantization) = %d, want %d", got, QuantizationPerChannelSize)
	}
	raw := make([]byte, QuantizationPerTensorSize)
	binary.LittleEndian.PutUint32(raw[0:4], math.Float32bits(0.125))
	binary.LittleEndian.PutUint64(raw[8:16], uint64(128))
	scale, zp, err := DecodePerTensorQuantization(raw)
	if err != nil {
		t.Fatalf("DecodePerTensorQuantization: %v", err)
	}
	if scale != 0.125 || zp != 128 {
		t.Fatalf("got scale=%v zp=%d, want 0.125 and 128", scale, zp)
	}

	scales := []float32{0.5, 0.25}
	zps := []int64{1, -2}
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&scales[0])
	pin.Pin(&zps[0])

	dim, outScales, outZPs := CopyPerChannelQuantization(RawPerChannelQuantization{
		QuantizedDimension: 3,
		NumChannels:        2,
		Scales:             unsafe.Pointer(&scales[0]),
		ZeroPoints:         unsafe.Pointer(&zps[0]),
	})
	if dim != 3 || len(outScales) != 2 || outScales[1] != 0.25 || outZPs[1] != -2 {
		t.Fatalf("unexpected per-channel copy: dim=%d scales=%v zps=%v", dim, outScales, outZPs)
	}
}
