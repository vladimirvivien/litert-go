package litert

import (
	"testing"
	"unsafe"
)

func TestElementTypeAndTensorType(t *testing.T) {
	tests := []struct {
		elem     ElementType
		wantStr  string
		wantSize int
		wantOK   bool
	}{
		{ElementFloat32, "f32", 4, true},
		{ElementUInt8, "u8", 1, true},
		{ElementInt8, "i8", 1, true},
		{ElementFloat16, "f16", 2, true},
		{ElementBFloat16, "bf16", 2, true},
		{ElementFloat64, "f64", 8, true},
		{ElementUInt16, "u16", 2, true},
		{ElementUInt32, "u32", 4, true},
		{ElementUInt64, "u64", 8, true},
		{ElementComplex128, "c128", 16, true},
		{ElementFloat8E4M3FN, "f8e4m3fn", 1, true},
		{ElementInt4, "i4", 0, false},
		{ElementString, "string", 0, false},
	}

	for _, tc := range tests {
		if got := tc.elem.String(); got != tc.wantStr {
			t.Errorf("ElementType(%d).String() = %q, want %q", tc.elem, got, tc.wantStr)
		}
		sz, ok := tc.elem.ByteSize()
		if sz != tc.wantSize || ok != tc.wantOK {
			t.Errorf("ElementType(%d).ByteSize() = (%d, %v), want (%d, %v)", tc.elem, sz, ok, tc.wantSize, tc.wantOK)
		}
	}

	tt := TensorType{ElementType: ElementUInt8, Shape: []int32{1, 224, 224, 3}}
	if tt.IsDynamic() {
		t.Fatal("static shape reported dynamic")
	}
	if n, ok := tt.NumElements(); !ok || n != 224*224*3 {
		t.Fatalf("NumElements = (%d, %v)", n, ok)
	}
	if sz, ok := tt.ByteSize(); !ok || sz != 224*224*3 {
		t.Fatalf("ByteSize = (%d, %v)", sz, ok)
	}

	dyn := TensorType{ElementType: ElementFloat32, Shape: []int32{-1, 10}}
	if !dyn.IsDynamic() {
		t.Fatal("negative dim should report dynamic")
	}
	if _, ok := dyn.ByteSize(); ok {
		t.Fatal("dynamic tensor ByteSize should report false")
	}

	scalar := TensorType{ElementType: ElementFloat32}
	if n, ok := scalar.NumElements(); !ok || n != 1 {
		t.Fatalf("scalar NumElements = (%d, %v), want (1, true)", n, ok)
	}
}

func TestAllocAlignedBytes(t *testing.T) {
	if AllocAlignedBytes(0) != nil {
		t.Fatal("expected nil for zero size")
	}
	for _, n := range []int{1, 17, 64, 1024, 4099} {
		b := AllocAlignedBytes(n)
		if len(b) != n || cap(b) != n {
			t.Fatalf("AllocAlignedBytes(%d) len=%d cap=%d", n, len(b), cap(b))
		}
		addr := uintptr(unsafe.Pointer(&b[0]))
		if addr%HostMemoryAlignment != 0 {
			t.Fatalf("AllocAlignedBytes(%d) not 64-byte aligned: %#x", n, addr)
		}
	}
}
