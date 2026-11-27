package capi

import (
	"bytes"
	"encoding/binary"
	"errors"
	"runtime"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

func TestLazyFun_NarrowReturnDoesNotOverrun(t *testing.T) {
	tests := []struct {
		name    string
		retType *ffi.Type
		value   uint64
	}{
		{name: "sint32", retType: &ffi.TypeSint32, value: 0x11223344},
		{name: "uint8", retType: &ffi.TypeUint8, value: 0x05},
		{name: "sint16", retType: &ffi.TypeSint16, value: 0x2244},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			val := tc.value
			cb := purego.NewCallback(func() uintptr { return uintptr(val) })

			var cif ffi.Cif
			if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 0, tc.retType); st != ffi.OK {
				t.Fatalf("PrepCif failed: %v", st)
			}
			fun := ffi.Fun{Addr: cb, Cif: &cif}

			lf := NewLazyFun(func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
				return fun, nil
			}, "stub_narrow", tc.retType)

			size := int(tc.retType.Size)
			var buf [8]byte
			for i := range buf {
				buf[i] = 0xAA
			}
			var pin runtime.Pinner
			defer pin.Unpin()
			pin.Pin(&buf[0])

			if err := lf.CallErr(unsafe.Pointer(&buf[0])); err != nil {
				t.Fatalf("CallErr: %v", err)
			}

			var got uint64
			switch size {
			case 1:
				got = uint64(buf[0])
			case 2:
				got = uint64(binary.LittleEndian.Uint16(buf[0:2]))
			case 4:
				got = uint64(binary.LittleEndian.Uint32(buf[0:4]))
			}
			if got != tc.value {
				t.Fatalf("return value = %#x, want %#x", got, tc.value)
			}
			for i := size; i < len(buf); i++ {
				if buf[i] != 0xAA {
					t.Fatalf("guard byte %d = %#x, want 0xAA (ret-slot overrun)", i, buf[i])
				}
			}
		})
	}
}

func TestLazyFun_MissingSymbolReturnsError(t *testing.T) {
	sentinel := errors.New("symbol not found")
	lf := NewLazyFun(func(name string, _ *ffi.Type, _ ...*ffi.Type) (ffi.Fun, error) {
		return ffi.Fun{}, sentinel
	}, "LiteRtMissingEntry", &ffi.TypeSint32)

	var pin runtime.Pinner
	defer pin.Unpin()

	_, err := lf.CallStatus(&pin)
	if err == nil {
		t.Fatal("expected error for missing symbol")
	}
	var ms *MissingSymbolError
	if !errors.As(err, &ms) {
		t.Fatalf("expected *MissingSymbolError, got %T (%v)", err, err)
	}
	if ms.Symbol != "LiteRtMissingEntry" {
		t.Fatalf("Symbol = %q, want LiteRtMissingEntry", ms.Symbol)
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("errors.Is did not match cause")
	}
}

func TestLazyFun_NilPrepReturnsError(t *testing.T) {
	lf := NewLazyFun(nil, "LiteRtUnbound", &ffi.TypeVoid)
	if err := lf.CallVoid(); err == nil {
		t.Fatal("expected error with nil PrepFunc")
	}
}

func TestCBytesGoStringGoBytes(t *testing.T) {
	cb := CBytes("litert")
	if len(cb) != 7 || cb[6] != 0 {
		t.Fatalf("unexpected CBytes output: %v", cb)
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&cb[0])

	if got := GoString(unsafe.Pointer(&cb[0])); got != "litert" {
		t.Fatalf("GoString = %q, want %q", got, "litert")
	}
	if got := GoString(nil); got != "" {
		t.Fatalf("GoString(nil) = %q, want empty", got)
	}

	raw := []byte{1, 2, 3, 4}
	pin.Pin(&raw[0])
	cp := GoBytes(unsafe.Pointer(&raw[0]), len(raw))
	raw[0] = 99
	if !bytes.Equal(cp, []byte{1, 2, 3, 4}) {
		t.Fatalf("GoBytes aliased source memory: %v", cp)
	}
	if GoBytes(nil, 4) != nil || GoBytes(unsafe.Pointer(&raw[0]), 0) != nil {
		t.Fatal("expected nil for empty/nil GoBytes")
	}
}

func TestOutAndPtrHelpers(t *testing.T) {
	cb := purego.NewCallback(func(out, val *int32) uintptr {
		*out = *val + 10
		return 0
	})
	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 2, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer); st != ffi.OK {
		t.Fatalf("PrepCif: %v", st)
	}
	fun := ffi.Fun{Addr: cb, Cif: &cif}
	lf := NewLazyFun(func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
		return fun, nil
	}, "stub_out", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	var pin runtime.Pinner
	defer pin.Unpin()
	var out, in int32
	in = 32
	pin.Pin(&in)
	inPtr := unsafe.Pointer(&in)
	pin.Pin(&inPtr)

	st, err := lf.CallStatus(&pin, Out(&pin, &out), unsafe.Pointer(&inPtr))
	if err != nil {
		t.Fatalf("CallStatus: %v", err)
	}
	if st != 0 || out != 42 {
		t.Fatalf("st=%d out=%d, want st=0 out=42", st, out)
	}
}

func TestCallStatusSyscall(t *testing.T) {
	cb := purego.NewCallback(func(a, b uintptr) uintptr {
		return a + b
	})
	lf := NewLazyFun(func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
		return ffi.Fun{Addr: cb}, nil
	}, "stub_syscall", &ffi.TypeSint32, &ffi.TypeUint64, &ffi.TypeUint64)

	st, err := lf.CallStatusSyscall([]uintptr{3, 4})
	if err != nil {
		t.Fatalf("CallStatusSyscall: %v", err)
	}
	if st != 7 {
		t.Fatalf("st = %d, want 7", st)
	}

	missing := NewLazyFun(func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
		return ffi.Fun{}, errors.New("missing")
	}, "stub_missing", &ffi.TypeSint32)
	if _, mErr := missing.CallStatusSyscall(nil); mErr == nil {
		t.Fatal("expected missing symbol error")
	}

	nilAddr := NewLazyFun(func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
		return ffi.Fun{}, nil
	}, "stub_nil", &ffi.TypeSint32)
	if _, nErr := nilAddr.CallStatusSyscall(nil); nErr == nil {
		t.Fatal("expected nil address error")
	}
}
