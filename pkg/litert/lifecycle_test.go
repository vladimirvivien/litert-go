package litert

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"

	"github.com/vladimirvivien/litert-go/internal/capi"
)

func TestClosedAndZeroValueGuards(t *testing.T) {
	var env *Environment
	if err := env.Close(); err != nil {
		t.Fatalf("nil Environment.Close: %v", err)
	}
	if err := (&Environment{}).AddOptions(false, EnvIntOption(EnvMinLoggerSeverity, 1)); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}

	var opts *Options
	if err := opts.Close(); err != nil {
		t.Fatalf("nil Options.Close: %v", err)
	}
	if err := (&Options{}).SetHardwareAccelerators(AccelCPU); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}

	var model *Model
	if err := model.Close(); err != nil {
		t.Fatalf("nil Model.Close: %v", err)
	}
	if _, err := (&Model{}).NumSignatures(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}

	var buf *TensorBuffer
	if err := buf.Close(); err != nil {
		t.Fatalf("nil TensorBuffer.Close: %v", err)
	}
	if _, err := (&TensorBuffer{}).Size(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
	if _, err := (&TensorBuffer{}).PackedSize(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}

	var cm *CompiledModel
	if err := cm.Close(); err != nil {
		t.Fatalf("nil CompiledModel.Close: %v", err)
	}
	if err := (&CompiledModel{}).Run(context.Background(), 0, nil, nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
	if _, err := (&CompiledModel{}).FullyAccelerated(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
	if _, err := (&CompiledModel{}).NonCPUFullyAccelerated(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
	if err := (&CompiledModel{}).ResizeInput(0, 0, []int32{1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
	if _, err := (&CompiledModel{}).InputBufferRequirements(0, 0); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
	if err := (&Profiler{}).Start(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}

	var runner *Runner
	if err := runner.Close(); err != nil {
		t.Fatalf("nil Runner.Close: %v", err)
	}
	if err := (&Runner{}).Run(context.Background()); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
}

func TestPartialInitRollbackAndDoubleClose(t *testing.T) {
	var created, destroyed atomic.Int32

	createOptsCB := purego.NewCallback(func(out *uintptr) uintptr {
		created.Add(1)
		*out = 0x1234
		return 0
	})
	setAccelFailCB := purego.NewCallback(func(_ uintptr, _ uintptr) uintptr {
		return uintptr(StatusErrorInvalidArgument)
	})
	destroyOptsCB := purego.NewCallback(func(_ uintptr) uintptr {
		destroyed.Add(1)
		return 0
	})

	prep := func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, uint32(len(args)), ret, args...); st != ffi.OK {
			return ffi.Fun{}, errors.New("PrepCif failed")
		}
		switch name {
		case "LiteRtCreateOptions":
			return ffi.Fun{Addr: createOptsCB, Cif: &cif}, nil
		case "LiteRtSetOptionsHardwareAccelerators":
			return ffi.Fun{Addr: setAccelFailCB, Cif: &cif}, nil
		case "LiteRtDestroyOptions":
			return ffi.Fun{Addr: destroyOptsCB, Cif: &cif}, nil
		default:
			return ffi.Fun{}, errors.New("not stubbed")
		}
	}

	syms := capi.NewSymbols(prep)
	if _, err := newOptionsWithSymbols(syms, AccelCPU); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument, got %v", err)
	}
	if created.Load() != 1 || destroyed.Load() != 1 {
		t.Fatalf("partial init leaked Options handle: created=%d destroyed=%d", created.Load(), destroyed.Load())
	}

	// Verify double-Close invokes native destructor only once.
	setAccelOkCB := purego.NewCallback(func(_ uintptr, _ uintptr) uintptr { return 0 })
	symsOK := capi.NewSymbols(func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		if name == "LiteRtSetOptionsHardwareAccelerators" {
			var cif ffi.Cif
			_ = ffi.PrepCif(&cif, ffi.DefaultAbi, uint32(len(args)), ret, args...)
			return ffi.Fun{Addr: setAccelOkCB, Cif: &cif}, nil
		}
		return prep(name, ret, args...)
	})
	o, err := newOptionsWithSymbols(symsOK, AccelCPU)
	if err != nil {
		t.Fatalf("newOptionsWithSymbols: %v", err)
	}
	if err := o.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := o.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if destroyed.Load() != 2 {
		t.Fatalf("expected destroy count 2, got %d", destroyed.Load())
	}
}

func TestHostBufferAlignmentValidation(t *testing.T) {
	syms := capi.NewSymbols(func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
		return ffi.Fun{}, errors.New("unused")
	})
	raw := make([]byte, 128)
	// Force unaligned slice start.
	off := 1
	if uintptr(unsafe.Pointer(&raw[1]))%HostMemoryAlignment == 0 {
		off = 2
	}
	unaligned := raw[off:]
	_, err := newHostBufferWithSymbols(syms, TensorType{ElementType: ElementFloat32, Shape: []int32{4}}, unaligned)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for unaligned host buffer, got %v", err)
	}
}
