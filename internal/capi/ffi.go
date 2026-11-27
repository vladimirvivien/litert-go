// Package capi provides low-level FFI symbol resolution, argument pinning,
// C struct layout codecs, and memory utilities for the LiteRT 2.4.1 C API.
package capi

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

// MissingSymbolError is returned when a required C entry point cannot be
// resolved in the loaded LiteRT shared library.
type MissingSymbolError struct {
	Symbol string
	Cause  error
}

// Error formats the missing symbol name and underlying lookup failure.
func (e *MissingSymbolError) Error() string {
	return fmt.Sprintf("capi: missing C symbol %q: %v", e.Symbol, e.Cause)
}

// Unwrap returns the underlying symbol-lookup error.
func (e *MissingSymbolError) Unwrap() error { return e.Cause }

// PrepFunc resolves and prepares a named C function signature for FFI invocation.
type PrepFunc func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error)

// LazyFun resolves a single C symbol on first invocation and caches the
// prepared call interface or lookup error.
type LazyFun struct {
	name string
	ret  *ffi.Type
	args []*ffi.Type
	prep PrepFunc

	once sync.Once
	fun  ffi.Fun
	err  error
}

// NewLazyFun declares a lazily resolved C function using prep.
func NewLazyFun(prep PrepFunc, name string, ret *ffi.Type, args ...*ffi.Type) *LazyFun {
	return &LazyFun{
		name: name,
		ret:  ret,
		args: args,
		prep: prep,
	}
}

func (l *LazyFun) resolve() error {
	l.once.Do(func() {
		if l.prep == nil {
			l.err = errors.New("symbol resolver not configured")
			return
		}
		l.fun, l.err = l.prep(l.name, l.ret, l.args...)
	})
	if l.err != nil {
		return &MissingSymbolError{Symbol: l.name, Cause: l.err}
	}
	return nil
}

// CallErr resolves the symbol on first call and invokes it. Sub-8-byte integral
// return values are routed through a register-width ffi.Arg scratch word so
// libffi does not overrun narrower caller slots.
func (l *LazyFun) CallErr(ret any, args ...any) error {
	if err := l.resolve(); err != nil {
		return err
	}

	dst, isPtr := ret.(unsafe.Pointer)
	if !isPtr || dst == nil || l.ret == nil || l.ret.Size == 0 || l.ret.Size >= 8 {
		l.fun.Call(ret, args...)
		return nil
	}

	var wide ffi.Arg
	l.fun.Call(unsafe.Pointer(&wide), args...)
	n := int(l.ret.Size)
	copy(unsafe.Slice((*byte)(dst), n), unsafe.Slice((*byte)(unsafe.Pointer(&wide)), n))
	return nil
}

// CallStatus invokes a LiteRtStatus-returning C function and returns the int32
// status code without spilling past a 32-bit slot.
func (l *LazyFun) CallStatus(pin *runtime.Pinner, args ...any) (int32, error) {
	var st int32
	pin.Pin(&st)
	if err := l.CallErr(unsafe.Pointer(&st), args...); err != nil {
		return 0, err
	}
	return st, nil
}

// CallStatusSyscall invokes a pointer/integer-only LiteRtStatus-returning C
// function directly via purego.SyscallN using a caller-owned []uintptr slice,
// avoiding per-call heap allocations on hot paths. Callers must keep any Go
// memory addresses referenced in args pinned for the duration of the call.
func (l *LazyFun) CallStatusSyscall(args []uintptr) (int32, error) {
	if err := l.resolve(); err != nil {
		return 0, err
	}
	if l.fun.Addr == 0 {
		return 0, &MissingSymbolError{Symbol: l.name, Cause: errors.New("resolved symbol has nil address")}
	}
	r1, _, _ := purego.SyscallN(l.fun.Addr, args...)
	return int32(r1), nil
}

// CallVoid invokes a void-returning C function.
func (l *LazyFun) CallVoid(args ...any) error {
	return l.CallErr(nil, args...)
}

// Out pins target and a pointer slot holding its address, returning the
// pointer-to-pointer address expected for C out-parameters (T*).
func Out[T any](pin *runtime.Pinner, target *T) unsafe.Pointer {
	pin.Pin(target)
	slot := unsafe.Pointer(target)
	pin.Pin(&slot)
	return unsafe.Pointer(&slot)
}

// Ptr pins target and returns its address for passing by pointer (const T* / T*)
// when the C signature takes a pointer value directly in the slot.
func Ptr[T any](pin *runtime.Pinner, target *T) unsafe.Pointer {
	pin.Pin(target)
	return unsafe.Pointer(target)
}

// CBytes returns a NUL-terminated byte slice containing s.
func CBytes(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}

// GoString copies a NUL-terminated C string into a Go-owned string.
func GoString(p unsafe.Pointer) string {
	if p == nil {
		return ""
	}
	var n int
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	return strings.Clone(unsafe.String((*byte)(p), n))
}

// GoBytes copies n bytes from C memory into a Go-owned byte slice.
func GoBytes(p unsafe.Pointer, n int) []byte {
	if p == nil || n <= 0 {
		return nil
	}
	out := make([]byte, n)
	copy(out, unsafe.Slice((*byte)(p), n))
	return out
}
