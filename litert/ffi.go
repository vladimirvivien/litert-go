// Package litert is a no-CGO Go binding for the LiteRT C API (the
// CompiledModel interface), bound via purego and jupiterrider/ffi. LiteRT
// handles are opaque C pointers, so the binding wraps each as a typed
// uintptr, resolves C symbols lazily on first use, and pins every Go
// variable whose address crosses the boundary (the pinning rules are at the
// bottom of this file).
//
// The surface covers what an LLM executor needs: environment and
// compilation options (including accelerator opaque options), model loading
// and signature introspection, compiled-model execution (synchronous,
// asynchronous with buffer events, and the pinned-argument Runner for hot
// loops), and tensor buffers.
package litert

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/jupiterrider/ffi"
)

// EnvVar names the directory (or full file path) holding the LiteRT C shared
// library. Load consults it when no path is passed.
const EnvVar = "LITERT_LIB"

var (
	loadMu    sync.Mutex
	libHandle ffi.Lib
	loaded    bool
)

// Load opens the LiteRT C shared library (libLiteRt) from dir. When dir is
// empty the LITERT_LIB environment variable is used; either may name the
// containing directory or the library file directly. Load is idempotent.
func Load(dir string) error {
	loadMu.Lock()
	defer loadMu.Unlock()
	if loaded {
		return nil
	}
	if dir == "" {
		dir = os.Getenv(EnvVar)
	}
	if dir == "" {
		return fmt.Errorf("litert: library path not set (pass dir or set %s)", EnvVar)
	}
	path := dir
	if info, err := os.Stat(dir); err == nil && info.IsDir() {
		path = filepath.Join(dir, libFilename())
	}
	lib, err := ffi.Load(path)
	if err != nil {
		return fmt.Errorf("litert: load %q: %w", path, err)
	}
	libHandle = lib
	loaded = true
	return nil
}

// libFilename returns the platform filename for the LiteRT C library. The
// prebuilt LiteRT library keeps the "lib" prefix on every platform, including
// Windows.
func libFilename() string {
	switch runtime.GOOS {
	case "windows":
		return "libLiteRt.dll"
	case "darwin":
		return "libLiteRt.dylib"
	default:
		return "libLiteRt.so"
	}
}

// prepSymbol resolves a symbol against the loaded library. It is a package
// variable so tests can inject failures without touching libHandle.
var prepSymbol = func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
	return libHandle.Prep(name, ret, args...)
}

// lazyFun resolves one C symbol against libHandle on first Call. Storing one
// error per symbol keeps the declaration surface to a single newLazyFun line
// per C entry point.
type lazyFun struct {
	name string
	ret  *ffi.Type
	args []*ffi.Type

	once sync.Once
	fun  ffi.Fun
	err  error
}

func newLazyFun(name string, ret *ffi.Type, args ...*ffi.Type) *lazyFun {
	return &lazyFun{name: name, ret: ret, args: args}
}

// Call resolves the symbol on first invocation, then forwards to ffi.Fun.Call.
// A missing symbol panics with the symbol name — the actionable failure when a
// staged library predates a binding the Go side knows about.
func (l *lazyFun) Call(ret any, args ...any) {
	l.once.Do(func() {
		l.fun, l.err = prepSymbol(l.name, l.ret, l.args...)
	})
	if l.err != nil {
		panic(fmt.Errorf("litert: missing C symbol %q in loaded library: %w", l.name, l.err))
	}
	l.fun.Call(ret, args...)
}

// invoke calls a LiteRtStatus-returning C function. Each entry in args is an
// unsafe.Pointer in libffi's calling convention: the address of a slot holding
// the argument value (a by-value argument such as a handle or integer) or the
// address of a slot holding the pointer to pass (a T* argument the C side reads
// or writes through). The caller must already have pinned every such slot in
// pin; invoke pins only the status return slot. The caller's deferred
// pin.Unpin releases them all once the call has returned.
//
// The return slot must be ffi.Arg-sized: libffi widens integral returns to a
// full ffi_arg, writing 8 bytes through the ret pointer. A narrower slot is a
// 4-byte heap overrun into the adjacent allocation on every call.
func invoke(pin *runtime.Pinner, fn *lazyFun, args ...any) Status {
	var st ffi.Arg
	pin.Pin(&st)
	fn.Call(&st, args...)
	return Status(int32(st))
}

// cbytes returns a NUL-terminated copy of s. The caller must pin the result
// (pin.Pin(&b[0])) for the duration of any call that reads it.
func cbytes(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}

// Passing a Go pointer to C across an ffi call is governed by one rule: every
// Go variable whose address reaches the C side — directly as an argument value
// or indirectly as the pointer a T* argument is read from or written through —
// must be pinned with runtime.Pinner for the duration of the call. The Go stack
// can move mid-call (stack growth during ffi marshalling), and the stack mover
// does not rewrite addresses laundered through unsafe.Pointer; an unpinned
// stack variable would leave the C side reading or writing a stale address.
// Pinning forces the object to the heap (the address handed to Pin escapes) and
// holds it fixed and live until Unpin.
//
// Scope one Pinner to a single C call: pin the arguments, invoke, Unpin. Do not
// accumulate pins across several chained calls on one Pinner — a wrapper that
// makes multiple C calls (e.g. bufferInfo) uses a separate Pinner per call.
// Each wrapper owns its Pinner(s), pins its argument slots, and defers Unpin;
// see the call sites in litert.go.

// goString copies a NUL-terminated C string at p into a Go string. The copy
// goes through a Go-owned slice so the result never aliases C memory, which the
// callee may reuse or overwrite.
func goString(p unsafe.Pointer) string {
	if p == nil {
		return ""
	}
	var n int
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	s := strings.Clone(unsafe.String((*byte)(p), n))
	return s
}
