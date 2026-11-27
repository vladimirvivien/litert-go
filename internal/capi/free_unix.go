//go:build !windows

package capi

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	freeOnce sync.Once
	freeFun  ffi.Fun
	freeErr  error
)

func initFree() {
	candidates := []string{"libc.so.6", "libSystem.B.dylib", "/usr/lib/libSystem.B.dylib", "libc.so.7", "libc.so"}
	if runtime.GOOS == "darwin" {
		candidates = []string{"/usr/lib/libSystem.B.dylib", "libSystem.B.dylib"}
	}
	for _, libName := range candidates {
		lib, err := ffi.Load(libName)
		if err != nil {
			continue
		}
		fn, err := lib.Prep("free", &ffi.TypeVoid, &ffi.TypePointer)
		if err == nil {
			freeFun = fn
			return
		}
	}
	freeErr = errFreeUnavailable
}

var errFreeUnavailable = &MissingSymbolError{Symbol: "free"}

// Free releases C heap memory allocated by LiteRT C APIs on Unix platforms.
func Free(p unsafe.Pointer) {
	if p == nil {
		return
	}
	freeOnce.Do(initFree)
	if freeErr != nil {
		return
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	slot := p
	pin.Pin(&slot)
	freeFun.Call(nil, unsafe.Pointer(&slot))
}
