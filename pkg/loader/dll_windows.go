//go:build windows

package loader

import (
	"syscall"
	"unsafe"
)

var setDllDirectory = syscall.NewLazyDLL("kernel32.dll").NewProc("SetDllDirectoryW")

func setDllDir(dir string) {
	if dir == "" {
		return
	}
	p, err := syscall.UTF16PtrFromString(dir)
	if err == nil {
		_, _, _ = setDllDirectory.Call(uintptr(unsafe.Pointer(p)))
	}
}
