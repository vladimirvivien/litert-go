//go:build windows

package lm

import (
	"io"
	"os"
	"syscall"
	"unsafe"
)

type windowsMmapCloser struct {
	hMap syscall.Handle
	addr uintptr
}

func (c *windowsMmapCloser) Close() error {
	err1 := syscall.UnmapViewOfFile(c.addr)
	err2 := syscall.CloseHandle(c.hMap)
	if err1 != nil {
		return err1
	}
	return err2
}

func mmapModel(path string) ([]byte, io.Closer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	fi, err := file.Stat()
	if err != nil {
		return nil, nil, err
	}
	size := fi.Size()

	// Create file mapping
	hMap, err := syscall.CreateFileMapping(syscall.Handle(file.Fd()), nil, syscall.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return nil, nil, err
	}

	// Map view of file
	addr, err := syscall.MapViewOfFile(hMap, syscall.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		_ = syscall.CloseHandle(hMap)
		return nil, nil, err
	}

	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)
	return data, &windowsMmapCloser{hMap: hMap, addr: addr}, nil
}
