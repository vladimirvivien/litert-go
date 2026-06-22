//go:build !windows

package lm

import (
	"io"
	"os"
	"syscall"
)

type unixMmapCloser struct {
	data []byte
}

func (c *unixMmapCloser) Close() error {
	return syscall.Munmap(c.data)
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

	// Memory map the file read-only
	data, err := syscall.Mmap(int(file.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}

	return data, &unixMmapCloser{data: data}, nil
}
