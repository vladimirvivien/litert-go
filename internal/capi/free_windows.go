//go:build windows

package capi

import "unsafe"

// Free is a no-op on Windows because the prebuilt libLiteRt.dll links the CRT
// statically, so freeing its heap allocations from an external CRT DLL handle
// is invalid.
func Free(unsafe.Pointer) {}
