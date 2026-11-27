// Package loader locates and dynamically opens the LiteRT C shared library
// (`libLiteRt.so`, `libLiteRt.dylib`, or `libLiteRt.dll`).
package loader

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/jupiterrider/ffi"
)

// EnvVar names the directory or shared-library file path consulted when no
// explicit directory is supplied to Load or Resolve.
const EnvVar = "LITERT_LIB"

var loadMu sync.Mutex

// LibraryFilename returns the platform filename for the LiteRT C API library.
func LibraryFilename() string {
	switch runtime.GOOS {
	case "windows":
		return "libLiteRt.dll"
	case "darwin":
		return "libLiteRt.dylib"
	default:
		return "libLiteRt.so"
	}
}

// Resolve determines the full path to the LiteRT shared library and its
// containing directory. When dir is empty, LITERT_LIB is consulted.
func Resolve(dir string) (libPath, libDir string, err error) {
	if dir == "" {
		dir = os.Getenv(EnvVar)
	}
	if dir == "" {
		return "", "", fmt.Errorf("loader: library path not set (pass dir or set %s)", EnvVar)
	}

	info, statErr := os.Stat(dir)
	if statErr != nil {
		return "", "", fmt.Errorf("loader: stat %q: %w", dir, statErr)
	}
	if info.IsDir() {
		libDir = dir
		libPath = filepath.Join(dir, LibraryFilename())
	} else {
		libPath = dir
		libDir = filepath.Dir(dir)
	}
	if _, err := os.Stat(libPath); err != nil {
		return "", "", fmt.Errorf("loader: missing library %q: %w", libPath, err)
	}
	return libPath, libDir, nil
}

// Load resolves and opens the LiteRT shared library from dir (or LITERT_LIB).
// On Windows it configures the process DLL search directory so accelerator and
// shader-compiler plugins in the same directory resolve cleanly.
func Load(dir string) (ffi.Lib, string, error) {
	loadMu.Lock()
	defer loadMu.Unlock()

	libPath, libDir, err := Resolve(dir)
	if err != nil {
		return ffi.Lib{}, "", err
	}
	setDllDir(libDir)
	lib, err := ffi.Load(libPath)
	if err != nil {
		return ffi.Lib{}, "", fmt.Errorf("loader: load %q: %w", libPath, err)
	}
	return lib, libDir, nil
}
