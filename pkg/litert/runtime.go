package litert

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/vladimirvivien/litert-go/internal/capi"
	"github.com/vladimirvivien/litert-go/pkg/loader"
)

var (
	runtimeMu     sync.Mutex
	runtimeSyms   *capi.Symbols
	runtimeDir    string
	runtimeInited bool
)

// Init loads the LiteRT shared library from dir (or LITERT_LIB when dir is
// empty) and resolves the LiteRT 2.4.1 C API symbol table. Calling Init more
// than once in the same process is idempotent.
func Init(dir string) error {
	_, err := ensureInit(dir)
	return err
}

func ensureInit(dir string) (*capi.Symbols, error) {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if runtimeInited {
		return runtimeSyms, nil
	}
	lib, resolvedDir, err := loader.Load(dir)
	if err != nil {
		return nil, &Error{Op: "Init", Sentinel: ErrDynamicLoading, Cause: err}
	}
	syms := capi.NewLibrarySymbols(lib)
	runtimeSyms = syms
	runtimeDir = resolvedDir
	runtimeInited = true
	return syms, nil
}

// LoadedDir returns the directory from which libLiteRt was loaded, or empty
// if Init has not been called yet.
func LoadedDir() string {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	return runtimeDir
}

// RuntimeABIVersion queries the loaded LiteRT runtime's ABI version string via
// LiteRtGetRuntimeAbiVersion.
func RuntimeABIVersion() (string, error) {
	syms, err := ensureInit("")
	if err != nil {
		return "", err
	}
	return runtimeABIVersion(syms)
}

func runtimeABIVersion(syms *capi.Symbols) (string, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	var ver unsafe.Pointer
	st, err := syms.GetRuntimeAbiVersion.CallStatus(&pin, capi.Out(&pin, &ver))
	if callErr := statusErr("LiteRtGetRuntimeAbiVersion", st, err); callErr != nil {
		return "", callErr
	}
	return capi.GoString(ver), nil
}
