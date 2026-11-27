package litert

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litert-go/internal/capi"
)

// EnvOptionTag identifies a LiteRtEnvironment configuration key from
// litert/c/litert_environment_options.h (2.4.1).
type EnvOptionTag int32

const (
	EnvCompilerPluginLibraryDir     EnvOptionTag = 0
	EnvDispatchLibraryDir           EnvOptionTag = 1
	EnvCompilerCacheDir             EnvOptionTag = 18
	EnvRuntimeLibraryDir            EnvOptionTag = 22
	EnvAutoRegisterAccelerators     EnvOptionTag = 24
	EnvMinLoggerSeverity            EnvOptionTag = 25
	EnvCompilerCacheMaxConfigsModel EnvOptionTag = 26
	EnvCompilerCacheMaxTotalSize    EnvOptionTag = 27
)

// EnvOption configures Environment creation.
type EnvOption struct {
	Tag     EnvOptionTag
	StrVal  string
	IntVal  int64
	BoolVal bool
	Kind    EnvOptionKind
}

// EnvOptionKind selects which field in EnvOption is serialized.
type EnvOptionKind int

const (
	EnvKindString EnvOptionKind = iota
	EnvKindInt
	EnvKindBool
)

// EnvStringOption builds a string-valued environment option.
func EnvStringOption(tag EnvOptionTag, value string) EnvOption {
	return EnvOption{Tag: tag, StrVal: value, Kind: EnvKindString}
}

// EnvIntOption builds an integer-valued environment option.
func EnvIntOption(tag EnvOptionTag, value int64) EnvOption {
	return EnvOption{Tag: tag, IntVal: value, Kind: EnvKindInt}
}

// EnvBoolOption builds a boolean-valued environment option.
func EnvBoolOption(tag EnvOptionTag, value bool) EnvOption {
	return EnvOption{Tag: tag, BoolVal: value, Kind: EnvKindBool}
}

// Environment owns a native LiteRtEnvironment handle and any pinned option
// buffers passed at creation.
type Environment struct {
	syms     *capi.Symbols
	handle   uintptr
	pin      runtime.Pinner
	retained [][]byte
	closed   bool
}

// NewEnvironment creates a LiteRtEnvironment with the supplied options.
func NewEnvironment(opts ...EnvOption) (*Environment, error) {
	syms, err := ensureInit("")
	if err != nil {
		return nil, err
	}
	return newEnvironmentWithSymbols(syms, opts...)
}

func newEnvironmentWithSymbols(syms *capi.Symbols, opts ...EnvOption) (*Environment, error) {
	if syms == nil {
		return nil, &Error{Op: "NewEnvironment", Sentinel: ErrClosed}
	}

	env := &Environment{syms: syms}
	var callPin runtime.Pinner
	defer callPin.Unpin()

	numOpts := int32(len(opts))
	callPin.Pin(&numOpts)

	var optsPtr unsafe.Pointer
	if len(opts) > 0 {
		vals, kept, err := serializeEnvOptions(&env.pin, opts)
		if err != nil {
			env.pin.Unpin()
			return nil, err
		}
		buf, err := capi.EncodeEnvOptions(vals)
		if err != nil {
			env.pin.Unpin()
			return nil, err
		}
		env.retained = append(env.retained, kept...)
		env.retained = append(env.retained, buf)
		env.pin.Pin(&buf[0])
		optsPtr = unsafe.Pointer(&buf[0])
	}
	callPin.Pin(&optsPtr)

	var handle uintptr
	st, callErr := syms.CreateEnvironment.CallStatus(
		&callPin,
		unsafe.Pointer(&numOpts),
		unsafe.Pointer(&optsPtr),
		capi.Out(&callPin, &handle),
	)
	if err := statusErr("LiteRtCreateEnvironment", st, callErr); err != nil {
		env.pin.Unpin()
		env.retained = nil
		return nil, err
	}
	env.handle = handle
	return env, nil
}

func serializeEnvOptions(pin *runtime.Pinner, opts []EnvOption) ([]capi.EnvOptionValue, [][]byte, error) {
	vals := make([]capi.EnvOptionValue, len(opts))
	var kept [][]byte
	for i, o := range opts {
		v := capi.EnvOptionValue{Tag: int32(o.Tag)}
		switch o.Kind {
		case EnvKindString:
			cs := capi.CBytes(o.StrVal)
			pin.Pin(&cs[0])
			kept = append(kept, cs)
			v.AnyType = capi.AnyTypeString
			v.PtrVal = unsafe.Pointer(&cs[0])
		case EnvKindInt:
			v.AnyType = capi.AnyTypeInt
			v.IntVal = o.IntVal
		case EnvKindBool:
			v.AnyType = capi.AnyTypeBool
			if o.BoolVal {
				v.IntVal = 1
			}
		default:
			return nil, nil, fmt.Errorf("litert: unknown EnvOptionKind %d", o.Kind)
		}
		vals[i] = v
	}
	return vals, kept, nil
}

func (e *Environment) valid() bool {
	return e != nil && !e.closed && e.handle != 0 && e.syms != nil
}

// AddOptions applies additional options to an open Environment.
func (e *Environment) AddOptions(overwrite bool, opts ...EnvOption) error {
	if !e.valid() {
		return ErrClosed
	}
	if len(opts) == 0 {
		return nil
	}

	vals, kept, err := serializeEnvOptions(&e.pin, opts)
	if err != nil {
		return err
	}
	buf, err := capi.EncodeEnvOptions(vals)
	if err != nil {
		return err
	}
	e.retained = append(e.retained, kept...)
	e.retained = append(e.retained, buf)
	e.pin.Pin(&buf[0])

	var callPin runtime.Pinner
	defer callPin.Unpin()

	h := e.handle
	numOpts := int32(len(opts))
	optsPtr := unsafe.Pointer(&buf[0])
	var ow uint8
	if overwrite {
		ow = 1
	}
	st, callErr := e.syms.AddEnvironmentOptions.CallStatus(
		&callPin,
		capi.Ptr(&callPin, &h),
		capi.Ptr(&callPin, &numOpts),
		capi.Ptr(&callPin, &optsPtr),
		capi.Ptr(&callPin, &ow),
	)
	return statusErr("LiteRtAddEnvironmentOptions", st, callErr)
}

// SupportsFP16 reports whether the runtime environment supports FP16 compute.
func (e *Environment) SupportsFP16() (bool, error) {
	if !e.valid() {
		return false, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := e.handle
	var out byte
	st, callErr := e.syms.EnvironmentSupportsFP16.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Out(&pin, &out),
	)
	if err := statusErr("LiteRtEnvironmentSupportsFP16", st, callErr); err != nil {
		return false, err
	}
	return out != 0, nil
}

// HasGPUEnvironment reports whether a shared GPU environment is attached.
func (e *Environment) HasGPUEnvironment() (bool, error) {
	if !e.valid() {
		return false, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := e.handle
	var out byte
	if err := e.syms.EnvironmentHasGpuEnvironment.CallVoid(
		capi.Ptr(&pin, &h),
		capi.Out(&pin, &out),
	); err != nil {
		return false, statusErr("LiteRtEnvironmentHasGpuEnvironment", 0, err)
	}
	return out != 0, nil
}

// Close destroys the native environment and releases retained option buffers.
// Close is safe to call multiple times.
func (e *Environment) Close() error {
	if e == nil || e.closed {
		return nil
	}
	e.closed = true

	var err error
	if e.handle != 0 && e.syms != nil {
		var pin runtime.Pinner
		h := e.handle
		err = statusErr("LiteRtDestroyEnvironment", 0, e.syms.DestroyEnvironment.CallVoid(capi.Ptr(&pin, &h)))
		pin.Unpin()
		e.handle = 0
	}
	e.pin.Unpin()
	e.retained = nil
	return err
}
