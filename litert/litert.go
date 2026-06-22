package litert

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/jupiterrider/ffi"
)

// Status is a LiteRtStatus code.
type Status int32

// StatusOk is kLiteRtStatusOk.
const StatusOk Status = 0

// Ok reports whether the status is success.
func (s Status) Ok() bool { return s == StatusOk }

func (s Status) err(op string) error {
	if s.Ok() {
		return nil
	}
	return fmt.Errorf("litert: %s failed (status %d)", op, int32(s))
}

// HwAccelerator selects the compilation target. Values are LiteRtHwAccelerators.
type HwAccelerator int32

const (
	AccelCPU HwAccelerator = 1 << 0
	AccelGPU HwAccelerator = 1 << 1
	AccelNPU HwAccelerator = 1 << 2
)

// BufferType is a LiteRtTensorBufferType.
type BufferType int32

const BufferHostMemory BufferType = 1

// LockMode is a LiteRtTensorBufferLockMode.
type LockMode int32

const (
	LockRead      LockMode = 0
	LockWrite     LockMode = 1
	LockReadWrite LockMode = 2
)

// ElementType is a LiteRtElementType.
type ElementType int32

const (
	ElementFloat32  ElementType = 1
	ElementInt32    ElementType = 2
	ElementInt64    ElementType = 4
	ElementBool     ElementType = 6
	ElementInt16    ElementType = 7
	ElementInt8     ElementType = 9
	ElementFloat16  ElementType = 10
	ElementInt4     ElementType = 18
	ElementBFloat16 ElementType = 19
)

// String names the element type for logs and error messages.
func (e ElementType) String() string {
	switch e {
	case ElementFloat32:
		return "f32"
	case ElementInt32:
		return "i32"
	case ElementInt64:
		return "i64"
	case ElementBool:
		return "bool"
	case ElementInt16:
		return "i16"
	case ElementInt8:
		return "i8"
	case ElementFloat16:
		return "f16"
	case ElementInt4:
		return "i4"
	case ElementBFloat16:
		return "bf16"
	default:
		return fmt.Sprintf("elem(%d)", int32(e))
	}
}

// Opaque LiteRt handles. Each is a C pointer.
type (
	Environment        uintptr
	Options            uintptr
	Model              uintptr
	Signature          uintptr
	Tensor             uintptr
	CompiledModel      uintptr
	TensorBuffer       uintptr
	bufferRequirements uintptr
)

// TensorType mirrors the element type and shape of LiteRtRankedTensorType.
type TensorType struct {
	ElementType ElementType
	Shape       []int32
}



func decodeTensorType(raw []byte) TensorType {
	et := ElementType(int32(binary.LittleEndian.Uint32(raw[0:4])))
	rank := int(binary.LittleEndian.Uint32(raw[4:8]) & 0x7f)
	shape := make([]int32, rank)
	for i := 0; i < rank; i++ {
		off := dimsOffset + 4*i
		shape[i] = int32(binary.LittleEndian.Uint32(raw[off : off+4]))
	}
	return TensorType{ElementType: et, Shape: shape}
}

func (tt TensorType) raw() []byte {
	b := make([]byte, rankedTensorTypeSize)
	binary.LittleEndian.PutUint32(b[0:4], uint32(tt.ElementType))
	binary.LittleEndian.PutUint32(b[4:8], uint32(len(tt.Shape))) // rank; has_strides[8:12]=0
	for i, d := range tt.Shape {
		binary.LittleEndian.PutUint32(b[dimsOffset+4*i:], uint32(d))
	}
	return b
}

// Each wrapper below owns a runtime.Pinner, pins every Go variable whose
// address reaches the C side, and defers Unpin. A by-value C argument (handle,
// integer) is passed as unsafe.Pointer(&slot) where slot holds the value. A T*
// argument the C side reads or writes through needs a second level: a pinned
// holder hp whose value is &slot, passed as unsafe.Pointer(&hp). See ffi.go.

// EnvOptionTag identifies a LiteRtEnvironment configuration option.
type EnvOptionTag int32

const (
	// EnvCompilerCacheDir specifies a writable directory where the environment's
	// compilers (e.g. CPU/XNNPACK JIT) can serialize/cache compiled kernels.
	EnvCompilerCacheDir EnvOptionTag = 17
	// EnvRuntimeLibraryDir is where the runtime looks for accelerator plugins
	// (e.g. libLiteRtWebGpuAccelerator), so they need not be on the OS path.
	EnvRuntimeLibraryDir EnvOptionTag = 22
	// EnvMinLoggerSeverity sets the minimum logging severity level for the environment:
	// 0 (Verbose), 1 (Info), 2 (Warning), 3 (Error), 4 (Fatal), 5 (None).
	EnvMinLoggerSeverity EnvOptionTag = 25
)

// EnvOption is a LiteRtEnvironment option, which can be string or integer valued.
type EnvOption struct {
	Tag    EnvOptionTag
	Str    string
	IntVal int64
	IsInt  bool
}

// envOptRetain keeps the option strings/buffers alive for the process: LiteRT
// references the option strings beyond the create call (until model init).
var envOptRetain [][]byte

// NewEnvironment creates a LiteRt environment with the given options.
func NewEnvironment(opts ...EnvOption) (Environment, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	var env uintptr
	pin.Pin(&env)
	envp := unsafe.Pointer(&env)
	pin.Pin(&envp)
	numOpts := int32(len(opts))
	pin.Pin(&numOpts)

	// LiteRtEnvOption is 24 bytes: tag (int32) @0, LiteRtAny value @8 whose type
	// (int32) is @8 and string pointer / integer @16.
	const optSize = 24
	const anyTypeInt, anyTypeString = int32(2), int32(8)
	var optsPtr unsafe.Pointer
	if len(opts) > 0 {
		buf := make([]byte, len(opts)*optSize)
		for i, o := range opts {
			base := i * optSize
			*(*int32)(unsafe.Pointer(&buf[base])) = int32(o.Tag)
			if o.IsInt {
				*(*int32)(unsafe.Pointer(&buf[base+8])) = anyTypeInt
				*(*int64)(unsafe.Pointer(&buf[base+16])) = o.IntVal
			} else {
				cs := cbytes(o.Str)
				envOptRetain = append(envOptRetain, cs)
				pin.Pin(&cs[0])
				*(*int32)(unsafe.Pointer(&buf[base+8])) = anyTypeString
				*(*uintptr)(unsafe.Pointer(&buf[base+16])) = uintptr(unsafe.Pointer(&cs[0]))
			}
		}
		envOptRetain = append(envOptRetain, buf)
		pin.Pin(&buf[0])
		optsPtr = unsafe.Pointer(&buf[0])
	}
	pin.Pin(&optsPtr)

	st := invoke(&pin, createEnvironmentFunc,
		unsafe.Pointer(&numOpts), unsafe.Pointer(&optsPtr), unsafe.Pointer(&envp))
	return Environment(env), st.err("LiteRtCreateEnvironment")
}

// Close destroys the environment.
func (e Environment) Close() {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(e)
	pin.Pin(&h)
	destroyEnvironmentFunc.Call(nil, unsafe.Pointer(&h))
}

// NewOptions creates compilation options targeting the given accelerator(s).
// Selecting AccelGPU uses LiteRT's default GPU backend; backend-specific
// configuration travels through Options.AddOpaqueOption.
func NewOptions(accel HwAccelerator) (Options, error) {
	o, err := createOptions()
	if err != nil {
		return 0, err
	}
	if err := o.setHardwareAccelerators(accel); err != nil {
		return 0, err
	}
	return o, nil
}

func createOptions() (Options, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	var o uintptr
	pin.Pin(&o)
	op := unsafe.Pointer(&o)
	pin.Pin(&op)

	st := invoke(&pin, createOptionsFunc, unsafe.Pointer(&op))
	return Options(o), st.err("LiteRtCreateOptions")
}

func (o Options) setHardwareAccelerators(accel HwAccelerator) error {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(o)
	pin.Pin(&h)
	a := int32(accel)
	pin.Pin(&a)

	st := invoke(&pin, setOptionsHardwareAcceleratorsFunc, unsafe.Pointer(&h), unsafe.Pointer(&a))
	return st.err("LiteRtSetOptionsHardwareAccelerators")
}

// Close destroys the options.
func (o Options) Close() {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(o)
	pin.Pin(&h)
	destroyOptionsFunc.Call(nil, unsafe.Pointer(&h))
}

// OpenModel loads a .tflite model from disk. LiteRtCreateModelFromFile takes
// the environment as its first argument, so it must be created first.
func OpenModel(env Environment, path string) (Model, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	e := uintptr(env)
	pin.Pin(&e)
	name := cbytes(path)
	pin.Pin(&name[0])
	namep := unsafe.Pointer(&name[0])
	pin.Pin(&namep)
	var m uintptr
	pin.Pin(&m)
	mp := unsafe.Pointer(&m)
	pin.Pin(&mp)

	st := invoke(&pin, createModelFromFileFunc,
		unsafe.Pointer(&e), unsafe.Pointer(&namep), unsafe.Pointer(&mp))
	return Model(m), st.err("LiteRtCreateModelFromFile")
}

// modelBufferTakesEnv reports whether the loaded runtime's
// LiteRtCreateModelFromBuffer takes a leading LiteRtEnvironment. The 2.1.5
// prebuilt does not; later runtimes do. LiteRtCreateModelFromFd shipped in the
// same API change, so its presence selects the variant.
var modelBufferTakesEnv = sync.OnceValue(func() bool {
	_, err := prepSymbol("LiteRtCreateModelFromFd", &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypeSint32, &ffi.TypePointer)
	return err == nil
})

// OpenModelFromBuffer loads a .tflite model from memory. The LiteRT C API keeps
// a reference to the buffer for the model's lifetime, so the caller must keep
// data alive until the returned Model is closed. The env argument is passed to
// runtimes whose LiteRtCreateModelFromBuffer takes it (post-2.1.5) and ignored
// otherwise.
func OpenModelFromBuffer(env Environment, data []byte) (Model, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("litert: empty model buffer")
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	pin.Pin(&data[0])
	datap := unsafe.Pointer(&data[0])
	pin.Pin(&datap)
	n := uint64(len(data))
	pin.Pin(&n)
	var m uintptr
	pin.Pin(&m)
	mp := unsafe.Pointer(&m)
	pin.Pin(&mp)

	var st Status
	if modelBufferTakesEnv() {
		e := uintptr(env)
		pin.Pin(&e)
		st = invoke(&pin, createModelFromBufferEnvFunc,
			unsafe.Pointer(&e), unsafe.Pointer(&datap), unsafe.Pointer(&n), unsafe.Pointer(&mp))
	} else {
		st = invoke(&pin, createModelFromBufferFunc,
			unsafe.Pointer(&datap), unsafe.Pointer(&n), unsafe.Pointer(&mp))
	}
	return Model(m), st.err("LiteRtCreateModelFromBuffer")
}

// Close destroys the model.
func (m Model) Close() {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(m)
	pin.Pin(&h)
	destroyModelFunc.Call(nil, unsafe.Pointer(&h))
}

// NumSignatures returns the count of named signatures in the model.
func (m Model) NumSignatures() (int, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(m)
	pin.Pin(&h)
	var n uint64
	pin.Pin(&n)
	np := unsafe.Pointer(&n)
	pin.Pin(&np)

	st := invoke(&pin, getNumModelSignaturesFunc, unsafe.Pointer(&h), unsafe.Pointer(&np))
	return int(n), st.err("LiteRtGetNumModelSignatures")
}

// Signature returns the signature at index i.
func (m Model) Signature(i int) (Signature, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(m)
	pin.Pin(&h)
	idx := uint64(i)
	pin.Pin(&idx)
	var sig uintptr
	pin.Pin(&sig)
	sp := unsafe.Pointer(&sig)
	pin.Pin(&sp)

	st := invoke(&pin, getModelSignatureFunc, unsafe.Pointer(&h), unsafe.Pointer(&idx), unsafe.Pointer(&sp))
	return Signature(sig), st.err("LiteRtGetModelSignature")
}

// Key returns the signature's name (e.g. "prefill", "decode").
func (s Signature) Key() (string, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(s)
	pin.Pin(&h)
	var out unsafe.Pointer
	pin.Pin(&out)
	outp := unsafe.Pointer(&out)
	pin.Pin(&outp)

	st := invoke(&pin, getSignatureKeyFunc, unsafe.Pointer(&h), unsafe.Pointer(&outp))
	return goString(out), st.err("LiteRtGetSignatureKey")
}

// NumInputs returns the number of input tensors in the signature.
func (s Signature) NumInputs() (int, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(s)
	pin.Pin(&h)
	var n uint64
	pin.Pin(&n)
	np := unsafe.Pointer(&n)
	pin.Pin(&np)

	st := invoke(&pin, getNumSignatureInputsFunc, unsafe.Pointer(&h), unsafe.Pointer(&np))
	return int(n), st.err("LiteRtGetNumSignatureInputs")
}

// InputName returns the name of input i.
func (s Signature) InputName(i int) (string, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(s)
	pin.Pin(&h)
	idx := uint64(i)
	pin.Pin(&idx)
	var out unsafe.Pointer
	pin.Pin(&out)
	outp := unsafe.Pointer(&out)
	pin.Pin(&outp)

	st := invoke(&pin, getSignatureInputNameFunc, unsafe.Pointer(&h), unsafe.Pointer(&idx), unsafe.Pointer(&outp))
	return goString(out), st.err("LiteRtGetSignatureInputName")
}

// NumOutputs returns the number of output tensors in the signature.
func (s Signature) NumOutputs() (int, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(s)
	pin.Pin(&h)
	var n uint64
	pin.Pin(&n)
	np := unsafe.Pointer(&n)
	pin.Pin(&np)

	st := invoke(&pin, getNumSignatureOutputsFunc, unsafe.Pointer(&h), unsafe.Pointer(&np))
	return int(n), st.err("LiteRtGetNumSignatureOutputs")
}

// OutputName returns the name of output i.
func (s Signature) OutputName(i int) (string, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(s)
	pin.Pin(&h)
	idx := uint64(i)
	pin.Pin(&idx)
	var out unsafe.Pointer
	pin.Pin(&out)
	outp := unsafe.Pointer(&out)
	pin.Pin(&outp)

	st := invoke(&pin, getSignatureOutputNameFunc, unsafe.Pointer(&h), unsafe.Pointer(&idx), unsafe.Pointer(&outp))
	return goString(out), st.err("LiteRtGetSignatureOutputName")
}

// InputType returns the element type and shape of a named input tensor.
func (s Signature) InputType(name string) (TensorType, error) {
	return s.tensorType(name, getSignatureInputTensorFunc, "LiteRtGetSignatureInputTensor")
}

// OutputType returns the element type and shape of a named output tensor.
func (s Signature) OutputType(name string) (TensorType, error) {
	return s.tensorType(name, getSignatureOutputTensorFunc, "LiteRtGetSignatureOutputTensor")
}

func (s Signature) tensorType(name string, fn *lazyFun, op string) (TensorType, error) {
	t, err := s.tensorHandle(name, fn, op)
	if err != nil {
		return TensorType{}, err
	}
	raw, err := rankedTensorType(t)
	if err != nil {
		return TensorType{}, err
	}
	return decodeTensorType(raw), nil
}

func (s Signature) tensorHandle(name string, fn *lazyFun, op string) (uintptr, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(s)
	pin.Pin(&h)
	nb := cbytes(name)
	pin.Pin(&nb[0])
	nbp := unsafe.Pointer(&nb[0])
	pin.Pin(&nbp)
	var t uintptr
	pin.Pin(&t)
	tp := unsafe.Pointer(&t)
	pin.Pin(&tp)

	st := invoke(&pin, fn, unsafe.Pointer(&h), unsafe.Pointer(&nbp), unsafe.Pointer(&tp))
	return t, st.err(op)
}

func rankedTensorType(t uintptr) ([]byte, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := t
	pin.Pin(&h)
	raw := make([]byte, rankedTensorTypeSize)
	pin.Pin(&raw[0])
	rawp := unsafe.Pointer(&raw[0])
	pin.Pin(&rawp)

	st := invoke(&pin, getRankedTensorTypeFunc, unsafe.Pointer(&h), unsafe.Pointer(&rawp))
	return raw, st.err("LiteRtGetRankedTensorType")
}

// Compile compiles the model for the given environment and options.
func Compile(env Environment, m Model, opts Options) (CompiledModel, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	e := uintptr(env)
	pin.Pin(&e)
	mh := uintptr(m)
	pin.Pin(&mh)
	oh := uintptr(opts)
	pin.Pin(&oh)
	var c uintptr
	pin.Pin(&c)
	cp := unsafe.Pointer(&c)
	pin.Pin(&cp)

	st := invoke(&pin, createCompiledModelFunc,
		unsafe.Pointer(&e), unsafe.Pointer(&mh), unsafe.Pointer(&oh), unsafe.Pointer(&cp))
	return CompiledModel(c), st.err("LiteRtCreateCompiledModel")
}

// Close destroys the compiled model.
func (c CompiledModel) Close() {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(c)
	pin.Pin(&h)
	destroyCompiledModelFunc.Call(nil, unsafe.Pointer(&h))
}

// FullyAccelerated reports whether every op was delegated to the selected
// accelerator (i.e. no CPU fallback).
func (c CompiledModel) FullyAccelerated() (bool, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(c)
	pin.Pin(&h)
	var out bool
	pin.Pin(&out)
	outp := unsafe.Pointer(&out)
	pin.Pin(&outp)

	st := invoke(&pin, compiledModelIsFullyAcceleratedFunc, unsafe.Pointer(&h), unsafe.Pointer(&outp))
	return out, st.err("LiteRtCompiledModelIsFullyAccelerated")
}

// ResizeInput resizes a signature input to concrete dims. LLM models declare
// dynamic dims (e.g. the batch dimension); they must be made concrete before
// buffers can be allocated and the signature run. This binds the NonStrict
// variant: these models encode dynamic dims as 0 rather than LiteRT-LM's
// kDynamicDimValue (-1), which the strict resize rejects.
func (c CompiledModel) ResizeInput(sig, input int, dims []int32) error {
	if len(dims) == 0 {
		return fmt.Errorf("litert: empty dims for ResizeInput")
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(c)
	pin.Pin(&h)
	sigN := uint64(sig)
	pin.Pin(&sigN)
	inN := uint64(input)
	pin.Pin(&inN)
	pin.Pin(&dims[0])
	dimsp := unsafe.Pointer(&dims[0])
	pin.Pin(&dimsp)
	nd := uint64(len(dims))
	pin.Pin(&nd)

	st := invoke(&pin, compiledModelResizeInputTensorFunc,
		unsafe.Pointer(&h), unsafe.Pointer(&sigN), unsafe.Pointer(&inN),
		unsafe.Pointer(&dimsp), unsafe.Pointer(&nd))
	return st.err("LiteRtCompiledModelResizeInputTensor")
}

// InputBufferInfo returns the required buffer size and supported buffer type
// for a signature input.
func (c CompiledModel) InputBufferInfo(sig, input int) (size uint64, bt BufferType, err error) {
	return c.bufferInfo(sig, input, getCompiledModelInputBufferRequirementsFunc, "Input")
}

// OutputBufferInfo returns the required buffer size and supported buffer type
// for a signature output.
func (c CompiledModel) OutputBufferInfo(sig, output int) (size uint64, bt BufferType, err error) {
	return c.bufferInfo(sig, output, getCompiledModelOutputBufferRequirementsFunc, "Output")
}

func (c CompiledModel) bufferInfo(sig, idx int, fn *lazyFun, kind string) (uint64, BufferType, error) {
	req, err := c.bufferReq(sig, idx, fn, kind)
	if err != nil {
		return 0, 0, err
	}
	size, err := reqBufferSize(req)
	if err != nil {
		return 0, 0, err
	}
	bt, err := reqBufferType(req)
	if err != nil {
		return 0, 0, err
	}
	return size, bt, nil
}

func (c CompiledModel) bufferReq(sig, idx int, fn *lazyFun, kind string) (uintptr, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(c)
	pin.Pin(&h)
	sigN := uint64(sig)
	pin.Pin(&sigN)
	idxN := uint64(idx)
	pin.Pin(&idxN)
	var req uintptr
	pin.Pin(&req)
	reqp := unsafe.Pointer(&req)
	pin.Pin(&reqp)

	st := invoke(&pin, fn, unsafe.Pointer(&h), unsafe.Pointer(&sigN), unsafe.Pointer(&idxN), unsafe.Pointer(&reqp))
	return req, st.err("LiteRtGetCompiledModel" + kind + "BufferRequirements")
}

func reqBufferSize(req uintptr) (uint64, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := req
	pin.Pin(&h)
	var size uint64
	pin.Pin(&size)
	sizep := unsafe.Pointer(&size)
	pin.Pin(&sizep)

	st := invoke(&pin, getTensorBufferRequirementsBufferSizeFunc, unsafe.Pointer(&h), unsafe.Pointer(&sizep))
	return size, st.err("LiteRtGetTensorBufferRequirementsBufferSize")
}

// reqBufferType returns the first supported buffer type. The C side indexes its
// supported-types vector without a bounds check, so the count is read first; an
// empty list means no type constraint, for which host memory is the CPU default.
func reqBufferType(req uintptr) (BufferType, error) {
	num, err := reqNumTypes(req)
	if err != nil {
		return 0, err
	}
	if num <= 0 {
		return BufferHostMemory, nil
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	h := req
	pin.Pin(&h)
	zero := int32(0)
	pin.Pin(&zero)
	var bt int32
	pin.Pin(&bt)
	btp := unsafe.Pointer(&bt)
	pin.Pin(&btp)

	st := invoke(&pin, getTensorBufferRequirementsSupportedTensorBufferTypeFunc,
		unsafe.Pointer(&h), unsafe.Pointer(&zero), unsafe.Pointer(&btp))
	return BufferType(bt), st.err("LiteRtGetTensorBufferRequirementsSupportedTensorBufferType")
}

func reqNumTypes(req uintptr) (int32, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := req
	pin.Pin(&h)
	var num int32
	pin.Pin(&num)
	nump := unsafe.Pointer(&num)
	pin.Pin(&nump)

	st := invoke(&pin, getNumTensorBufferRequirementsSupportedTypesFunc, unsafe.Pointer(&h), unsafe.Pointer(&nump))
	return num, st.err("LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes")
}

// NewManagedBuffer allocates a managed tensor buffer of the given type, shape,
// and size.
func NewManagedBuffer(env Environment, bt BufferType, tt TensorType, size uint64) (TensorBuffer, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	e := uintptr(env)
	pin.Pin(&e)
	btN := int32(bt)
	pin.Pin(&btN)
	raw := tt.raw()
	pin.Pin(&raw[0])
	rawp := unsafe.Pointer(&raw[0])
	pin.Pin(&rawp)
	sz := size
	pin.Pin(&sz)
	var buf uintptr
	pin.Pin(&buf)
	bufp := unsafe.Pointer(&buf)
	pin.Pin(&bufp)

	st := invoke(&pin, createManagedTensorBufferFunc,
		unsafe.Pointer(&e), unsafe.Pointer(&btN), unsafe.Pointer(&rawp),
		unsafe.Pointer(&sz), unsafe.Pointer(&bufp))
	return TensorBuffer(buf), st.err("LiteRtCreateManagedTensorBuffer")
}

// Lock maps the buffer to host memory and returns its address. The pointer is
// valid until Unlock.
func (b TensorBuffer) Lock(mode LockMode) (unsafe.Pointer, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(b)
	pin.Pin(&h)
	var addr unsafe.Pointer
	pin.Pin(&addr)
	addrp := unsafe.Pointer(&addr)
	pin.Pin(&addrp)
	m := int32(mode)
	pin.Pin(&m)

	st := invoke(&pin, lockTensorBufferFunc, unsafe.Pointer(&h), unsafe.Pointer(&addrp), unsafe.Pointer(&m))
	return addr, st.err("LiteRtLockTensorBuffer")
}

// Unlock unmaps the buffer.
func (b TensorBuffer) Unlock() error {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(b)
	pin.Pin(&h)
	return invoke(&pin, unlockTensorBufferFunc, unsafe.Pointer(&h)).err("LiteRtUnlockTensorBuffer")
}

// Close destroys the tensor buffer.
func (b TensorBuffer) Close() {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(b)
	pin.Pin(&h)
	destroyTensorBufferFunc.Call(nil, unsafe.Pointer(&h))
}

// HasEvent reports whether an asynchronous run attached a synchronization
// event to the buffer.
func (b TensorBuffer) HasEvent() (bool, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(b)
	pin.Pin(&h)
	var has byte
	pin.Pin(&has)
	hasp := unsafe.Pointer(&has)
	pin.Pin(&hasp)
	st := invoke(&pin, hasTensorBufferEventFunc, unsafe.Pointer(&h), unsafe.Pointer(&hasp))
	return has != 0, st.err("LiteRtHasTensorBufferEvent")
}

// ClearEvent detaches the synchronization event an asynchronous run attached
// to the buffer, without waiting on it.
// Clear zeroes the buffer's contents through the runtime, which handles every
// buffer type (host, device, delegate-managed external). Locking and zeroing
// by hand over-runs delegate-managed buffers whose mapped window is smaller
// than the requirements size.
func (b TensorBuffer) Clear() error {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(b)
	pin.Pin(&h)
	return invoke(&pin, clearTensorBufferDataFunc, unsafe.Pointer(&h)).err("LiteRtClearTensorBuffer")
}

// ClearEvent detaches the buffer's synchronization event without waiting on
// it. The runtime rejects output buffers that still carry an event from an
// earlier asynchronous run.
func (b TensorBuffer) ClearEvent() error {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := uintptr(b)
	pin.Pin(&h)
	return invoke(&pin, clearTensorBufferEventFunc, unsafe.Pointer(&h)).err("LiteRtClearTensorBufferEvent")
}

// Wait blocks until the synchronization event an asynchronous run attached to
// the buffer signals, and returns immediately when no event is attached.
// Locking the buffer also waits; Wait synchronizes without mapping the buffer
// to host memory.
func (b TensorBuffer) Wait() error {
	has, err := b.HasEvent()
	if err != nil || !has {
		return err
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(b)
	pin.Pin(&h)

	var ev uintptr
	pin.Pin(&ev)
	evp := unsafe.Pointer(&ev)
	pin.Pin(&evp)
	st := invoke(&pin, getTensorBufferEventFunc, unsafe.Pointer(&h), unsafe.Pointer(&evp))
	if err := st.err("LiteRtGetTensorBufferEvent"); err != nil {
		return err
	}
	if ev == 0 {
		return nil
	}
	timeout := int64(-1)
	pin.Pin(&timeout)
	st = invoke(&pin, waitEventFunc, unsafe.Pointer(&ev), unsafe.Pointer(&timeout))
	return st.err("LiteRtWaitEvent")
}

// Run invokes the signature at sig with the given input and output buffers.
func (c CompiledModel) Run(sig int, inputs, outputs []TensorBuffer) error {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(c)
	pin.Pin(&h)
	sigN := uint64(sig)
	pin.Pin(&sigN)
	nin := uint64(len(inputs))
	pin.Pin(&nin)
	nout := uint64(len(outputs))
	pin.Pin(&nout)

	inArr := handles(inputs)
	pin.Pin(&inArr[0])
	inp := unsafe.Pointer(&inArr[0])
	pin.Pin(&inp)
	outArr := handles(outputs)
	pin.Pin(&outArr[0])
	outp := unsafe.Pointer(&outArr[0])
	pin.Pin(&outp)

	st := invoke(&pin, runCompiledModelFunc,
		unsafe.Pointer(&h), unsafe.Pointer(&sigN),
		unsafe.Pointer(&nin), unsafe.Pointer(&inp),
		unsafe.Pointer(&nout), unsafe.Pointer(&outp))
	return st.err("LiteRtRunCompiledModel")
}

func handles(bufs []TensorBuffer) []uintptr {
	h := make([]uintptr, len(bufs))
	for i, b := range bufs {
		h[i] = uintptr(b)
	}
	return h
}
