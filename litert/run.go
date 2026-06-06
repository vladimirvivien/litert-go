package litert

import (
	"runtime"
	"unsafe"
)

// Runner repeatedly invokes one signature of a compiled model with a fixed set
// of input and output buffers. It pins every Run argument in a single
// runtime.Pinner once, at construction, and reuses the marshalled slots on each
// Run — for hot loops (per-token decode) where the buffer set is constant and
// only the buffer contents change between calls. Compared with CompiledModel.Run
// it does no per-call allocation or pinning.
//
// The caller owns the bound buffers and must keep them open until the Runner is
// closed. A Runner must be used through the pointer returned by NewRunner and is
// not safe for concurrent use.
type Runner struct {
	st     int32
	cm     uintptr
	sig    uint64
	nin    uint64
	nout   uint64
	inArr  []uintptr
	outArr []uintptr
	inp    unsafe.Pointer
	outp   unsafe.Pointer
	pin    runtime.Pinner
}

// NewRunner binds cm's signature at index sig to the given input and output
// buffers. inputs and outputs must be non-empty. Close releases the pinned
// arguments; it does not close the buffers.
func NewRunner(cm CompiledModel, sig int, inputs, outputs []TensorBuffer) *Runner {
	r := &Runner{
		cm:     uintptr(cm),
		sig:    uint64(sig),
		nin:    uint64(len(inputs)),
		nout:   uint64(len(outputs)),
		inArr:  handles(inputs),
		outArr: handles(outputs),
	}
	r.inp = unsafe.Pointer(&r.inArr[0])
	r.outp = unsafe.Pointer(&r.outArr[0])

	r.pin.Pin(&r.st)
	r.pin.Pin(&r.cm)
	r.pin.Pin(&r.sig)
	r.pin.Pin(&r.nin)
	r.pin.Pin(&r.nout)
	r.pin.Pin(&r.inArr[0])
	r.pin.Pin(&r.inp)
	r.pin.Pin(&r.outArr[0])
	r.pin.Pin(&r.outp)
	return r
}

// Run invokes the bound signature. The buffer contents may have changed since
// the previous call; the buffer set may not.
func (r *Runner) Run() error {
	runCompiledModelFunc.Call(&r.st,
		unsafe.Pointer(&r.cm), unsafe.Pointer(&r.sig),
		unsafe.Pointer(&r.nin), unsafe.Pointer(&r.inp),
		unsafe.Pointer(&r.nout), unsafe.Pointer(&r.outp))
	return Status(r.st).err("LiteRtRunCompiledModel")
}

// Close releases the pinned call arguments. It does not close the bound
// buffers, which the caller owns.
func (r *Runner) Close() { r.pin.Unpin() }
