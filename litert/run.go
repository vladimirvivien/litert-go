package litert

import (
	"runtime"
	"unsafe"

	"github.com/jupiterrider/ffi"
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
	// st must be ffi.Arg-sized: libffi widens the LiteRtStatus return to a
	// full ffi_arg and writes 8 bytes through the ret pointer.
	st     ffi.Arg
	cm     uintptr
	sig    uint64
	nin    uint64
	nout   uint64
	inArr  []uintptr
	outArr []uintptr
	inp    unsafe.Pointer
	outp   unsafe.Pointer
	async  byte
	asyncP unsafe.Pointer
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
	r.asyncP = unsafe.Pointer(&r.async)

	r.pin.Pin(&r.st)
	r.pin.Pin(&r.cm)
	r.pin.Pin(&r.sig)
	r.pin.Pin(&r.nin)
	r.pin.Pin(&r.nout)
	r.pin.Pin(&r.inArr[0])
	r.pin.Pin(&r.inp)
	r.pin.Pin(&r.outArr[0])
	r.pin.Pin(&r.outp)
	r.pin.Pin(&r.async)
	r.pin.Pin(&r.asyncP)
	return r
}

// Run invokes the bound signature. The buffer contents may have changed since
// the previous call; the buffer set may not. The runtime rejects output
// buffers that still carry a synchronization event from an earlier
// asynchronous run, so Run detaches stale events from the bound outputs
// first.
func (r *Runner) Run() error {
	for _, h := range r.outArr {
		if err := TensorBuffer(h).ClearEvent(); err != nil {
			return err
		}
	}
	runCompiledModelFunc.Call(&r.st,
		unsafe.Pointer(&r.cm), unsafe.Pointer(&r.sig),
		unsafe.Pointer(&r.nin), unsafe.Pointer(&r.inp),
		unsafe.Pointer(&r.nout), unsafe.Pointer(&r.outp))
	return Status(int32(r.st)).err("LiteRtRunCompiledModel")
}

// RunAsync invokes the bound signature asynchronously when the backend
// supports it, falling back to a synchronous run otherwise; the returned bool
// reports which happened. Under asynchronous execution the output buffers
// carry synchronization events: locking an output (or TensorBuffer.Wait)
// blocks until the producing work finishes. Do not rewrite the input buffers
// until the submitted run has been awaited through one of its outputs.
//
// Output-buffer events are assigned by the runtime, and a submission is
// rejected when an output still carries one, so RunAsync detaches stale events
// from the bound outputs left by the previous submission. Device-side ordering
// is unaffected: runs on one accelerator queue execute in submission order.
func (r *Runner) RunAsync() (bool, error) {
	for _, h := range r.outArr {
		if err := TensorBuffer(h).ClearEvent(); err != nil {
			return false, err
		}
	}
	r.async = 0
	runCompiledModelAsyncFunc.Call(&r.st,
		unsafe.Pointer(&r.cm), unsafe.Pointer(&r.sig),
		unsafe.Pointer(&r.nin), unsafe.Pointer(&r.inp),
		unsafe.Pointer(&r.nout), unsafe.Pointer(&r.outp),
		unsafe.Pointer(&r.asyncP))
	return r.async != 0, Status(int32(r.st)).err("LiteRtRunCompiledModelAsync")
}

// Close releases the pinned call arguments. It does not close the bound
// buffers, which the caller owns.
func (r *Runner) Close() { r.pin.Unpin() }
