package litert

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/jupiterrider/ffi"
)

// Runner binds a CompiledModel signature to a fixed set of input and output
// TensorBuffers and pins all FFI argument slots once at construction so repeated
// synchronous calls run with zero heap allocations.
type Runner struct {
	cm         *CompiledModel
	inBufs     []*TensorBuffer
	outBufs    []*TensorBuffer
	ownedBufs  bool
	inHandles  []uintptr
	outHandles []uintptr

	stWide    ffi.Arg
	cmHandle  uintptr
	sigIndex  uint64
	numIn     uint64
	inPtr     unsafe.Pointer
	numOut    uint64
	outPtr    unsafe.Pointer
	asyncFlag byte
	asyncPtr  unsafe.Pointer

	syncArgs  []unsafe.Pointer
	asyncArgs []unsafe.Pointer

	pin       runtime.Pinner
	usedAsync bool
	closed    bool
}

// NewRunner creates a Runner binding cm signature sig to inputs and outputs.
// Callers retain ownership of inputs and outputs unless constructed through
// Engine.NewRunner.
func NewRunner(cm *CompiledModel, sig int, inputs, outputs []*TensorBuffer) (*Runner, error) {
	if !cm.valid() {
		return nil, ErrClosed
	}
	if sig < 0 {
		return nil, &Error{Op: "NewRunner", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("negative signature %d", sig)}
	}
	for _, b := range inputs {
		if !b.valid() {
			return nil, ErrClosed
		}
	}
	for _, b := range outputs {
		if !b.valid() {
			return nil, ErrClosed
		}
	}

	r := &Runner{
		cm:       cm,
		inBufs:   append([]*TensorBuffer(nil), inputs...),
		outBufs:  append([]*TensorBuffer(nil), outputs...),
		cmHandle: cm.handle,
		sigIndex: uint64(sig),
		numIn:    uint64(len(inputs)),
		numOut:   uint64(len(outputs)),
	}

	if len(inputs) > 0 {
		r.inHandles = make([]uintptr, len(inputs))
		for i, b := range inputs {
			r.inHandles[i] = b.handle
		}
		r.inPtr = unsafe.Pointer(&r.inHandles[0])
		r.pin.Pin(&r.inHandles[0])
	}
	if len(outputs) > 0 {
		r.outHandles = make([]uintptr, len(outputs))
		for i, b := range outputs {
			r.outHandles[i] = b.handle
		}
		r.outPtr = unsafe.Pointer(&r.outHandles[0])
		r.pin.Pin(&r.outHandles[0])
	}

	r.asyncPtr = unsafe.Pointer(&r.asyncFlag)

	r.pin.Pin(&r.stWide)
	r.pin.Pin(&r.cmHandle)
	r.pin.Pin(&r.sigIndex)
	r.pin.Pin(&r.numIn)
	r.pin.Pin(&r.inPtr)
	r.pin.Pin(&r.numOut)
	r.pin.Pin(&r.outPtr)
	r.pin.Pin(&r.asyncFlag)
	r.pin.Pin(&r.asyncPtr)

	r.syncArgs = []unsafe.Pointer{
		unsafe.Pointer(&r.cmHandle),
		unsafe.Pointer(&r.sigIndex),
		unsafe.Pointer(&r.numIn),
		unsafe.Pointer(&r.inPtr),
		unsafe.Pointer(&r.numOut),
		unsafe.Pointer(&r.outPtr),
	}
	r.asyncArgs = []unsafe.Pointer{
		unsafe.Pointer(&r.cmHandle),
		unsafe.Pointer(&r.sigIndex),
		unsafe.Pointer(&r.numIn),
		unsafe.Pointer(&r.inPtr),
		unsafe.Pointer(&r.numOut),
		unsafe.Pointer(&r.outPtr),
		unsafe.Pointer(&r.asyncPtr),
	}
	r.pin.Pin(&r.syncArgs[0])
	r.pin.Pin(&r.asyncArgs[0])
	return r, nil
}

// Run executes the bound signature synchronously using pre-pinned slots.
func (r *Runner) Run(ctx context.Context) error {
	if r == nil || r.closed || !r.cm.valid() {
		return ErrClosed
	}
	stop, err := r.cm.armCancel(ctx)
	if err != nil {
		return err
	}
	if stop != nil {
		defer func() { _ = stop() }()
	}

	if r.usedAsync {
		for _, b := range r.outBufs {
			if clearErr := b.ClearEvent(); clearErr != nil {
				return clearErr
			}
		}
		r.usedAsync = false
	}

	st, callErr := r.cm.syms.RunCompiledModel.CallStatusWide(&r.stWide, r.syncArgs...)
	return statusErr("LiteRtRunCompiledModel", st, callErr)
}

// RunAsync executes the bound signature asynchronously when supported by the
// backend, clearing stale output events from prior asynchronous runs first.
func (r *Runner) RunAsync(ctx context.Context) (bool, error) {
	if r == nil || r.closed || !r.cm.valid() {
		return false, ErrClosed
	}
	stop, err := r.cm.armCancel(ctx)
	if err != nil {
		return false, err
	}
	if stop != nil {
		defer func() { _ = stop() }()
	}

	for _, b := range r.outBufs {
		if clearErr := b.ClearEvent(); clearErr != nil {
			return false, clearErr
		}
	}
	r.asyncFlag = 0
	st, callErr := r.cm.syms.RunCompiledModelAsync.CallStatusWide(&r.stWide, r.asyncArgs...)
	if err := statusErr("LiteRtRunCompiledModelAsync", st, callErr); err != nil {
		return false, err
	}
	r.usedAsync = r.asyncFlag != 0
	return r.usedAsync, nil
}

// Input returns bound input buffer i, or nil if out of range.
func (r *Runner) Input(i int) *TensorBuffer {
	if r == nil || i < 0 || i >= len(r.inBufs) {
		return nil
	}
	return r.inBufs[i]
}

// Output returns bound output buffer i, or nil if out of range.
func (r *Runner) Output(i int) *TensorBuffer {
	if r == nil || i < 0 || i >= len(r.outBufs) {
		return nil
	}
	return r.outBufs[i]
}

// NumInputs returns the number of bound input buffers.
func (r *Runner) NumInputs() int { return len(r.inBufs) }

// NumOutputs returns the number of bound output buffers.
func (r *Runner) NumOutputs() int { return len(r.outBufs) }

// Close unpins the runner's argument slots (and closes buffers if owned).
func (r *Runner) Close() error {
	if r == nil || r.closed {
		return nil
	}
	r.closed = true
	r.pin.Unpin()
	if r.ownedBufs {
		for _, b := range r.inBufs {
			_ = b.Close()
		}
		for _, b := range r.outBufs {
			_ = b.Close()
		}
	}
	return nil
}
