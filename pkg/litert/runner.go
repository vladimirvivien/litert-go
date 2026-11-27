package litert

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"
)

// Runner binds a CompiledModel signature to a fixed set of input and output
// TensorBuffers and pins all FFI argument slots once at construction so repeated
// synchronous calls run with zero heap allocations.
type Runner struct {
	cm         *CompiledModel
	inBufs     []*TensorBuffer
	outBufs    []*TensorBuffer
	inIndex    map[string]int
	outIndex   map[string]int
	ownedBufs  bool
	inHandles  []uintptr
	outHandles []uintptr
	asyncFlag  byte

	syncArgs  []uintptr
	asyncArgs []uintptr

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
		cm:      cm,
		inBufs:  append([]*TensorBuffer(nil), inputs...),
		outBufs: append([]*TensorBuffer(nil), outputs...),
	}

	var inAddr, outAddr uintptr
	if len(inputs) > 0 {
		r.inHandles = make([]uintptr, len(inputs))
		for i, b := range inputs {
			r.inHandles[i] = b.handle
		}
		r.pin.Pin(&r.inHandles[0])
		inAddr = uintptr(unsafe.Pointer(&r.inHandles[0]))
	}
	if len(outputs) > 0 {
		r.outHandles = make([]uintptr, len(outputs))
		for i, b := range outputs {
			r.outHandles[i] = b.handle
		}
		r.pin.Pin(&r.outHandles[0])
		outAddr = uintptr(unsafe.Pointer(&r.outHandles[0]))
	}

	r.pin.Pin(&r.asyncFlag)
	asyncAddr := uintptr(unsafe.Pointer(&r.asyncFlag))

	r.syncArgs = []uintptr{
		cm.handle,
		uintptr(sig),
		uintptr(len(inputs)),
		inAddr,
		uintptr(len(outputs)),
		outAddr,
	}
	r.asyncArgs = []uintptr{
		cm.handle,
		uintptr(sig),
		uintptr(len(inputs)),
		inAddr,
		uintptr(len(outputs)),
		outAddr,
		asyncAddr,
	}
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

	st, callErr := r.cm.syms.RunCompiledModel.CallStatusSyscall(r.syncArgs)
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
	st, callErr := r.cm.syms.RunCompiledModelAsync.CallStatusSyscall(r.asyncArgs)
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

// InputByName returns the bound input buffer named name, or nil if absent.
func (r *Runner) InputByName(name string) *TensorBuffer {
	if r == nil || r.inIndex == nil {
		return nil
	}
	idx, ok := r.inIndex[name]
	if !ok {
		return nil
	}
	return r.Input(idx)
}

// OutputByName returns the bound output buffer named name, or nil if absent.
func (r *Runner) OutputByName(name string) *TensorBuffer {
	if r == nil || r.outIndex == nil {
		return nil
	}
	idx, ok := r.outIndex[name]
	if !ok {
		return nil
	}
	return r.Output(idx)
}

func (r *Runner) setTensorNames(inNames, outNames []string) {
	if len(inNames) > 0 {
		r.inIndex = make(map[string]int, len(inNames))
		for i, name := range inNames {
			if name != "" {
				r.inIndex[name] = i
			}
		}
	}
	if len(outNames) > 0 {
		r.outIndex = make(map[string]int, len(outNames))
		for i, name := range outNames {
			if name != "" {
				r.outIndex[name] = i
			}
		}
	}
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
