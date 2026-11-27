package litert

import (
	"context"
	"fmt"
	"runtime"
	"time"
	"unsafe"

	"github.com/vladimirvivien/litert-go/internal/capi"
)

// CompiledModel owns an accelerator-compiled LiteRtCompiledModel handle.
type CompiledModel struct {
	syms             *capi.Symbols
	handle           uintptr
	env              *Environment
	pin              runtime.Pinner
	cancel           cancelSlot
	cancelRegistered bool
	closed           bool
}

// Compile compiles model for env using opts (or nil for default CPU options).
func Compile(env *Environment, model *Model, opts *Options) (*CompiledModel, error) {
	if !env.valid() || !model.valid() {
		return nil, ErrClosed
	}
	var optHandle uintptr
	if opts != nil {
		if !opts.valid() {
			return nil, ErrClosed
		}
		optHandle = opts.handle
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	e := env.handle
	m := model.handle
	var handle uintptr
	st, callErr := env.syms.CreateCompiledModel.CallStatus(
		&pin,
		capi.Ptr(&pin, &e),
		capi.Ptr(&pin, &m),
		capi.Ptr(&pin, &optHandle),
		capi.Out(&pin, &handle),
	)
	if err := statusErr("LiteRtCreateCompiledModel", st, callErr); err != nil {
		return nil, err
	}
	return &CompiledModel{
		syms:   env.syms,
		handle: handle,
		env:    env,
	}, nil
}

// Close destroys the compiled model and releases pinned cancellation state.
func (c *CompiledModel) Close() error {
	if c == nil || c.closed {
		return nil
	}
	c.closed = true

	var err error
	if c.handle != 0 && c.syms != nil {
		var pin runtime.Pinner
		h := c.handle
		err = statusErr("LiteRtDestroyCompiledModel", 0, c.syms.DestroyCompiledModel.CallVoid(capi.Ptr(&pin, &h)))
		pin.Unpin()
		c.handle = 0
	}
	c.pin.Unpin()
	return err
}

func (c *CompiledModel) valid() bool {
	return c != nil && !c.closed && c.handle != 0 && c.syms != nil
}

// FullyAccelerated reports whether all operations delegated to requested accelerators.
func (c *CompiledModel) FullyAccelerated() (bool, error) {
	if !c.valid() {
		return false, ErrClosed
	}
	return c.queryBool(c.syms.CompiledModelIsFullyAccelerated, "LiteRtCompiledModelIsFullyAccelerated")
}

// NonCPUFullyAccelerated reports whether 100% of ops delegated to non-CPU accelerators.
func (c *CompiledModel) NonCPUFullyAccelerated() (bool, error) {
	if !c.valid() {
		return false, ErrClosed
	}
	return c.queryBool(c.syms.CompiledModelIsNonCpuFullyAccelerated, "LiteRtCompiledModelIsNonCpuFullyAccelerated")
}

func (c *CompiledModel) queryBool(fn *capi.LazyFun, op string) (bool, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	var out byte
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &out))
	if err := statusErr(op, st, callErr); err != nil {
		return false, err
	}
	return out != 0, nil
}

// LastInferenceDuration returns the runtime-measured duration of the last run.
func (c *CompiledModel) LastInferenceDuration() (time.Duration, error) {
	if !c.valid() {
		return 0, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	var ns int64
	st, callErr := c.syms.GetCompiledModelLastInferenceDuration.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Out(&pin, &ns),
	)
	if err := statusErr("LiteRtGetCompiledModelLastInferenceDuration", st, callErr); err != nil {
		return 0, err
	}
	if ns < 0 {
		return 0, nil
	}
	return time.Duration(ns) * time.Nanosecond, nil
}

// DelegationMetrics returns node and partition counts per delegated accelerator.
func (c *CompiledModel) DelegationMetrics() (DelegationMetrics, error) {
	if !c.valid() {
		return DelegationMetrics{}, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	var dm uintptr
	st, callErr := c.syms.CompiledModelGetDelegationMetrics.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Out(&pin, &dm),
	)
	if err := statusErr("LiteRtCompiledModelGetDelegationMetrics", st, callErr); err != nil {
		return DelegationMetrics{}, err
	}
	if dm == 0 {
		return DelegationMetrics{}, nil
	}

	var totalNodes int32
	st, callErr = c.syms.GetDelegationMetricsTotalNodeCount.CallStatus(
		&pin,
		capi.Ptr(&pin, &dm),
		capi.Out(&pin, &totalNodes),
	)
	if err := statusErr("LiteRtGetDelegationMetricsTotalNodeCount", st, callErr); err != nil {
		return DelegationMetrics{}, err
	}

	var numAccels uint64
	st, callErr = c.syms.GetNumDelegationMetricsAccelerators.CallStatus(
		&pin,
		capi.Ptr(&pin, &dm),
		capi.Out(&pin, &numAccels),
	)
	if err := statusErr("LiteRtGetNumDelegationMetricsAccelerators", st, callErr); err != nil {
		return DelegationMetrics{}, err
	}

	accels := make([]AcceleratorDelegation, 0, numAccels)
	for i := range numAccels {
		ad, adErr := c.readAcceleratorDelegation(dm, i)
		if adErr != nil {
			return DelegationMetrics{}, adErr
		}
		accels = append(accels, ad)
	}
	return DelegationMetrics{
		TotalNodeCount: int(totalNodes),
		Accelerators:   accels,
	}, nil
}

func (c *CompiledModel) readAcceleratorDelegation(dm uintptr, index uint64) (AcceleratorDelegation, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	dmH := dm
	idx := index
	var am uintptr
	st, callErr := c.syms.GetDelegationMetricsAccelerator.CallStatus(
		&pin,
		capi.Ptr(&pin, &dmH),
		capi.Ptr(&pin, &idx),
		capi.Out(&pin, &am),
	)
	if err := statusErr("LiteRtGetDelegationMetricsAccelerator", st, callErr); err != nil {
		return AcceleratorDelegation{}, err
	}

	var namePtr unsafe.Pointer
	st, callErr = c.syms.GetAcceleratorDelegationMetricsName.CallStatus(&pin, capi.Ptr(&pin, &am), capi.Out(&pin, &namePtr))
	if err := statusErr("LiteRtGetAcceleratorDelegationMetricsName", st, callErr); err != nil {
		return AcceleratorDelegation{}, err
	}

	var hw int32
	st, callErr = c.syms.GetAcceleratorDelegationMetricsHardwareType.CallStatus(&pin, capi.Ptr(&pin, &am), capi.Out(&pin, &hw))
	if err := statusErr("LiteRtGetAcceleratorDelegationMetricsHardwareType", st, callErr); err != nil {
		return AcceleratorDelegation{}, err
	}

	var nodes int32
	st, callErr = c.syms.GetAcceleratorDelegationMetricsDelegatedCount.CallStatus(&pin, capi.Ptr(&pin, &am), capi.Out(&pin, &nodes))
	if err := statusErr("LiteRtGetAcceleratorDelegationMetricsDelegatedNodeCount", st, callErr); err != nil {
		return AcceleratorDelegation{}, err
	}

	var parts int32
	st, callErr = c.syms.GetAcceleratorDelegationMetricsPartitionCount.CallStatus(&pin, capi.Ptr(&pin, &am), capi.Out(&pin, &parts))
	if err := statusErr("LiteRtGetAcceleratorDelegationMetricsPartitionCount", st, callErr); err != nil {
		return AcceleratorDelegation{}, err
	}

	return AcceleratorDelegation{
		Name:               capi.GoString(namePtr),
		HardwareType:       Accelerator(hw),
		DelegatedNodeCount: int(nodes),
		PartitionCount:     int(parts),
	}, nil
}

// Errors retrieves error messages buffered on the compiled model when buffer
// error reporting is enabled.
func (c *CompiledModel) Errors() (string, error) {
	if !c.valid() {
		return "", ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	var out unsafe.Pointer
	st, callErr := c.syms.CompiledModelGetErrorMessages.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Out(&pin, &out),
	)
	if err := statusErr("LiteRtCompiledModelGetErrorMessages", st, callErr); err != nil {
		return "", err
	}
	if out == nil {
		return "", nil
	}
	msg := capi.GoString(out)
	capi.Free(out)
	return msg, nil
}

// ClearErrors clears buffered errors on the compiled model.
func (c *CompiledModel) ClearErrors() error {
	if !c.valid() {
		return ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	st, callErr := c.syms.CompiledModelClearErrors.CallStatus(&pin, capi.Ptr(&pin, &h))
	return statusErr("LiteRtCompiledModelClearErrors", st, callErr)
}

// ResizeInput strictly resizes a signature input declaring dynamic dimensions.
func (c *CompiledModel) ResizeInput(sig, input int, dims []int32) error {
	if !c.valid() {
		return ErrClosed
	}
	return c.resizeInput(c.syms.CompiledModelResizeInputTensor, "LiteRtCompiledModelResizeInputTensor", sig, input, dims)
}

// ResizeInputNonStrict resizes a signature input even when its declared shape
// is static; follow with OutputLayouts(sig, numOutputs, true).
func (c *CompiledModel) ResizeInputNonStrict(sig, input int, dims []int32) error {
	if !c.valid() {
		return ErrClosed
	}
	return c.resizeInput(c.syms.CompiledModelResizeInputTensorNonStrict, "LiteRtCompiledModelResizeInputTensorNonStrict", sig, input, dims)
}

func (c *CompiledModel) resizeInput(fn *capi.LazyFun, op string, sig, input int, dims []int32) error {
	if sig < 0 || input < 0 || len(dims) == 0 {
		return &Error{Op: op, Sentinel: ErrInvalidArgument, Msg: "invalid signature, input index, or empty dims"}
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	s := uint64(sig)
	in := uint64(input)
	pin.Pin(&dims[0])
	dimsPtr := unsafe.Pointer(&dims[0])
	nd := uint64(len(dims))

	st, callErr := fn.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &s),
		capi.Ptr(&pin, &in),
		capi.Ptr(&pin, &dimsPtr),
		capi.Ptr(&pin, &nd),
	)
	return statusErr(op, st, callErr)
}

// InputLayout returns the current resolved input layout (shape & strides) after
// any ResizeInput calls.
func (c *CompiledModel) InputLayout(sig, input int) (capi.Layout, error) {
	if !c.valid() {
		return capi.Layout{}, ErrClosed
	}
	if sig < 0 || input < 0 {
		return capi.Layout{}, &Error{Op: "InputLayout", Sentinel: ErrInvalidArgument}
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	s := uint64(sig)
	in := uint64(input)
	raw := make([]byte, capi.LayoutSize)
	pin.Pin(&raw[0])
	rawPtr := unsafe.Pointer(&raw[0])

	st, callErr := c.syms.GetCompiledModelInputTensorLayout.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &s),
		capi.Ptr(&pin, &in),
		capi.Ptr(&pin, &rawPtr),
	)
	if err := statusErr("LiteRtGetCompiledModelInputTensorLayout", st, callErr); err != nil {
		return capi.Layout{}, err
	}
	layout, decErr := capi.DecodeLayout(raw)
	if decErr != nil {
		return capi.Layout{}, &Error{Op: "DecodeLayout", Sentinel: ErrInvalidArgument, Cause: decErr}
	}
	return layout, nil
}

// OutputLayouts returns current output layouts for sig. Set updateAllocation
// true after resizing inputs to propagate shape updates through the graph.
func (c *CompiledModel) OutputLayouts(sig, numOutputs int, updateAllocation bool) ([]capi.Layout, error) {
	if !c.valid() {
		return nil, ErrClosed
	}
	if sig < 0 || numOutputs <= 0 {
		return nil, &Error{Op: "OutputLayouts", Sentinel: ErrInvalidArgument, Msg: "numOutputs must be positive"}
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	s := uint64(sig)
	n := uint64(numOutputs)
	raw := make([]byte, numOutputs*capi.LayoutSize)
	pin.Pin(&raw[0])
	rawPtr := unsafe.Pointer(&raw[0])
	var upd uint8
	if updateAllocation {
		upd = 1
	}

	st, callErr := c.syms.GetCompiledModelOutputTensorLayouts.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &s),
		capi.Ptr(&pin, &n),
		capi.Ptr(&pin, &rawPtr),
		capi.Ptr(&pin, &upd),
	)
	if err := statusErr("LiteRtGetCompiledModelOutputTensorLayouts", st, callErr); err != nil {
		return nil, err
	}
	layouts := make([]capi.Layout, numOutputs)
	for i := range numOutputs {
		off := i * capi.LayoutSize
		l, decErr := capi.DecodeLayout(raw[off : off+capi.LayoutSize])
		if decErr != nil {
			return nil, &Error{Op: "DecodeLayout", Sentinel: ErrInvalidArgument, Cause: decErr}
		}
		layouts[i] = l
	}
	return layouts, nil
}

// InputBufferRequirements queries size, alignment, and supported storage types
// for input index on sig.
func (c *CompiledModel) InputBufferRequirements(sig, input int) (BufferRequirements, error) {
	if !c.valid() {
		return BufferRequirements{}, ErrClosed
	}
	return c.bufferRequirements(c.syms.GetCompiledModelInputBufferRequirements, "LiteRtGetCompiledModelInputBufferRequirements", sig, input)
}

// OutputBufferRequirements queries size, alignment, and supported storage types
// for output index on sig.
func (c *CompiledModel) OutputBufferRequirements(sig, output int) (BufferRequirements, error) {
	if !c.valid() {
		return BufferRequirements{}, ErrClosed
	}
	return c.bufferRequirements(c.syms.GetCompiledModelOutputBufferRequirements, "LiteRtGetCompiledModelOutputBufferRequirements", sig, output)
}

func (c *CompiledModel) bufferRequirements(fn *capi.LazyFun, op string, sig, idx int) (BufferRequirements, error) {
	if sig < 0 || idx < 0 {
		return BufferRequirements{}, &Error{Op: op, Sentinel: ErrInvalidArgument}
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	s := uint64(sig)
	i := uint64(idx)
	var req uintptr
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &s), capi.Ptr(&pin, &i), capi.Out(&pin, &req))
	if err := statusErr(op, st, callErr); err != nil {
		return BufferRequirements{}, err
	}

	var size uint64
	st, callErr = c.syms.GetTensorBufferRequirementsBufferSize.CallStatus(&pin, capi.Ptr(&pin, &req), capi.Out(&pin, &size))
	if err := statusErr("LiteRtGetTensorBufferRequirementsBufferSize", st, callErr); err != nil {
		return BufferRequirements{}, err
	}

	var align uint64
	st, callErr = c.syms.GetTensorBufferRequirementsAlignment.CallStatus(&pin, capi.Ptr(&pin, &req), capi.Out(&pin, &align))
	if err := statusErr("LiteRtGetTensorBufferRequirementsAlignment", st, callErr); err != nil {
		return BufferRequirements{}, err
	}

	var numTypes int32
	st, callErr = c.syms.GetNumTensorBufferRequirementsSupportedTypes.CallStatus(&pin, capi.Ptr(&pin, &req), capi.Out(&pin, &numTypes))
	if err := statusErr("LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes", st, callErr); err != nil {
		return BufferRequirements{}, err
	}

	types := make([]BufferType, 0, max(int(numTypes), 1))
	if numTypes <= 0 {
		types = append(types, BufferHostMemory)
	} else {
		for ti := range int(numTypes) {
			var bt int32
			tIdx := int32(ti)
			st, callErr = c.syms.GetTensorBufferRequirementsSupportedType.CallStatus(
				&pin,
				capi.Ptr(&pin, &req),
				capi.Ptr(&pin, &tIdx),
				capi.Out(&pin, &bt),
			)
			if err := statusErr("LiteRtGetTensorBufferRequirementsSupportedTensorBufferType", st, callErr); err != nil {
				return BufferRequirements{}, err
			}
			types = append(types, BufferType(bt))
		}
	}

	return BufferRequirements{
		BufferSize:     size,
		Alignment:      align,
		SupportedTypes: types,
	}, nil
}

// CreateInputBuffer allocates an optimal TensorBuffer for input index of sig.
func (c *CompiledModel) CreateInputBuffer(sig int, spec TensorSpec, input int) (*TensorBuffer, error) {
	if !c.valid() {
		return nil, ErrClosed
	}
	return c.createBufferFromRequirements(
		c.syms.GetCompiledModelInputBufferRequirements,
		"LiteRtGetCompiledModelInputBufferRequirements",
		sig, input, spec.Type,
	)
}

// CreateOutputBuffer allocates an optimal TensorBuffer for output index of sig.
func (c *CompiledModel) CreateOutputBuffer(sig int, spec TensorSpec, output int) (*TensorBuffer, error) {
	if !c.valid() {
		return nil, ErrClosed
	}
	return c.createBufferFromRequirements(
		c.syms.GetCompiledModelOutputBufferRequirements,
		"LiteRtGetCompiledModelOutputBufferRequirements",
		sig, output, spec.Type,
	)
}

func (c *CompiledModel) createBufferFromRequirements(reqFn *capi.LazyFun, reqOp string, sig, idx int, tensorType TensorType) (*TensorBuffer, error) {
	if !c.valid() || !c.env.valid() {
		return nil, ErrClosed
	}
	if sig < 0 || idx < 0 {
		return nil, &Error{Op: reqOp, Sentinel: ErrInvalidArgument}
	}
	rawType, err := capi.EncodeRankedTensorType(int32(tensorType.ElementType), tensorType.Shape)
	if err != nil {
		return nil, &Error{Op: "CreateManagedTensorBufferFromRequirements", Sentinel: ErrInvalidArgument, Cause: err}
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	s := uint64(sig)
	i := uint64(idx)
	var req uintptr
	st, callErr := reqFn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &s), capi.Ptr(&pin, &i), capi.Out(&pin, &req))
	if err := statusErr(reqOp, st, callErr); err != nil {
		return nil, err
	}

	envH := c.env.handle
	pin.Pin(&rawType[0])
	rawPtr := unsafe.Pointer(&rawType[0])
	var bufHandle uintptr
	st, callErr = c.syms.CreateManagedTensorBufferFromRequirements.CallStatus(
		&pin,
		capi.Ptr(&pin, &envH),
		capi.Ptr(&pin, &rawPtr),
		capi.Ptr(&pin, &req),
		capi.Out(&pin, &bufHandle),
	)
	if err := statusErr("LiteRtCreateManagedTensorBufferFromRequirements", st, callErr); err != nil {
		return nil, err
	}
	return &TensorBuffer{syms: c.syms, handle: bufHandle}, nil
}

// Run executes signature sig synchronously with cancellation support via ctx.
func (c *CompiledModel) Run(ctx context.Context, sig int, inputs, outputs []*TensorBuffer) error {
	if !c.valid() {
		return ErrClosed
	}
	if sig < 0 {
		return &Error{Op: "Run", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("negative signature %d", sig)}
	}
	stop, err := c.armCancel(ctx)
	if err != nil {
		return err
	}
	if stop != nil {
		defer func() { _ = stop() }()
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	inHandles, inPtr, err := pinBufferHandles(&pin, inputs)
	if err != nil {
		return err
	}
	outHandles, outPtr, err := pinBufferHandles(&pin, outputs)
	if err != nil {
		return err
	}
	runtime.KeepAlive(inHandles)
	runtime.KeepAlive(outHandles)

	h := c.handle
	s := uint64(sig)
	nin := uint64(len(inputs))
	nout := uint64(len(outputs))

	st, callErr := c.syms.RunCompiledModel.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &s),
		capi.Ptr(&pin, &nin),
		capi.Ptr(&pin, &inPtr),
		capi.Ptr(&pin, &nout),
		capi.Ptr(&pin, &outPtr),
	)
	return statusErr("LiteRtRunCompiledModel", st, callErr)
}

// RunAsync executes signature sig asynchronously when supported by the backend.
// The returned boolean reports whether execution ran asynchronously; when true,
// callers must await output buffers (for example via TensorBuffer.Wait or Lock)
// before reading results or rewriting inputs.
func (c *CompiledModel) RunAsync(ctx context.Context, sig int, inputs, outputs []*TensorBuffer) (bool, error) {
	if !c.valid() {
		return false, ErrClosed
	}
	if sig < 0 {
		return false, &Error{Op: "RunAsync", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("negative signature %d", sig)}
	}
	stop, err := c.armCancel(ctx)
	if err != nil {
		return false, err
	}
	if stop != nil {
		defer func() { _ = stop() }()
	}

	for _, b := range outputs {
		if b != nil {
			if clearErr := b.ClearEvent(); clearErr != nil {
				return false, clearErr
			}
		}
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	inHandles, inPtr, err := pinBufferHandles(&pin, inputs)
	if err != nil {
		return false, err
	}
	outHandles, outPtr, err := pinBufferHandles(&pin, outputs)
	if err != nil {
		return false, err
	}
	runtime.KeepAlive(inHandles)
	runtime.KeepAlive(outHandles)

	h := c.handle
	s := uint64(sig)
	nin := uint64(len(inputs))
	nout := uint64(len(outputs))
	var asyncFlag byte

	st, callErr := c.syms.RunCompiledModelAsync.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &s),
		capi.Ptr(&pin, &nin),
		capi.Ptr(&pin, &inPtr),
		capi.Ptr(&pin, &nout),
		capi.Ptr(&pin, &outPtr),
		capi.Out(&pin, &asyncFlag),
	)
	if runErr := statusErr("LiteRtRunCompiledModelAsync", st, callErr); runErr != nil {
		return false, runErr
	}
	return asyncFlag != 0, nil
}

// Profiler returns the profiler handle owned by CompiledModel.
func (c *CompiledModel) Profiler() (*Profiler, error) {
	if !c.valid() {
		return nil, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := c.handle
	var ph uintptr
	st, callErr := c.syms.CompiledModelGetProfiler.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &ph))
	if err := statusErr("LiteRtCompiledModelGetProfiler", st, callErr); err != nil {
		return nil, err
	}
	return &Profiler{syms: c.syms, handle: ph, cm: c}, nil
}

// Profiler wraps a LiteRtProfiler owned by a CompiledModel.
type Profiler struct {
	syms   *capi.Symbols
	handle uintptr
	cm     *CompiledModel
}

func (p *Profiler) valid() bool {
	return p != nil && p.handle != 0 && p.syms != nil && p.cm.valid()
}

// Start begins recording profiler events.
func (p *Profiler) Start() error {
	if !p.valid() {
		return ErrClosed
	}
	return p.callSimple(p.syms.StartProfiler, "LiteRtStartProfiler")
}

// Stop stops recording profiler events.
func (p *Profiler) Stop() error {
	if !p.valid() {
		return ErrClosed
	}
	return p.callSimple(p.syms.StopProfiler, "LiteRtStopProfiler")
}

// Reset clears recorded profiler events.
func (p *Profiler) Reset() error {
	if !p.valid() {
		return ErrClosed
	}
	return p.callSimple(p.syms.ResetProfiler, "LiteRtResetProfiler")
}

func (p *Profiler) callSimple(fn *capi.LazyFun, op string) error {
	var pin runtime.Pinner
	defer pin.Unpin()
	h := p.handle
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h))
	return statusErr(op, st, callErr)
}

// NumEvents returns the number of captured profiling events.
func (p *Profiler) NumEvents() (int, error) {
	if !p.valid() {
		return 0, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := p.handle
	var n int32
	st, callErr := p.syms.GetNumProfilerEvents.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &n))
	if err := statusErr("LiteRtGetNumProfilerEvents", st, callErr); err != nil {
		return 0, err
	}
	return int(n), nil
}

// Summary formats and returns the profiler summary string.
func (p *Profiler) Summary() (string, error) {
	if !p.valid() {
		return "", ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	ph := p.handle
	cmh := p.cm.handle
	var out unsafe.Pointer
	st, callErr := p.syms.GetProfileSummary.CallStatus(
		&pin,
		capi.Ptr(&pin, &ph),
		capi.Ptr(&pin, &cmh),
		capi.Out(&pin, &out),
	)
	if err := statusErr("LiteRtGetProfileSummary", st, callErr); err != nil {
		return "", err
	}
	if out == nil {
		return "", nil
	}
	s := capi.GoString(out)
	capi.Free(out)
	return s, nil
}

func pinBufferHandles(pin *runtime.Pinner, bufs []*TensorBuffer) ([]uintptr, unsafe.Pointer, error) {
	if len(bufs) == 0 {
		return nil, nil, nil
	}
	handles := make([]uintptr, len(bufs))
	for i, b := range bufs {
		if !b.valid() {
			return nil, nil, ErrClosed
		}
		handles[i] = b.handle
	}
	pin.Pin(&handles[0])
	return handles, unsafe.Pointer(&handles[0]), nil
}
