package litert

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/vladimirvivien/litert-go/internal/capi"
)

// HostMemoryAlignment is LITERT_HOST_MEMORY_BUFFER_ALIGNMENT (64 bytes).
const HostMemoryAlignment = 64

// AllocAlignedBytes allocates a byte slice of length n whose first element is
// aligned to HostMemoryAlignment (64 bytes) for zero-copy host TensorBuffers.
func AllocAlignedBytes(n int) []byte {
	if n <= 0 {
		return nil
	}
	raw := make([]byte, n+HostMemoryAlignment-1)
	addr := uintptr(unsafe.Pointer(&raw[0]))
	rem := int(addr % HostMemoryAlignment)
	off := 0
	if rem != 0 {
		off = HostMemoryAlignment - rem
	}
	return raw[off : off+n : off+n]
}

type pinnedHostRef struct {
	refs atomic.Int32
	pin  runtime.Pinner
	data []byte
}

func newPinnedHostRef(data []byte) *pinnedHostRef {
	ref := &pinnedHostRef{data: data}
	ref.refs.Store(1)
	ref.pin.Pin(&ref.data[0])
	return ref
}

func (r *pinnedHostRef) retain() {
	r.refs.Add(1)
}

func (r *pinnedHostRef) release() {
	if r == nil {
		return
	}
	if r.refs.Add(-1) == 0 {
		r.pin.Unpin()
		r.data = nil
	}
}

// TensorBuffer owns a native LiteRtTensorBuffer handle.
type TensorBuffer struct {
	syms    *capi.Symbols
	handle  uintptr
	hostRef *pinnedHostRef
	closed  bool
}

// NewManagedBuffer allocates a LiteRT-managed tensor buffer of bufferType,
// tensorType, and byte size.
func NewManagedBuffer(env *Environment, bufferType BufferType, tensorType TensorType, size uint64) (*TensorBuffer, error) {
	if !env.valid() {
		return nil, ErrClosed
	}
	rawType, err := capi.EncodeRankedTensorType(int32(tensorType.ElementType), tensorType.Shape)
	if err != nil {
		return nil, &Error{Op: "NewManagedBuffer", Sentinel: ErrInvalidArgument, Cause: err}
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	e := env.handle
	bt := int32(bufferType)
	pin.Pin(&rawType[0])
	rawPtr := unsafe.Pointer(&rawType[0])
	sz := size
	var handle uintptr

	st, callErr := env.syms.CreateManagedTensorBuffer.CallStatus(
		&pin,
		capi.Ptr(&pin, &e),
		capi.Ptr(&pin, &bt),
		capi.Ptr(&pin, &rawPtr),
		capi.Ptr(&pin, &sz),
		capi.Out(&pin, &handle),
	)
	if err := statusErr("LiteRtCreateManagedTensorBuffer", st, callErr); err != nil {
		return nil, err
	}
	return &TensorBuffer{syms: env.syms, handle: handle}, nil
}

// NewBufferFromRequirements allocates an alignment-aware managed tensor buffer
// matching compiled-model BufferRequirements.
func NewBufferFromRequirements(env *Environment, tensorType TensorType, reqs BufferRequirements) (*TensorBuffer, error) {
	if len(reqs.SupportedTypes) == 0 {
		return NewManagedBuffer(env, BufferHostMemory, tensorType, reqs.BufferSize)
	}
	bt := reqs.SupportedTypes[0]
	size := reqs.BufferSize
	if reqs.Alignment > 1 && size%reqs.Alignment != 0 {
		size = ((size + reqs.Alignment - 1) / reqs.Alignment) * reqs.Alignment
	}
	return NewManagedBuffer(env, bt, tensorType, size)
}

// NewHostBuffer wraps an existing 64-byte-aligned Go byte slice in a zero-copy
// host-memory TensorBuffer. The slice remains pinned until the TensorBuffer (and
// any duplicates) are closed.
func NewHostBuffer(tensorType TensorType, data []byte) (*TensorBuffer, error) {
	syms, err := ensureInit("")
	if err != nil {
		return nil, err
	}
	return newHostBufferWithSymbols(syms, tensorType, data)
}

func newHostBufferWithSymbols(syms *capi.Symbols, tensorType TensorType, data []byte) (*TensorBuffer, error) {
	if syms == nil {
		return nil, &Error{Op: "NewHostBuffer", Sentinel: ErrClosed}
	}
	if len(data) == 0 {
		return nil, &Error{Op: "NewHostBuffer", Sentinel: ErrInvalidArgument, Msg: "empty host buffer"}
	}
	if uintptr(unsafe.Pointer(&data[0]))%HostMemoryAlignment != 0 {
		return nil, &Error{
			Op:       "NewHostBuffer",
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("host buffer must be %d-byte aligned (use AllocAlignedBytes)", HostMemoryAlignment),
		}
	}

	rawType, err := capi.EncodeRankedTensorType(int32(tensorType.ElementType), tensorType.Shape)
	if err != nil {
		return nil, &Error{Op: "NewHostBuffer", Sentinel: ErrInvalidArgument, Cause: err}
	}

	hostRef := newPinnedHostRef(data)
	var callPin runtime.Pinner
	defer callPin.Unpin()

	callPin.Pin(&rawType[0])
	rawPtr := unsafe.Pointer(&rawType[0])
	dataPtr := unsafe.Pointer(&data[0])
	sz := uint64(len(data))
	var dealloc uintptr
	var handle uintptr

	st, callErr := syms.CreateTensorBufferFromHostMemory.CallStatus(
		&callPin,
		capi.Ptr(&callPin, &rawPtr),
		capi.Ptr(&callPin, &dataPtr),
		capi.Ptr(&callPin, &sz),
		capi.Ptr(&callPin, &dealloc),
		capi.Out(&callPin, &handle),
	)
	if err := statusErr("LiteRtCreateTensorBufferFromHostMemory", st, callErr); err != nil {
		hostRef.release()
		return nil, err
	}
	return &TensorBuffer{syms: syms, handle: handle, hostRef: hostRef}, nil
}

// Duplicate increments the native tensor buffer reference count and returns an
// independently closable TensorBuffer sharing the underlying storage.
func (b *TensorBuffer) Duplicate() (*TensorBuffer, error) {
	if !b.valid() {
		return nil, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	st, callErr := b.syms.DuplicateTensorBuffer.CallStatus(&pin, capi.Ptr(&pin, &h))
	if err := statusErr("LiteRtDuplicateTensorBuffer", st, callErr); err != nil {
		return nil, err
	}
	if b.hostRef != nil {
		b.hostRef.retain()
	}
	return &TensorBuffer{
		syms:    b.syms,
		handle:  b.handle,
		hostRef: b.hostRef,
	}, nil
}

// Close destroys the native tensor buffer reference and releases any host pin.
func (b *TensorBuffer) Close() error {
	if b == nil || b.closed {
		return nil
	}
	b.closed = true

	var err error
	if b.handle != 0 && b.syms != nil {
		var pin runtime.Pinner
		h := b.handle
		err = statusErr("LiteRtDestroyTensorBuffer", 0, b.syms.DestroyTensorBuffer.CallVoid(capi.Ptr(&pin, &h)))
		pin.Unpin()
		b.handle = 0
	}
	b.hostRef.release()
	b.hostRef = nil
	return err
}

// Type returns the underlying BufferType.
func (b *TensorBuffer) Type() (BufferType, error) {
	if !b.valid() {
		return BufferUnknown, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	var bt int32
	st, callErr := b.syms.GetTensorBufferType.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &bt))
	if err := statusErr("LiteRtGetTensorBufferType", st, callErr); err != nil {
		return BufferUnknown, err
	}
	return BufferType(bt), nil
}

// TensorType returns the ranked tensor type stored on the buffer.
func (b *TensorBuffer) TensorType() (TensorType, error) {
	if !b.valid() {
		return TensorType{}, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	raw := make([]byte, capi.RankedTensorTypeSize)
	pin.Pin(&raw[0])
	rawPtr := unsafe.Pointer(&raw[0])

	st, callErr := b.syms.GetTensorBufferTensorType.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &rawPtr))
	if err := statusErr("LiteRtGetTensorBufferTensorType", st, callErr); err != nil {
		return TensorType{}, err
	}
	elem, layout, decErr := capi.DecodeRankedTensorType(raw)
	if decErr != nil {
		return TensorType{}, &Error{Op: "DecodeRankedTensorType", Sentinel: ErrInvalidArgument, Cause: decErr}
	}
	return TensorType{
		ElementType: ElementType(elem),
		Shape:       layout.Dimensions,
		HasStrides:  layout.HasStrides,
		Strides:     layout.Strides,
	}, nil
}

// Size returns the underlying hardware buffer allocation size in bytes.
func (b *TensorBuffer) Size() (uint64, error) {
	if !b.valid() {
		return 0, ErrClosed
	}
	return b.querySize(b.syms.GetTensorBufferSize, "LiteRtGetTensorBufferSize")
}

// PackedSize returns the packed payload size in bytes used for host read/write.
func (b *TensorBuffer) PackedSize() (uint64, error) {
	if !b.valid() {
		return 0, ErrClosed
	}
	return b.querySize(b.syms.GetTensorBufferPackedSize, "LiteRtGetTensorBufferPackedSize")
}

func (b *TensorBuffer) valid() bool {
	return b != nil && !b.closed && b.handle != 0 && b.syms != nil
}

func (b *TensorBuffer) querySize(fn *capi.LazyFun, op string) (uint64, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	var sz uint64
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &sz))
	if err := statusErr(op, st, callErr); err != nil {
		return 0, err
	}
	return sz, nil
}

// Lock maps the buffer into host memory for reading or writing until Unlock.
func (b *TensorBuffer) Lock(mode LockMode) (unsafe.Pointer, error) {
	if !b.valid() {
		return nil, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	m := int32(mode)
	var addr unsafe.Pointer
	st, callErr := b.syms.LockTensorBuffer.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Out(&pin, &addr),
		capi.Ptr(&pin, &m),
	)
	if err := statusErr("LiteRtLockTensorBuffer", st, callErr); err != nil {
		return nil, err
	}
	return addr, nil
}

// Unlock unmaps a previously locked buffer.
func (b *TensorBuffer) Unlock() error {
	if !b.valid() {
		return ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	st, callErr := b.syms.UnlockTensorBuffer.CallStatus(&pin, capi.Ptr(&pin, &h))
	return statusErr("LiteRtUnlockTensorBuffer", st, callErr)
}

// WriteBytes copies src into the buffer's packed host view.
func (b *TensorBuffer) WriteBytes(src []byte) error {
	if len(src) == 0 {
		return nil
	}
	packed, err := b.PackedSize()
	if err != nil {
		return err
	}
	if uint64(len(src)) > packed {
		return &Error{
			Op:       "WriteBytes",
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("input length %d exceeds buffer packed size %d", len(src), packed),
		}
	}
	addr, err := b.Lock(LockWrite)
	if err != nil {
		return err
	}
	copy(unsafe.Slice((*byte)(addr), len(src)), src)
	return b.Unlock()
}

// ReadBytes copies bytes from the buffer's packed host view into dst.
func (b *TensorBuffer) ReadBytes(dst []byte) error {
	if len(dst) == 0 {
		return nil
	}
	packed, err := b.PackedSize()
	if err != nil {
		return err
	}
	if uint64(len(dst)) > packed {
		return &Error{
			Op:       "ReadBytes",
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("destination length %d exceeds buffer packed size %d", len(dst), packed),
		}
	}
	addr, err := b.Lock(LockRead)
	if err != nil {
		return err
	}
	copy(dst, unsafe.Slice((*byte)(addr), len(dst)))
	return b.Unlock()
}

// ReadAllBytes reads the entire packed tensor payload into a new byte slice.
func (b *TensorBuffer) ReadAllBytes() ([]byte, error) {
	packed, err := b.PackedSize()
	if err != nil {
		return nil, err
	}
	out := make([]byte, packed)
	if err := b.ReadBytes(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Clear zeroes the buffer contents through the runtime.
func (b *TensorBuffer) Clear() error {
	if !b.valid() {
		return ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	st, callErr := b.syms.ClearTensorBuffer.CallStatus(&pin, capi.Ptr(&pin, &h))
	return statusErr("LiteRtClearTensorBuffer", st, callErr)
}

// HasEvent reports whether an asynchronous run attached a synchronization event.
func (b *TensorBuffer) HasEvent() (bool, error) {
	if !b.valid() {
		return false, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	var has byte
	st, callErr := b.syms.HasTensorBufferEvent.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &has))
	if err := statusErr("LiteRtHasTensorBufferEvent", st, callErr); err != nil {
		return false, err
	}
	return has != 0, nil
}

// Event returns the synchronization event attached to the buffer, or nil when
// none is attached. The returned Event is owned by the buffer and must not be
// closed independently.
func (b *TensorBuffer) Event() (*Event, error) {
	if !b.valid() {
		return nil, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	var evHandle uintptr
	st, callErr := b.syms.GetTensorBufferEvent.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &evHandle))
	if err := statusErr("LiteRtGetTensorBufferEvent", st, callErr); err != nil {
		return nil, err
	}
	if evHandle == 0 {
		return nil, nil
	}
	return &Event{syms: b.syms, handle: evHandle}, nil
}

// ClearEvent detaches any synchronization event attached to the buffer.
func (b *TensorBuffer) ClearEvent() error {
	if !b.valid() {
		return ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := b.handle
	st, callErr := b.syms.ClearTensorBufferEvent.CallStatus(&pin, capi.Ptr(&pin, &h))
	return statusErr("LiteRtClearTensorBufferEvent", st, callErr)
}

// Wait blocks until any attached synchronization event signals or timeout
// expires (pass a negative duration to wait indefinitely).
func (b *TensorBuffer) Wait(timeout time.Duration) error {
	has, err := b.HasEvent()
	if err != nil || !has {
		return err
	}
	ev, err := b.Event()
	if err != nil || ev == nil {
		return err
	}
	return ev.Wait(timeout)
}

// Event wraps a LiteRtEvent synchronization primitive.
type Event struct {
	syms   *capi.Symbols
	handle uintptr
}

// Wait blocks until the event signals or timeout elapses (< 0 waits forever).
func (e *Event) Wait(timeout time.Duration) error {
	if e == nil || e.handle == 0 || e.syms == nil {
		return ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := e.handle
	ms := int64(-1)
	if timeout >= 0 {
		ms = timeout.Milliseconds()
	}
	st, callErr := e.syms.WaitEvent.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &ms))
	return statusErr("LiteRtWaitEvent", st, callErr)
}

// IsSignaled reports whether the event has already signaled.
func (e *Event) IsSignaled() (bool, error) {
	if e == nil || e.handle == 0 || e.syms == nil {
		return false, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := e.handle
	var sig byte
	st, callErr := e.syms.IsEventSignaled.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &sig))
	if err := statusErr("LiteRtIsEventSignaled", st, callErr); err != nil {
		return false, err
	}
	return sig != 0, nil
}
