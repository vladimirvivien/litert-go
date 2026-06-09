package litert

import (
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// opaqueNoopFree is a non-null payload destructor that does nothing.
// LiteRtCreateOpaqueOptions requires a non-null destructor, and the runtime
// holds the payload pointer (it does not copy it) until the owning options or
// compiled model is destroyed. litert-go keeps the payload bytes alive in
// opaquePayloadRetain and lets the Go GC reclaim them, so the C side must not
// free them.
var opaqueNoopFree = purego.NewCallback(func(uintptr) uintptr { return 0 })

// opaquePayloadRetain keeps opaque-option identifier and payload bytes alive for
// the process lifetime.
var opaquePayloadRetain [][]byte

// AddOpaqueOption attaches a type-erased accelerator option (a payload string
// under an identifier) to the compilation options. The GPU accelerator reads its
// "gpu_options" payload as a TOML document.
func (o Options) AddOpaqueOption(identifier, payload string) error {
	idb := cbytes(identifier)
	pb := cbytes(payload)
	opaquePayloadRetain = append(opaquePayloadRetain, idb, pb)

	opaque, err := createOpaqueOption(&idb[0], &pb[0])
	if err != nil {
		return err
	}
	return o.addOpaqueOption(opaque)
}

func createOpaqueOption(id, payload *byte) (uintptr, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	pin.Pin(id)
	pin.Pin(payload)
	idPtr := unsafe.Pointer(id)
	pin.Pin(&idPtr)
	pbPtr := unsafe.Pointer(payload)
	pin.Pin(&pbPtr)
	dest := opaqueNoopFree
	pin.Pin(&dest)

	var opaque uintptr
	pin.Pin(&opaque)
	op := unsafe.Pointer(&opaque)
	pin.Pin(&op)

	st := invoke(&pin, createOpaqueOptionsFunc,
		unsafe.Pointer(&idPtr), unsafe.Pointer(&pbPtr), unsafe.Pointer(&dest), unsafe.Pointer(&op))
	return opaque, st.err("LiteRtCreateOpaqueOptions")
}

func (o Options) addOpaqueOption(opaque uintptr) error {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := uintptr(o)
	pin.Pin(&h)
	pin.Pin(&opaque)

	st := invoke(&pin, addOpaqueOptionsFunc, unsafe.Pointer(&h), unsafe.Pointer(&opaque))
	return st.err("LiteRtAddOpaqueOptions")
}
