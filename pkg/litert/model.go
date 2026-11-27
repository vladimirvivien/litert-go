package litert

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litert-go/internal/capi"
)

// Model owns a loaded LiteRtModel and any Go byte slice pinned for its lifetime.
type Model struct {
	syms    *capi.Symbols
	handle  uintptr
	backing []byte
	pin     runtime.Pinner
	closed  bool
}

func (m *Model) valid() bool {
	return m != nil && !m.closed && m.handle != 0 && m.syms != nil
}

// OpenModelFromFile loads a .tflite model file from disk.
func OpenModelFromFile(env *Environment, path string) (*Model, error) {
	if !env.valid() {
		return nil, ErrClosed
	}
	if path == "" {
		return nil, &Error{Op: "OpenModelFromFile", Sentinel: ErrInvalidArgument, Msg: "empty model path"}
	}

	var pin runtime.Pinner
	defer pin.Unpin()

	e := env.handle
	pathBytes := capi.CBytes(path)
	pin.Pin(&pathBytes[0])
	pathPtr := unsafe.Pointer(&pathBytes[0])
	var handle uintptr

	st, callErr := env.syms.CreateModelFromFile.CallStatus(
		&pin,
		capi.Ptr(&pin, &e),
		capi.Ptr(&pin, &pathPtr),
		capi.Out(&pin, &handle),
	)
	if err := statusErr("LiteRtCreateModelFromFile", st, callErr); err != nil {
		return nil, err
	}
	return &Model{syms: env.syms, handle: handle}, nil
}

// OpenModelFromBuffer loads a .tflite model from an in-memory byte slice.
// The Model pins data for its entire lifetime and unpins it on Close. Callers
// must not mutate data while the Model is open.
func OpenModelFromBuffer(env *Environment, data []byte) (*Model, error) {
	if !env.valid() {
		return nil, ErrClosed
	}
	if len(data) == 0 {
		return nil, &Error{Op: "OpenModelFromBuffer", Sentinel: ErrInvalidArgument, Msg: "empty model buffer"}
	}

	m := &Model{
		syms:    env.syms,
		backing: data,
	}
	m.pin.Pin(&m.backing[0])

	var callPin runtime.Pinner
	defer callPin.Unpin()

	e := env.handle
	dataPtr := unsafe.Pointer(&m.backing[0])
	size := uint64(len(m.backing))
	var handle uintptr

	st, callErr := env.syms.CreateModelFromBuffer.CallStatus(
		&callPin,
		capi.Ptr(&callPin, &e),
		capi.Ptr(&callPin, &dataPtr),
		capi.Ptr(&callPin, &size),
		capi.Out(&callPin, &handle),
	)
	if err := statusErr("LiteRtCreateModelFromBuffer", st, callErr); err != nil {
		m.pin.Unpin()
		m.backing = nil
		return nil, err
	}
	m.handle = handle
	return m, nil
}

// Close destroys the model and releases any pinned backing buffer.
func (m *Model) Close() error {
	if m == nil || m.closed {
		return nil
	}
	m.closed = true

	var err error
	if m.handle != 0 && m.syms != nil {
		var pin runtime.Pinner
		h := m.handle
		err = statusErr("LiteRtDestroyModel", 0, m.syms.DestroyModel.CallVoid(capi.Ptr(&pin, &h)))
		pin.Unpin()
		m.handle = 0
	}
	m.pin.Unpin()
	m.backing = nil
	return err
}

// Metadata reads an embedded model metadata buffer by key (for example TFLite
// associated label maps or flatbuffer metadata).
func (m *Model) Metadata(key string) ([]byte, error) {
	if !m.valid() {
		return nil, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := m.handle
	keyBytes := capi.CBytes(key)
	pin.Pin(&keyBytes[0])
	keyPtr := unsafe.Pointer(&keyBytes[0])
	var buf unsafe.Pointer
	var size uint64

	st, callErr := m.syms.GetModelMetadata.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &keyPtr),
		capi.Out(&pin, &buf),
		capi.Out(&pin, &size),
	)
	if err := statusErr("LiteRtGetModelMetadata", st, callErr); err != nil {
		return nil, err
	}
	return capi.GoBytes(buf, int(size)), nil
}

// NumSignatures returns the number of named signatures defined on the model.
func (m *Model) NumSignatures() (int, error) {
	if !m.valid() {
		return 0, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := m.handle
	var n uint64
	st, callErr := m.syms.GetNumModelSignatures.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &n))
	if err := statusErr("LiteRtGetNumModelSignatures", st, callErr); err != nil {
		return 0, err
	}
	return int(n), nil
}

// Signature returns metadata for signature index i.
func (m *Model) Signature(i int) (*Signature, error) {
	if !m.valid() {
		return nil, ErrClosed
	}
	if i < 0 {
		return nil, &Error{Op: "Signature", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("negative signature index %d", i)}
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := m.handle
	idx := uint64(i)
	var handle uintptr
	st, callErr := m.syms.GetModelSignature.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &idx),
		capi.Out(&pin, &handle),
	)
	if err := statusErr("LiteRtGetModelSignature", st, callErr); err != nil {
		return nil, err
	}

	key, err := signatureKey(m.syms, handle)
	if err != nil {
		return nil, err
	}
	inputs, err := readSignatureTensors(m.syms, handle, true)
	if err != nil {
		return nil, err
	}
	outputs, err := readSignatureTensors(m.syms, handle, false)
	if err != nil {
		return nil, err
	}
	return &Signature{
		key:     key,
		index:   i,
		inputs:  inputs,
		outputs: outputs,
	}, nil
}

// Signatures returns every signature defined on the model.
func (m *Model) Signatures() ([]*Signature, error) {
	n, err := m.NumSignatures()
	if err != nil {
		return nil, err
	}
	sigs := make([]*Signature, 0, n)
	for i := range n {
		sig, sigErr := m.Signature(i)
		if sigErr != nil {
			return nil, sigErr
		}
		sigs = append(sigs, sig)
	}
	return sigs, nil
}

// SignatureByKey finds the signature named key.
func (m *Model) SignatureByKey(key string) (*Signature, error) {
	n, err := m.NumSignatures()
	if err != nil {
		return nil, err
	}
	for i := range n {
		sig, sigErr := m.Signature(i)
		if sigErr != nil {
			return nil, sigErr
		}
		if sig.key == key {
			return sig, nil
		}
	}
	return nil, &Error{Op: "SignatureByKey", Sentinel: ErrNotFound, Msg: fmt.Sprintf("signature %q not found", key)}
}

// DefaultSignature returns signature 0 when signatures exist, or falls back to
// inspecting the main subgraph for classic signatureless .tflite models.
func (m *Model) DefaultSignature() (*Signature, error) {
	n, err := m.NumSignatures()
	if err != nil {
		return nil, err
	}
	if n > 0 {
		return m.Signature(0)
	}
	idx, err := m.MainSubgraphIndex()
	if err != nil {
		return nil, err
	}
	sg, err := m.Subgraph(idx)
	if err != nil {
		return nil, err
	}
	return &Signature{
		key:     sg.name,
		index:   0,
		inputs:  sg.inputs,
		outputs: sg.outputs,
	}, nil
}

// NumSubgraphs returns the number of subgraphs in the model.
func (m *Model) NumSubgraphs() (int, error) {
	if !m.valid() {
		return 0, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := m.handle
	var n uint64
	st, callErr := m.syms.GetNumModelSubgraphs.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &n))
	if err := statusErr("LiteRtGetNumModelSubgraphs", st, callErr); err != nil {
		return 0, err
	}
	return int(n), nil
}

// MainSubgraphIndex returns the entry subgraph index.
func (m *Model) MainSubgraphIndex() (int, error) {
	if !m.valid() {
		return 0, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := m.handle
	var idx uint64
	st, callErr := m.syms.GetMainModelSubgraphIndex.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &idx))
	if err := statusErr("LiteRtGetMainModelSubgraphIndex", st, callErr); err != nil {
		return 0, err
	}
	return int(idx), nil
}

// Subgraph returns metadata for subgraph index i.
func (m *Model) Subgraph(i int) (*Subgraph, error) {
	if !m.valid() {
		return nil, ErrClosed
	}
	if i < 0 {
		return nil, &Error{Op: "Subgraph", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("negative subgraph index %d", i)}
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := m.handle
	idx := uint64(i)
	var handle uintptr
	st, callErr := m.syms.GetModelSubgraph.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &idx),
		capi.Out(&pin, &handle),
	)
	if err := statusErr("LiteRtGetModelSubgraph", st, callErr); err != nil {
		return nil, err
	}

	name, err := subgraphName(m.syms, handle)
	if err != nil {
		return nil, err
	}
	inputs, err := readSubgraphTensors(m.syms, handle, true)
	if err != nil {
		return nil, err
	}
	outputs, err := readSubgraphTensors(m.syms, handle, false)
	if err != nil {
		return nil, err
	}
	return &Subgraph{
		name:    name,
		index:   i,
		inputs:  inputs,
		outputs: outputs,
	}, nil
}

// Signature describes a named entry point on a LiteRT Model.
type Signature struct {
	key     string
	index   int
	inputs  []TensorSpec
	outputs []TensorSpec
}

// Key returns the signature key string.
func (s *Signature) Key() string { return s.key }

// Index returns the zero-based signature index.
func (s *Signature) Index() int { return s.index }

// Inputs returns a copy of the ordered input tensor specifications.
func (s *Signature) Inputs() []TensorSpec {
	out := make([]TensorSpec, len(s.inputs))
	copy(out, s.inputs)
	return out
}

// Outputs returns a copy of the ordered output tensor specifications.
func (s *Signature) Outputs() []TensorSpec {
	out := make([]TensorSpec, len(s.outputs))
	copy(out, s.outputs)
	return out
}

// InputSpec looks up an input tensor specification and its index by name.
func (s *Signature) InputSpec(name string) (TensorSpec, int, error) {
	return findTensorSpec("InputSpec", s.inputs, name)
}

// OutputSpec looks up an output tensor specification and its index by name.
func (s *Signature) OutputSpec(name string) (TensorSpec, int, error) {
	return findTensorSpec("OutputSpec", s.outputs, name)
}

// Subgraph describes a subgraph's name and boundary tensors.
type Subgraph struct {
	name    string
	index   int
	inputs  []TensorSpec
	outputs []TensorSpec
}

// Name returns the subgraph name.
func (sg *Subgraph) Name() string { return sg.name }

// Index returns the zero-based subgraph index.
func (sg *Subgraph) Index() int { return sg.index }

// Inputs returns a copy of the subgraph input specifications.
func (sg *Subgraph) Inputs() []TensorSpec {
	out := make([]TensorSpec, len(sg.inputs))
	copy(out, sg.inputs)
	return out
}

// Outputs returns a copy of the subgraph output specifications.
func (sg *Subgraph) Outputs() []TensorSpec {
	out := make([]TensorSpec, len(sg.outputs))
	copy(out, sg.outputs)
	return out
}

func findTensorSpec(op string, specs []TensorSpec, name string) (TensorSpec, int, error) {
	for i, sp := range specs {
		if sp.Name == name {
			return sp, i, nil
		}
	}
	return TensorSpec{}, -1, &Error{Op: op, Sentinel: ErrNotFound, Msg: fmt.Sprintf("tensor %q not found", name)}
}

func signatureKey(syms *capi.Symbols, sig uintptr) (string, error) {
	return callOutCString(syms.GetSignatureKey, "LiteRtGetSignatureKey", sig)
}

func subgraphName(syms *capi.Symbols, sg uintptr) (string, error) {
	return callOutCString(syms.GetSubgraphName, "LiteRtGetSubgraphName", sg)
}

func tensorName(syms *capi.Symbols, t uintptr) (string, error) {
	return callOutCString(syms.GetTensorName, "LiteRtGetTensorName", t)
}

func callOutCString(fn *capi.LazyFun, op string, handle uintptr) (string, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := handle
	var out unsafe.Pointer
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &out))
	if err := statusErr(op, st, callErr); err != nil {
		return "", err
	}
	return capi.GoString(out), nil
}

func readSignatureTensors(syms *capi.Symbols, sig uintptr, isInput bool) ([]TensorSpec, error) {
	countFn, countOp := syms.GetNumSignatureOutputs, "LiteRtGetNumSignatureOutputs"
	nameFn, nameOp := syms.GetSignatureOutputName, "LiteRtGetSignatureOutputName"
	tensorFn, tensorOp := syms.GetSignatureOutputTensorByIndex, "LiteRtGetSignatureOutputTensorByIndex"
	if isInput {
		countFn, countOp = syms.GetNumSignatureInputs, "LiteRtGetNumSignatureInputs"
		nameFn, nameOp = syms.GetSignatureInputName, "LiteRtGetSignatureInputName"
		tensorFn, tensorOp = syms.GetSignatureInputTensorByIndex, "LiteRtGetSignatureInputTensorByIndex"
	}

	n, err := callCount(countFn, countOp, sig)
	if err != nil {
		return nil, err
	}
	specs := make([]TensorSpec, 0, n)
	for i := range n {
		name, nameErr := callIndexedCString(nameFn, nameOp, sig, i)
		if nameErr != nil {
			return nil, nameErr
		}
		th, tErr := callIndexedHandle(tensorFn, tensorOp, sig, i)
		if tErr != nil {
			return nil, tErr
		}
		sp, spErr := readTensorSpec(syms, th, name)
		if spErr != nil {
			return nil, spErr
		}
		specs = append(specs, sp)
	}
	return specs, nil
}

func readSubgraphTensors(syms *capi.Symbols, sg uintptr, isInput bool) ([]TensorSpec, error) {
	countFn, countOp := syms.GetNumSubgraphOutputs, "LiteRtGetNumSubgraphOutputs"
	tensorFn, tensorOp := syms.GetSubgraphOutput, "LiteRtGetSubgraphOutput"
	if isInput {
		countFn, countOp = syms.GetNumSubgraphInputs, "LiteRtGetNumSubgraphInputs"
		tensorFn, tensorOp = syms.GetSubgraphInput, "LiteRtGetSubgraphInput"
	}

	n, err := callCount(countFn, countOp, sg)
	if err != nil {
		return nil, err
	}
	specs := make([]TensorSpec, 0, n)
	for i := range n {
		th, tErr := callIndexedHandle(tensorFn, tensorOp, sg, i)
		if tErr != nil {
			return nil, tErr
		}
		name, nameErr := tensorName(syms, th)
		if nameErr != nil {
			return nil, nameErr
		}
		sp, spErr := readTensorSpec(syms, th, name)
		if spErr != nil {
			return nil, spErr
		}
		specs = append(specs, sp)
	}
	return specs, nil
}

func callCount(fn *capi.LazyFun, op string, handle uintptr) (int, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := handle
	var n uint64
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &n))
	if err := statusErr(op, st, callErr); err != nil {
		return 0, err
	}
	return int(n), nil
}

func callIndexedCString(fn *capi.LazyFun, op string, handle uintptr, index int) (string, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := handle
	idx := uint64(index)
	var out unsafe.Pointer
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &idx), capi.Out(&pin, &out))
	if err := statusErr(op, st, callErr); err != nil {
		return "", err
	}
	return capi.GoString(out), nil
}

func callIndexedHandle(fn *capi.LazyFun, op string, handle uintptr, index int) (uintptr, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := handle
	idx := uint64(index)
	var out uintptr
	st, callErr := fn.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &idx), capi.Out(&pin, &out))
	if err := statusErr(op, st, callErr); err != nil {
		return 0, err
	}
	return out, nil
}

func readTensorSpec(syms *capi.Symbols, tensorHandle uintptr, name string) (TensorSpec, error) {
	tt, err := readTensorType(syms, tensorHandle)
	if err != nil {
		return TensorSpec{}, err
	}
	q, err := readQuantization(syms, tensorHandle)
	if err != nil {
		return TensorSpec{}, err
	}
	return TensorSpec{
		Name:         name,
		Type:         tt,
		Quantization: q,
	}, nil
}

func readTensorType(syms *capi.Symbols, tensorHandle uintptr) (TensorType, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := tensorHandle
	var typeID int32
	st, callErr := syms.GetTensorTypeId.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &typeID))
	if err := statusErr("LiteRtGetTensorTypeId", st, callErr); err != nil {
		return TensorType{}, err
	}

	if typeID == 1 { // kLiteRtUnrankedTensorType
		var elem int32
		st, callErr = syms.GetUnrankedTensorType.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &elem))
		if err := statusErr("LiteRtGetUnrankedTensorType", st, callErr); err != nil {
			return TensorType{}, err
		}
		return TensorType{ElementType: ElementType(elem), Unranked: true}, nil
	}

	raw := make([]byte, capi.RankedTensorTypeSize)
	pin.Pin(&raw[0])
	rawPtr := unsafe.Pointer(&raw[0])
	st, callErr = syms.GetRankedTensorType.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &rawPtr))
	if err := statusErr("LiteRtGetRankedTensorType", st, callErr); err != nil {
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

func readQuantization(syms *capi.Symbols, tensorHandle uintptr) (Quantization, error) {
	var pin runtime.Pinner
	defer pin.Unpin()

	h := tensorHandle
	var qType int32
	st, callErr := syms.GetQuantizationTypeId.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &qType))
	if err := statusErr("LiteRtGetQuantizationTypeId", st, callErr); err != nil {
		return Quantization{}, err
	}

	switch QuantizationType(qType) {
	case QuantNone:
		return Quantization{Type: QuantNone}, nil
	case QuantPerTensor:
		raw := make([]byte, capi.QuantizationPerTensorSize)
		pin.Pin(&raw[0])
		rawPtr := unsafe.Pointer(&raw[0])
		st, callErr = syms.GetPerTensorQuantization.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Ptr(&pin, &rawPtr))
		if err := statusErr("LiteRtGetPerTensorQuantization", st, callErr); err != nil {
			return Quantization{}, err
		}
		scale, zp, decErr := capi.DecodePerTensorQuantization(raw)
		if decErr != nil {
			return Quantization{}, &Error{Op: "DecodePerTensorQuantization", Sentinel: ErrInvalidArgument, Cause: decErr}
		}
		return Quantization{
			Type:      QuantPerTensor,
			Scale:     scale,
			ZeroPoint: zp,
		}, nil
	case QuantPerChannel:
		var raw capi.RawPerChannelQuantization
		st, callErr = syms.GetPerChannelQuantization.CallStatus(&pin, capi.Ptr(&pin, &h), capi.Out(&pin, &raw))
		if err := statusErr("LiteRtGetPerChannelQuantization", st, callErr); err != nil {
			return Quantization{}, err
		}
		dim, scales, zps := capi.CopyPerChannelQuantization(raw)
		return Quantization{
			Type:         QuantPerChannel,
			QuantizedDim: dim,
			Scales:       scales,
			ZeroPoints:   zps,
		}, nil
	default:
		return Quantization{Type: QuantizationType(qType)}, nil
	}
}
