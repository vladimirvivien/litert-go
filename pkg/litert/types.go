package litert

import "fmt"

// Accelerator selects target hardware execution delegates (LiteRtHwAccelerators).
type Accelerator int32

const (
	AccelNone Accelerator = 0
	AccelCPU  Accelerator = 1 << 0
	AccelGPU  Accelerator = 1 << 1
	AccelNPU  Accelerator = 1 << 2
)

// BufferType identifies a LiteRtTensorBuffer backing storage kind.
type BufferType int32

const (
	BufferUnknown           BufferType = 0
	BufferHostMemory        BufferType = 1
	BufferAHWB              BufferType = 2
	BufferION               BufferType = 3
	BufferDMABuf            BufferType = 4
	BufferFastRPC           BufferType = 5
	BufferGLBuffer          BufferType = 6
	BufferGLTexture         BufferType = 7
	BufferOpenCLBuffer      BufferType = 10
	BufferOpenCLBufferFP16  BufferType = 11
	BufferOpenCLTexture     BufferType = 12
	BufferOpenCLTextureFP16 BufferType = 13
	BufferOpenCLPacked      BufferType = 14
	BufferOpenCLImage       BufferType = 15
	BufferOpenCLImageFP16   BufferType = 16
	BufferWebGPUBuffer      BufferType = 20
	BufferWebGPUBufferFP16  BufferType = 21
	BufferWebGPUTexture     BufferType = 22
	BufferWebGPUTextureFP16 BufferType = 23
	BufferWebGPUImage       BufferType = 24
	BufferWebGPUImageFP16   BufferType = 25
	BufferWebGPUPacked      BufferType = 26
	BufferMetalBuffer       BufferType = 30
	BufferMetalBufferFP16   BufferType = 31
	BufferMetalTexture      BufferType = 32
	BufferMetalTextureFP16  BufferType = 33
	BufferMetalPacked       BufferType = 34
	BufferVulkanBuffer      BufferType = 40
	BufferVulkanBufferFP16  BufferType = 41
	BufferVulkanTexture     BufferType = 42
	BufferVulkanTextureFP16 BufferType = 43
	BufferVulkanImage       BufferType = 44
	BufferVulkanImageFP16   BufferType = 45
	BufferVulkanPacked      BufferType = 46
)

// LockMode controls host mapping access for TensorBuffer.Lock.
type LockMode int32

const (
	LockRead      LockMode = 0
	LockWrite     LockMode = 1
	LockReadWrite LockMode = 2
)

// LogSeverity selects the minimum runtime log threshold.
type LogSeverity int64

const (
	LogVerbose LogSeverity = 0
	LogInfo    LogSeverity = 1
	LogWarning LogSeverity = 2
	LogError   LogSeverity = 3
	LogFatal   LogSeverity = 4
	LogNone    LogSeverity = 5
)

// ElementType represents a LiteRtElementType tensor scalar dtype.
type ElementType int32

const (
	ElementNone         ElementType = 0
	ElementFloat32      ElementType = 1
	ElementInt32        ElementType = 2
	ElementUInt8        ElementType = 3
	ElementInt64        ElementType = 4
	ElementString       ElementType = 5
	ElementBool         ElementType = 6
	ElementInt16        ElementType = 7
	ElementComplex64    ElementType = 8
	ElementInt8         ElementType = 9
	ElementFloat16      ElementType = 10
	ElementFloat64      ElementType = 11
	ElementComplex128   ElementType = 12
	ElementUInt64       ElementType = 13
	ElementResource     ElementType = 14
	ElementVariant      ElementType = 15
	ElementUInt32       ElementType = 16
	ElementUInt16       ElementType = 17
	ElementInt4         ElementType = 18
	ElementBFloat16     ElementType = 19
	ElementInt2         ElementType = 20
	ElementUInt4        ElementType = 21
	ElementFloat8E4M3FN ElementType = 22
	ElementFloat8E5M2   ElementType = 23
)

// String returns a concise name for the element type.
func (e ElementType) String() string {
	switch e {
	case ElementNone:
		return "none"
	case ElementFloat32:
		return "f32"
	case ElementInt32:
		return "i32"
	case ElementUInt8:
		return "u8"
	case ElementInt64:
		return "i64"
	case ElementString:
		return "string"
	case ElementBool:
		return "bool"
	case ElementInt16:
		return "i16"
	case ElementComplex64:
		return "c64"
	case ElementInt8:
		return "i8"
	case ElementFloat16:
		return "f16"
	case ElementFloat64:
		return "f64"
	case ElementComplex128:
		return "c128"
	case ElementUInt64:
		return "u64"
	case ElementResource:
		return "resource"
	case ElementVariant:
		return "variant"
	case ElementUInt32:
		return "u32"
	case ElementUInt16:
		return "u16"
	case ElementInt4:
		return "i4"
	case ElementBFloat16:
		return "bf16"
	case ElementInt2:
		return "i2"
	case ElementUInt4:
		return "u4"
	case ElementFloat8E4M3FN:
		return "f8e4m3fn"
	case ElementFloat8E5M2:
		return "f8e5m2"
	default:
		return fmt.Sprintf("elem(%d)", int32(e))
	}
}

// ByteSize returns the per-element storage size in bytes for byte-aligned dtypes.
// Sub-byte packed types (Int2, Int4, UInt4) and variable-length types report false.
func (e ElementType) ByteSize() (int, bool) {
	switch e {
	case ElementBool, ElementInt8, ElementUInt8, ElementFloat8E4M3FN, ElementFloat8E5M2:
		return 1, true
	case ElementInt16, ElementUInt16, ElementFloat16, ElementBFloat16:
		return 2, true
	case ElementInt32, ElementUInt32, ElementFloat32:
		return 4, true
	case ElementInt64, ElementUInt64, ElementFloat64, ElementComplex64:
		return 8, true
	case ElementComplex128:
		return 16, true
	default:
		return 0, false
	}
}

// TensorType describes an element type, shape, and optional explicit strides.
type TensorType struct {
	ElementType ElementType
	Shape       []int32
	HasStrides  bool
	Strides     []uint32
	Unranked    bool
}

// IsDynamic reports whether the tensor is unranked or contains any dimension < 0.
func (tt TensorType) IsDynamic() bool {
	if tt.Unranked {
		return true
	}
	for _, d := range tt.Shape {
		if d < 0 {
			return true
		}
	}
	return false
}

// NumElements computes the total scalar element count when all dimensions are static (>= 0).
func (tt TensorType) NumElements() (int, bool) {
	if tt.Unranked {
		return 0, false
	}
	n := 1
	for _, d := range tt.Shape {
		if d < 0 {
			return 0, false
		}
		if d == 0 {
			return 0, true
		}
		if n > (1<<62)/int(d) {
			return 0, false
		}
		n *= int(d)
	}
	return n, true
}

// ByteSize computes the packed payload size in bytes for statically shaped,
// byte-aligned element types.
func (tt TensorType) ByteSize() (uint64, bool) {
	n, ok := tt.NumElements()
	if !ok {
		return 0, false
	}
	elemBytes, ok := tt.ElementType.ByteSize()
	if !ok {
		return 0, false
	}
	return uint64(n) * uint64(elemBytes), true
}

// QuantizationType identifies a LiteRtQuantizationTypeId scheme.
type QuantizationType int32

const (
	QuantNone       QuantizationType = 0
	QuantPerTensor  QuantizationType = 1
	QuantPerChannel QuantizationType = 2
	QuantBlockWise  QuantizationType = 3
)

// Quantization holds per-tensor or per-channel affine quantization parameters.
type Quantization struct {
	Type         QuantizationType
	Scale        float32
	ZeroPoint    int64
	QuantizedDim int32
	Scales       []float32
	ZeroPoints   []int64
}

// TensorSpec describes a signature or subgraph input/output tensor.
type TensorSpec struct {
	Name         string
	Type         TensorType
	Quantization Quantization
}

// BufferRequirements describes the size, alignment, and supported storage types
// required by a compiled model input or output tensor.
type BufferRequirements struct {
	BufferSize     uint64
	Alignment      uint64
	SupportedTypes []BufferType
}

// AcceleratorDelegation summarizes nodes and partitions delegated to one accelerator.
type AcceleratorDelegation struct {
	Name               string
	HardwareType       Accelerator
	DelegatedNodeCount int
	PartitionCount     int
}

// DelegationMetrics describes operation delegation across a compiled model.
type DelegationMetrics struct {
	TotalNodeCount int
	Accelerators   []AcceleratorDelegation
}
