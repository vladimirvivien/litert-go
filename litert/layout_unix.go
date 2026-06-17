//go:build !windows

package litert

// The GCC/Clang builds of libLiteRt pack the LiteRtLayout rank:7/has_strides:1
// bitfields into one 32-bit word, so the struct layout is: element_type[0:4],
// packed rank/has_strides[4:8], dimensions[8:40], strides[40:72] = 72 bytes.
const (
	rankedTensorTypeSize = 72
	dimsOffset           = 8
)
