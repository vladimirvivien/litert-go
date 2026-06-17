//go:build windows

package litert

// The MSVC build of libLiteRt does not pack the LiteRtLayout rank:7/has_strides:1
// bitfields into one word, so the struct is: element_type[0:4], rank[4:8],
// has_strides[8:12], dimensions[12:44], strides[44:76] = 76 bytes. 80 is a safe
// over-allocation.
const (
	rankedTensorTypeSize = 80
	dimsOffset           = 12
)
