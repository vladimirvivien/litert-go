package litert

import (
	"fmt"
	"unsafe"
)

// Element constrains scalar tensor element types that support typed slice I/O.
type Element interface {
	float32 | float64 | int8 | uint8 | int16 | uint16 | int32 | uint32 | int64 | uint64 | bool
}

func elementTypeOf[T Element]() ElementType {
	switch any((*T)(nil)).(type) {
	case *float32:
		return ElementFloat32
	case *float64:
		return ElementFloat64
	case *int8:
		return ElementInt8
	case *uint8:
		return ElementUInt8
	case *int16:
		return ElementInt16
	case *uint16:
		return ElementUInt16
	case *int32:
		return ElementInt32
	case *uint32:
		return ElementUInt32
	case *int64:
		return ElementInt64
	case *uint64:
		return ElementUInt64
	case *bool:
		return ElementBool
	default:
		return ElementNone
	}
}

func checkSliceBuffer[T Element](op string, buf *TensorBuffer) (uint64, ElementType, error) {
	if !buf.valid() {
		return 0, ElementNone, ErrClosed
	}
	tt, err := buf.TensorType()
	if err != nil {
		return 0, ElementNone, err
	}
	want := elementTypeOf[T]()
	if tt.ElementType != want {
		return 0, want, &Error{
			Op:       op,
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("tensor element type %s does not match Go slice element type %s", tt.ElementType, want),
		}
	}
	packed, err := buf.PackedSize()
	if err != nil {
		return 0, want, err
	}
	return packed, want, nil
}

// WriteSlice copies src into buf after validating that T matches the buffer's
// element type and fits within the packed buffer size.
func WriteSlice[T Element](buf *TensorBuffer, src []T) error {
	if !buf.valid() {
		return ErrClosed
	}
	if len(src) == 0 {
		return nil
	}
	packed, _, err := checkSliceBuffer[T]("WriteSlice", buf)
	if err != nil {
		return err
	}
	byteLen := uint64(len(src)) * uint64(unsafe.Sizeof(src[0]))
	if byteLen > packed {
		return &Error{
			Op:       "WriteSlice",
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("input byte length %d exceeds buffer packed size %d", byteLen, packed),
		}
	}
	addr, err := buf.Lock(LockWrite)
	if err != nil {
		return err
	}
	n := int(byteLen)
	copy(unsafe.Slice((*byte)(addr), n), unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), n))
	return buf.Unlock()
}

// ReadSliceInto copies the packed contents of buf into dst after validating
// that T matches the buffer's element type and fits within dst.
func ReadSliceInto[T Element](buf *TensorBuffer, dst []T) error {
	if !buf.valid() {
		return ErrClosed
	}
	if len(dst) == 0 {
		return nil
	}
	packed, _, err := checkSliceBuffer[T]("ReadSliceInto", buf)
	if err != nil {
		return err
	}
	byteLen := uint64(len(dst)) * uint64(unsafe.Sizeof(dst[0]))
	if byteLen > packed {
		return &Error{
			Op:       "ReadSliceInto",
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("destination byte length %d exceeds buffer packed size %d", byteLen, packed),
		}
	}
	addr, err := buf.Lock(LockRead)
	if err != nil {
		return err
	}
	n := int(byteLen)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), n), unsafe.Slice((*byte)(addr), n))
	return buf.Unlock()
}

// ReadSlice reads all packed elements from buf into a newly allocated typed Go slice.
func ReadSlice[T Element](buf *TensorBuffer) ([]T, error) {
	packed, _, err := checkSliceBuffer[T]("ReadSlice", buf)
	if err != nil {
		return nil, err
	}
	var zero T
	elemSize := uint64(unsafe.Sizeof(zero))
	if elemSize == 0 || packed%elemSize != 0 {
		return nil, &Error{
			Op:       "ReadSlice",
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("packed size %d is not a multiple of element size %d", packed, elemSize),
		}
	}
	count := int(packed / elemSize)
	dst := make([]T, count)
	if count == 0 {
		return dst, nil
	}
	addr, err := buf.Lock(LockRead)
	if err != nil {
		return nil, err
	}
	n := int(packed)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), n), unsafe.Slice((*byte)(addr), n))
	if uErr := buf.Unlock(); uErr != nil {
		return nil, uErr
	}
	return dst, nil
}
