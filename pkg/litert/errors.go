// Package litert provides Go bindings for the LiteRT 2.4.1 CompiledModel C API
// without CGO, targeting general-purpose .tflite inference.
package litert

import (
	"context"
	"errors"
	"fmt"
)

// Status represents a LiteRtStatus code returned by the LiteRT C runtime.
type Status int32

// Status codes defined in litert/c/litert_common.h (2.4.1).
const (
	StatusOk                             Status = 0
	StatusErrorInvalidArgument           Status = 1
	StatusErrorMemoryAllocationFailure   Status = 2
	StatusErrorRuntimeFailure            Status = 3
	StatusErrorMissingInputTensor        Status = 4
	StatusErrorUnsupported               Status = 5
	StatusErrorNotFound                  Status = 6
	StatusErrorTimeoutExpired            Status = 7
	StatusErrorWrongVersion              Status = 8
	StatusErrorUnknown                   Status = 9
	StatusErrorAlreadyExists             Status = 10
	StatusErrorFailedPrecondition        Status = 11
	StatusCancelled                      Status = 100
	StatusErrorFileIO                    Status = 500
	StatusErrorInvalidFlatbuffer         Status = 501
	StatusErrorDynamicLoading            Status = 502
	StatusErrorSerialization             Status = 503
	StatusErrorCompilation               Status = 504
	StatusErrorIndexOOB                  Status = 1000
	StatusErrorInvalidIrType             Status = 1001
	StatusErrorInvalidGraphInvariant     Status = 1002
	StatusErrorGraphModification         Status = 1003
	StatusErrorInvalidToolConfig         Status = 1500
	StatusLegalizeNoMatch                Status = 2000
	StatusErrorInvalidLegalization       Status = 2001
	StatusPatternNoMatch                 Status = 3000
	StatusInvalidTransformation          Status = 3001
	StatusErrorUnsupportedRuntimeVersion Status = 4000
	StatusErrorUnsupportedCompiler       Status = 4001
	StatusErrorIncompatibleByteCode      Status = 4002
	StatusErrorUnsupportedOpShapeInferer Status = 5000
	StatusErrorShapeInferenceFailed      Status = 5001
)

// Ok reports whether s indicates success.
func (s Status) Ok() bool { return s == StatusOk }

// String returns the canonical status identifier.
func (s Status) String() string {
	switch s {
	case StatusOk:
		return "kLiteRtStatusOk"
	case StatusErrorInvalidArgument:
		return "kLiteRtStatusErrorInvalidArgument"
	case StatusErrorMemoryAllocationFailure:
		return "kLiteRtStatusErrorMemoryAllocationFailure"
	case StatusErrorRuntimeFailure:
		return "kLiteRtStatusErrorRuntimeFailure"
	case StatusErrorMissingInputTensor:
		return "kLiteRtStatusErrorMissingInputTensor"
	case StatusErrorUnsupported:
		return "kLiteRtStatusErrorUnsupported"
	case StatusErrorNotFound:
		return "kLiteRtStatusErrorNotFound"
	case StatusErrorTimeoutExpired:
		return "kLiteRtStatusErrorTimeoutExpired"
	case StatusErrorWrongVersion:
		return "kLiteRtStatusErrorWrongVersion"
	case StatusErrorUnknown:
		return "kLiteRtStatusErrorUnknown"
	case StatusErrorAlreadyExists:
		return "kLiteRtStatusErrorAlreadyExists"
	case StatusErrorFailedPrecondition:
		return "kLiteRtStatusErrorFailedPrecondition"
	case StatusCancelled:
		return "kLiteRtStatusCancelled"
	case StatusErrorFileIO:
		return "kLiteRtStatusErrorFileIO"
	case StatusErrorInvalidFlatbuffer:
		return "kLiteRtStatusErrorInvalidFlatbuffer"
	case StatusErrorDynamicLoading:
		return "kLiteRtStatusErrorDynamicLoading"
	case StatusErrorSerialization:
		return "kLiteRtStatusErrorSerialization"
	case StatusErrorCompilation:
		return "kLiteRtStatusErrorCompilation"
	case StatusErrorIndexOOB:
		return "kLiteRtStatusErrorIndexOOB"
	case StatusErrorInvalidIrType:
		return "kLiteRtStatusErrorInvalidIrType"
	case StatusErrorInvalidGraphInvariant:
		return "kLiteRtStatusErrorInvalidGraphInvariant"
	case StatusErrorGraphModification:
		return "kLiteRtStatusErrorGraphModification"
	case StatusErrorInvalidToolConfig:
		return "kLiteRtStatusErrorInvalidToolConfig"
	case StatusLegalizeNoMatch:
		return "kLiteRtStatusLegalizeNoMatch"
	case StatusErrorInvalidLegalization:
		return "kLiteRtStatusErrorInvalidLegalization"
	case StatusPatternNoMatch:
		return "kLiteRtStatusPatternNoMatch"
	case StatusInvalidTransformation:
		return "kLiteRtStatusInvalidTransformation"
	case StatusErrorUnsupportedRuntimeVersion:
		return "kLiteRtStatusErrorUnsupportedRuntimeVersion"
	case StatusErrorUnsupportedCompiler:
		return "kLiteRtStatusErrorUnsupportedCompilerVersion"
	case StatusErrorIncompatibleByteCode:
		return "kLiteRtStatusErrorIncompatibleByteCodeVersion"
	case StatusErrorUnsupportedOpShapeInferer:
		return "kLiteRtStatusErrorUnsupportedOpShapeInferer"
	case StatusErrorShapeInferenceFailed:
		return "kLiteRtStatusErrorShapeInferenceFailed"
	default:
		return fmt.Sprintf("LiteRtStatus(%d)", int32(s))
	}
}

// Sentinel errors matched via errors.Is.
var (
	ErrClosed             = errors.New("litert: handle is closed or uninitialized")
	ErrInvalidArgument    = errors.New("litert: invalid argument")
	ErrMemoryAllocation   = errors.New("litert: memory allocation failure")
	ErrRuntimeFailure     = errors.New("litert: runtime failure")
	ErrMissingInputTensor = errors.New("litert: missing input tensor")
	ErrUnsupported        = errors.New("litert: unsupported operation")
	ErrNotFound           = errors.New("litert: not found")
	ErrTimeout            = errors.New("litert: timeout expired")
	ErrWrongVersion       = errors.New("litert: wrong version")
	ErrAlreadyExists      = errors.New("litert: already exists")
	ErrFailedPrecondition = errors.New("litert: failed precondition")
	ErrCancelled          = errors.New("litert: cancelled")
	ErrFileIO             = errors.New("litert: file I/O error")
	ErrInvalidFlatbuffer  = errors.New("litert: invalid flatbuffer")
	ErrDynamicLoading     = errors.New("litert: dynamic loading error")
	ErrSerialization      = errors.New("litert: serialization error")
	ErrCompilation        = errors.New("litert: compilation error")
	ErrIndexOOB           = errors.New("litert: index out of bounds")
	ErrShapeInference     = errors.New("litert: shape inference failed")
	ErrMissingSymbol      = errors.New("litert: missing C API symbol")
)

func sentinelForStatus(s Status) error {
	switch s {
	case StatusErrorInvalidArgument:
		return ErrInvalidArgument
	case StatusErrorMemoryAllocationFailure:
		return ErrMemoryAllocation
	case StatusErrorRuntimeFailure:
		return ErrRuntimeFailure
	case StatusErrorMissingInputTensor:
		return ErrMissingInputTensor
	case StatusErrorUnsupported:
		return ErrUnsupported
	case StatusErrorNotFound:
		return ErrNotFound
	case StatusErrorTimeoutExpired:
		return ErrTimeout
	case StatusErrorWrongVersion, StatusErrorUnsupportedRuntimeVersion,
		StatusErrorUnsupportedCompiler, StatusErrorIncompatibleByteCode:
		return ErrWrongVersion
	case StatusErrorAlreadyExists:
		return ErrAlreadyExists
	case StatusErrorFailedPrecondition:
		return ErrFailedPrecondition
	case StatusCancelled:
		return ErrCancelled
	case StatusErrorFileIO:
		return ErrFileIO
	case StatusErrorInvalidFlatbuffer:
		return ErrInvalidFlatbuffer
	case StatusErrorDynamicLoading:
		return ErrDynamicLoading
	case StatusErrorSerialization:
		return ErrSerialization
	case StatusErrorCompilation:
		return ErrCompilation
	case StatusErrorIndexOOB:
		return ErrIndexOOB
	case StatusErrorUnsupportedOpShapeInferer, StatusErrorShapeInferenceFailed:
		return ErrShapeInference
	default:
		return nil
	}
}

// Error describes a failure returned from a LiteRT operation.
type Error struct {
	Op       string
	Status   Status
	Msg      string
	Sentinel error
	Cause    error
}

// Error formats the operation, status, and contextual details.
func (e *Error) Error() string {
	switch {
	case e.Msg != "" && e.Cause != nil:
		return fmt.Sprintf("litert: %s: %s: %v", e.Op, e.Msg, e.Cause)
	case e.Msg != "":
		return fmt.Sprintf("litert: %s: %s", e.Op, e.Msg)
	case e.Cause != nil:
		return fmt.Sprintf("litert: %s: %v", e.Op, e.Cause)
	case e.Status != StatusOk:
		return fmt.Sprintf("litert: %s failed: %s (%d)", e.Op, e.Status.String(), int32(e.Status))
	default:
		return fmt.Sprintf("litert: %s failed", e.Op)
	}
}

// Unwrap exposes the matching sentinel error and any underlying cause.
func (e *Error) Unwrap() []error {
	var errs []error
	if e.Sentinel != nil {
		errs = append(errs, e.Sentinel)
	}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

func statusErr(op string, raw int32, callErr error) error {
	if callErr != nil {
		return &Error{
			Op:       op,
			Sentinel: ErrMissingSymbol,
			Cause:    callErr,
		}
	}
	st := Status(raw)
	if st.Ok() {
		return nil
	}
	var cause error
	if st == StatusCancelled {
		cause = context.Canceled
	}
	return &Error{
		Op:       op,
		Status:   st,
		Sentinel: sentinelForStatus(st),
		Cause:    cause,
	}
}
