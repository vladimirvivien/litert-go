package litert

import (
	"context"
	"errors"
	"testing"
)

func TestStatusStringsAndSentinels(t *testing.T) {
	if !StatusOk.Ok() {
		t.Fatal("StatusOk should be Ok()")
	}
	if StatusErrorInvalidArgument.Ok() {
		t.Fatal("StatusErrorInvalidArgument should not be Ok()")
	}

	tests := []struct {
		status   Status
		wantName string
		sentinel error
	}{
		{StatusOk, "kLiteRtStatusOk", nil},
		{StatusErrorInvalidArgument, "kLiteRtStatusErrorInvalidArgument", ErrInvalidArgument},
		{StatusErrorMemoryAllocationFailure, "kLiteRtStatusErrorMemoryAllocationFailure", ErrMemoryAllocation},
		{StatusErrorRuntimeFailure, "kLiteRtStatusErrorRuntimeFailure", ErrRuntimeFailure},
		{StatusErrorMissingInputTensor, "kLiteRtStatusErrorMissingInputTensor", ErrMissingInputTensor},
		{StatusErrorUnsupported, "kLiteRtStatusErrorUnsupported", ErrUnsupported},
		{StatusErrorNotFound, "kLiteRtStatusErrorNotFound", ErrNotFound},
		{StatusErrorTimeoutExpired, "kLiteRtStatusErrorTimeoutExpired", ErrTimeout},
		{StatusErrorWrongVersion, "kLiteRtStatusErrorWrongVersion", ErrWrongVersion},
		{StatusErrorAlreadyExists, "kLiteRtStatusErrorAlreadyExists", ErrAlreadyExists},
		{StatusErrorFailedPrecondition, "kLiteRtStatusErrorFailedPrecondition", ErrFailedPrecondition},
		{StatusCancelled, "kLiteRtStatusCancelled", ErrCancelled},
		{StatusErrorFileIO, "kLiteRtStatusErrorFileIO", ErrFileIO},
		{StatusErrorInvalidFlatbuffer, "kLiteRtStatusErrorInvalidFlatbuffer", ErrInvalidFlatbuffer},
		{StatusErrorDynamicLoading, "kLiteRtStatusErrorDynamicLoading", ErrDynamicLoading},
		{StatusErrorSerialization, "kLiteRtStatusErrorSerialization", ErrSerialization},
		{StatusErrorCompilation, "kLiteRtStatusErrorCompilation", ErrCompilation},
		{StatusErrorIndexOOB, "kLiteRtStatusErrorIndexOOB", ErrIndexOOB},
		{StatusErrorShapeInferenceFailed, "kLiteRtStatusErrorShapeInferenceFailed", ErrShapeInference},
	}

	for _, tc := range tests {
		if got := tc.status.String(); got != tc.wantName {
			t.Errorf("Status(%d).String() = %q, want %q", tc.status, got, tc.wantName)
		}
		err := statusErr("TestOp", int32(tc.status), nil)
		if tc.status.Ok() {
			if err != nil {
				t.Fatalf("expected nil error for StatusOk, got %v", err)
			}
			continue
		}
		if err == nil {
			t.Fatalf("expected error for status %d", tc.status)
		}
		if tc.sentinel != nil && !errors.Is(err, tc.sentinel) {
			t.Errorf("errors.Is(%v, %v) = false", err, tc.sentinel)
		}
		if _, ok := errors.AsType[*Error](err); !ok {
			t.Errorf("errors.As failed for %v", err)
		}
	}

	cancelErr := statusErr("Run", int32(StatusCancelled), nil)
	if !errors.Is(cancelErr, context.Canceled) {
		t.Fatal("StatusCancelled should unwrap to context.Canceled")
	}

	cause := errors.New("dlsym failed")
	symErr := statusErr("Run", 0, cause)
	if !errors.Is(symErr, ErrMissingSymbol) || !errors.Is(symErr, cause) {
		t.Fatalf("missing symbol error unwrap mismatch: %v", symErr)
	}
}
