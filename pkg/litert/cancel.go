package litert

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/vladimirvivien/litert-go/internal/capi"
)

type cancelSlot struct {
	cancelled atomic.Bool
}

var (
	cancelCBOnce sync.Once
	cancelCBAddr uintptr
)

func getCancelCallback() uintptr {
	cancelCBOnce.Do(func() {
		cancelCBAddr = purego.NewCallback(func(data unsafe.Pointer) uintptr {
			if data == nil {
				return 0
			}
			if (*cancelSlot)(data).cancelled.Load() {
				return 1
			}
			return 0
		})
	})
	return cancelCBAddr
}

func (cm *CompiledModel) ensureCancelRegistered() error {
	if cm.cancelRegistered {
		return nil
	}
	cm.pin.Pin(&cm.cancel)
	var callPin runtime.Pinner
	defer callPin.Unpin()

	h := cm.handle
	dataPtr := unsafe.Pointer(&cm.cancel)
	cb := getCancelCallback()
	st, callErr := cm.syms.SetCompiledModelCancellationFunction.CallStatus(
		&callPin,
		capi.Ptr(&callPin, &h),
		capi.Ptr(&callPin, &dataPtr),
		capi.Ptr(&callPin, &cb),
	)
	if err := statusErr("LiteRtSetCompiledModelCancellationFunction", st, callErr); err != nil {
		return err
	}
	cm.cancelRegistered = true
	return nil
}

// armCancel resets the cancellation flag and wires ctx cancellation if
// ctx.Done() is non-nil. Calling armCancel with context.Background() performs
// zero allocations.
func (cm *CompiledModel) armCancel(ctx context.Context) (stop func() bool, err error) {
	cm.cancel.cancelled.Store(false)
	if ctx == nil {
		return nil, nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	if ctx.Done() == nil {
		return nil, nil
	}
	if regErr := cm.ensureCancelRegistered(); regErr != nil {
		return nil, regErr
	}
	stop = context.AfterFunc(ctx, func() {
		cm.cancel.cancelled.Store(true)
	})
	return stop, nil
}
