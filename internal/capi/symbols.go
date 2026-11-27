package capi

import "github.com/jupiterrider/ffi"

// Symbols holds lazily resolved LiteRT 2.4.1 C API entry points.
type Symbols struct {
	// Runtime
	GetRuntimeAbiVersion *LazyFun

	// Environment
	CreateEnvironment            *LazyFun
	AddEnvironmentOptions        *LazyFun
	EnvironmentSupportsFP16      *LazyFun
	EnvironmentHasGpuEnvironment *LazyFun
	DestroyEnvironment           *LazyFun

	// Options & Opaque options
	CreateOptions                  *LazyFun
	DestroyOptions                 *LazyFun
	SetOptionsHardwareAccelerators *LazyFun
	GetOptionsHardwareAccelerators *LazyFun
	AddOpaqueOptions               *LazyFun
	CreateOpaqueOptions            *LazyFun
	DestroyOpaqueOptions           *LazyFun

	// Model, Signature, Subgraph, Tensor, Metadata
	CreateModelFromFile             *LazyFun
	CreateModelFromBuffer           *LazyFun
	DestroyModel                    *LazyFun
	GetModelMetadata                *LazyFun
	GetNumModelSignatures           *LazyFun
	GetModelSignature               *LazyFun
	GetMainModelSubgraphIndex       *LazyFun
	GetNumModelSubgraphs            *LazyFun
	GetModelSubgraph                *LazyFun
	GetSubgraphName                 *LazyFun
	GetNumSubgraphInputs            *LazyFun
	GetSubgraphInput                *LazyFun
	GetNumSubgraphOutputs           *LazyFun
	GetSubgraphOutput               *LazyFun
	GetSignatureKey                 *LazyFun
	GetNumSignatureInputs           *LazyFun
	GetSignatureInputName           *LazyFun
	GetSignatureInputTensor         *LazyFun
	GetSignatureInputTensorByIndex  *LazyFun
	GetNumSignatureOutputs          *LazyFun
	GetSignatureOutputName          *LazyFun
	GetSignatureOutputTensor        *LazyFun
	GetSignatureOutputTensorByIndex *LazyFun
	GetTensorName                   *LazyFun
	GetTensorTypeId                 *LazyFun
	GetRankedTensorType             *LazyFun
	GetUnrankedTensorType           *LazyFun
	GetQuantizationTypeId           *LazyFun
	GetPerTensorQuantization        *LazyFun
	GetPerChannelQuantization       *LazyFun

	// CompiledModel & Diagnostics & DelegationMetrics
	CreateCompiledModel                           *LazyFun
	DestroyCompiledModel                          *LazyFun
	GetCompiledModelInputBufferRequirements       *LazyFun
	GetCompiledModelOutputBufferRequirements      *LazyFun
	GetCompiledModelInputTensorLayout             *LazyFun
	GetCompiledModelOutputTensorLayouts           *LazyFun
	CompiledModelResizeInputTensor                *LazyFun
	CompiledModelResizeInputTensorNonStrict       *LazyFun
	RunCompiledModel                              *LazyFun
	RunCompiledModelAsync                         *LazyFun
	SetCompiledModelCancellationFunction          *LazyFun
	GetCompiledModelLastInferenceDuration         *LazyFun
	CompiledModelIsFullyAccelerated               *LazyFun
	CompiledModelIsNonCpuFullyAccelerated         *LazyFun
	CompiledModelGetDelegationMetrics             *LazyFun
	GetDelegationMetricsTotalNodeCount            *LazyFun
	GetNumDelegationMetricsAccelerators           *LazyFun
	GetDelegationMetricsAccelerator               *LazyFun
	GetAcceleratorDelegationMetricsName           *LazyFun
	GetAcceleratorDelegationMetricsHardwareType   *LazyFun
	GetAcceleratorDelegationMetricsDelegatedCount *LazyFun
	GetAcceleratorDelegationMetricsPartitionCount *LazyFun
	CompiledModelGetProfiler                      *LazyFun
	CompiledModelClearErrors                      *LazyFun
	CompiledModelGetErrorMessages                 *LazyFun

	// Profiler
	StartProfiler        *LazyFun
	StopProfiler         *LazyFun
	ResetProfiler        *LazyFun
	GetNumProfilerEvents *LazyFun
	GetProfileSummary    *LazyFun

	// TensorBufferRequirements, TensorBuffer, Event
	GetNumTensorBufferRequirementsSupportedTypes *LazyFun
	GetTensorBufferRequirementsSupportedType     *LazyFun
	GetTensorBufferRequirementsBufferSize        *LazyFun
	GetTensorBufferRequirementsAlignment         *LazyFun
	CreateManagedTensorBuffer                    *LazyFun
	CreateManagedTensorBufferFromRequirements    *LazyFun
	CreateTensorBufferFromHostMemory             *LazyFun
	DuplicateTensorBuffer                        *LazyFun
	GetTensorBufferType                          *LazyFun
	GetTensorBufferTensorType                    *LazyFun
	GetTensorBufferSize                          *LazyFun
	GetTensorBufferPackedSize                    *LazyFun
	LockTensorBuffer                             *LazyFun
	UnlockTensorBuffer                           *LazyFun
	ClearTensorBuffer                            *LazyFun
	HasTensorBufferEvent                         *LazyFun
	GetTensorBufferEvent                         *LazyFun
	ClearTensorBufferEvent                       *LazyFun
	DestroyTensorBuffer                          *LazyFun
	WaitEvent                                    *LazyFun
	IsEventSignaled                              *LazyFun
	DestroyEvent                                 *LazyFun
}

// NewLibrarySymbols binds a Symbols table against a loaded ffi.Lib.
func NewLibrarySymbols(lib ffi.Lib) *Symbols {
	return NewSymbols(lib.Prep)
}

// NewSymbols binds every LiteRT 2.4.1 C API function against prep.
func NewSymbols(prep PrepFunc) *Symbols {
	fn := func(name string, ret *ffi.Type, args ...*ffi.Type) *LazyFun {
		return NewLazyFun(prep, name, ret, args...)
	}
	s32 := &ffi.TypeSint32
	u64 := &ffi.TypeUint64
	s64 := &ffi.TypeSint64
	u8 := &ffi.TypeUint8
	ptr := &ffi.TypePointer
	void := &ffi.TypeVoid

	return &Symbols{
		GetRuntimeAbiVersion: fn("LiteRtGetRuntimeAbiVersion", s32, ptr),

		CreateEnvironment:            fn("LiteRtCreateEnvironment", s32, s32, ptr, ptr),
		AddEnvironmentOptions:        fn("LiteRtAddEnvironmentOptions", s32, ptr, s32, ptr, u8),
		EnvironmentSupportsFP16:      fn("LiteRtEnvironmentSupportsFP16", s32, ptr, ptr),
		EnvironmentHasGpuEnvironment: fn("LiteRtEnvironmentHasGpuEnvironment", void, ptr, ptr),
		DestroyEnvironment:           fn("LiteRtDestroyEnvironment", void, ptr),

		CreateOptions:                  fn("LiteRtCreateOptions", s32, ptr),
		DestroyOptions:                 fn("LiteRtDestroyOptions", void, ptr),
		SetOptionsHardwareAccelerators: fn("LiteRtSetOptionsHardwareAccelerators", s32, ptr, s32),
		GetOptionsHardwareAccelerators: fn("LiteRtGetOptionsHardwareAccelerators", s32, ptr, ptr),
		AddOpaqueOptions:               fn("LiteRtAddOpaqueOptions", s32, ptr, ptr),
		CreateOpaqueOptions:            fn("LiteRtCreateOpaqueOptions", s32, ptr, ptr, ptr, ptr),
		DestroyOpaqueOptions:           fn("LiteRtDestroyOpaqueOptions", void, ptr),

		CreateModelFromFile:             fn("LiteRtCreateModelFromFile", s32, ptr, ptr, ptr),
		CreateModelFromBuffer:           fn("LiteRtCreateModelFromBuffer", s32, ptr, ptr, u64, ptr),
		DestroyModel:                    fn("LiteRtDestroyModel", void, ptr),
		GetModelMetadata:                fn("LiteRtGetModelMetadata", s32, ptr, ptr, ptr, ptr),
		GetNumModelSignatures:           fn("LiteRtGetNumModelSignatures", s32, ptr, ptr),
		GetModelSignature:               fn("LiteRtGetModelSignature", s32, ptr, u64, ptr),
		GetMainModelSubgraphIndex:       fn("LiteRtGetMainModelSubgraphIndex", s32, ptr, ptr),
		GetNumModelSubgraphs:            fn("LiteRtGetNumModelSubgraphs", s32, ptr, ptr),
		GetModelSubgraph:                fn("LiteRtGetModelSubgraph", s32, ptr, u64, ptr),
		GetSubgraphName:                 fn("LiteRtGetSubgraphName", s32, ptr, ptr),
		GetNumSubgraphInputs:            fn("LiteRtGetNumSubgraphInputs", s32, ptr, ptr),
		GetSubgraphInput:                fn("LiteRtGetSubgraphInput", s32, ptr, u64, ptr),
		GetNumSubgraphOutputs:           fn("LiteRtGetNumSubgraphOutputs", s32, ptr, ptr),
		GetSubgraphOutput:               fn("LiteRtGetSubgraphOutput", s32, ptr, u64, ptr),
		GetSignatureKey:                 fn("LiteRtGetSignatureKey", s32, ptr, ptr),
		GetNumSignatureInputs:           fn("LiteRtGetNumSignatureInputs", s32, ptr, ptr),
		GetSignatureInputName:           fn("LiteRtGetSignatureInputName", s32, ptr, u64, ptr),
		GetSignatureInputTensor:         fn("LiteRtGetSignatureInputTensor", s32, ptr, ptr, ptr),
		GetSignatureInputTensorByIndex:  fn("LiteRtGetSignatureInputTensorByIndex", s32, ptr, u64, ptr),
		GetNumSignatureOutputs:          fn("LiteRtGetNumSignatureOutputs", s32, ptr, ptr),
		GetSignatureOutputName:          fn("LiteRtGetSignatureOutputName", s32, ptr, u64, ptr),
		GetSignatureOutputTensor:        fn("LiteRtGetSignatureOutputTensor", s32, ptr, ptr, ptr),
		GetSignatureOutputTensorByIndex: fn("LiteRtGetSignatureOutputTensorByIndex", s32, ptr, u64, ptr),
		GetTensorName:                   fn("LiteRtGetTensorName", s32, ptr, ptr),
		GetTensorTypeId:                 fn("LiteRtGetTensorTypeId", s32, ptr, ptr),
		GetRankedTensorType:             fn("LiteRtGetRankedTensorType", s32, ptr, ptr),
		GetUnrankedTensorType:           fn("LiteRtGetUnrankedTensorType", s32, ptr, ptr),
		GetQuantizationTypeId:           fn("LiteRtGetQuantizationTypeId", s32, ptr, ptr),
		GetPerTensorQuantization:        fn("LiteRtGetPerTensorQuantization", s32, ptr, ptr),
		GetPerChannelQuantization:       fn("LiteRtGetPerChannelQuantization", s32, ptr, ptr),

		CreateCompiledModel:                           fn("LiteRtCreateCompiledModel", s32, ptr, ptr, ptr, ptr),
		DestroyCompiledModel:                          fn("LiteRtDestroyCompiledModel", void, ptr),
		GetCompiledModelInputBufferRequirements:       fn("LiteRtGetCompiledModelInputBufferRequirements", s32, ptr, u64, u64, ptr),
		GetCompiledModelOutputBufferRequirements:      fn("LiteRtGetCompiledModelOutputBufferRequirements", s32, ptr, u64, u64, ptr),
		GetCompiledModelInputTensorLayout:             fn("LiteRtGetCompiledModelInputTensorLayout", s32, ptr, u64, u64, ptr),
		GetCompiledModelOutputTensorLayouts:           fn("LiteRtGetCompiledModelOutputTensorLayouts", s32, ptr, u64, u64, ptr, u8),
		CompiledModelResizeInputTensor:                fn("LiteRtCompiledModelResizeInputTensor", s32, ptr, u64, u64, ptr, u64),
		CompiledModelResizeInputTensorNonStrict:       fn("LiteRtCompiledModelResizeInputTensorNonStrict", s32, ptr, u64, u64, ptr, u64),
		RunCompiledModel:                              fn("LiteRtRunCompiledModel", s32, ptr, u64, u64, ptr, u64, ptr),
		RunCompiledModelAsync:                         fn("LiteRtRunCompiledModelAsync", s32, ptr, u64, u64, ptr, u64, ptr, ptr),
		SetCompiledModelCancellationFunction:          fn("LiteRtSetCompiledModelCancellationFunction", s32, ptr, ptr, ptr),
		GetCompiledModelLastInferenceDuration:         fn("LiteRtGetCompiledModelLastInferenceDuration", s32, ptr, ptr),
		CompiledModelIsFullyAccelerated:               fn("LiteRtCompiledModelIsFullyAccelerated", s32, ptr, ptr),
		CompiledModelIsNonCpuFullyAccelerated:         fn("LiteRtCompiledModelIsNonCpuFullyAccelerated", s32, ptr, ptr),
		CompiledModelGetDelegationMetrics:             fn("LiteRtCompiledModelGetDelegationMetrics", s32, ptr, ptr),
		GetDelegationMetricsTotalNodeCount:            fn("LiteRtGetDelegationMetricsTotalNodeCount", s32, ptr, ptr),
		GetNumDelegationMetricsAccelerators:           fn("LiteRtGetNumDelegationMetricsAccelerators", s32, ptr, ptr),
		GetDelegationMetricsAccelerator:               fn("LiteRtGetDelegationMetricsAccelerator", s32, ptr, u64, ptr),
		GetAcceleratorDelegationMetricsName:           fn("LiteRtGetAcceleratorDelegationMetricsName", s32, ptr, ptr),
		GetAcceleratorDelegationMetricsHardwareType:   fn("LiteRtGetAcceleratorDelegationMetricsHardwareType", s32, ptr, ptr),
		GetAcceleratorDelegationMetricsDelegatedCount: fn("LiteRtGetAcceleratorDelegationMetricsDelegatedNodeCount", s32, ptr, ptr),
		GetAcceleratorDelegationMetricsPartitionCount: fn("LiteRtGetAcceleratorDelegationMetricsPartitionCount", s32, ptr, ptr),
		CompiledModelGetProfiler:                      fn("LiteRtCompiledModelGetProfiler", s32, ptr, ptr),
		CompiledModelClearErrors:                      fn("LiteRtCompiledModelClearErrors", s32, ptr),
		CompiledModelGetErrorMessages:                 fn("LiteRtCompiledModelGetErrorMessages", s32, ptr, ptr),

		StartProfiler:        fn("LiteRtStartProfiler", s32, ptr),
		StopProfiler:         fn("LiteRtStopProfiler", s32, ptr),
		ResetProfiler:        fn("LiteRtResetProfiler", s32, ptr),
		GetNumProfilerEvents: fn("LiteRtGetNumProfilerEvents", s32, ptr, ptr),
		GetProfileSummary:    fn("LiteRtGetProfileSummary", s32, ptr, ptr, ptr),

		GetNumTensorBufferRequirementsSupportedTypes: fn("LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes", s32, ptr, ptr),
		GetTensorBufferRequirementsSupportedType:     fn("LiteRtGetTensorBufferRequirementsSupportedTensorBufferType", s32, ptr, s32, ptr),
		GetTensorBufferRequirementsBufferSize:        fn("LiteRtGetTensorBufferRequirementsBufferSize", s32, ptr, ptr),
		GetTensorBufferRequirementsAlignment:         fn("LiteRtGetTensorBufferRequirementsAlignment", s32, ptr, ptr),
		CreateManagedTensorBuffer:                    fn("LiteRtCreateManagedTensorBuffer", s32, ptr, s32, ptr, u64, ptr),
		CreateManagedTensorBufferFromRequirements:    fn("LiteRtCreateManagedTensorBufferFromRequirements", s32, ptr, ptr, ptr, ptr),
		CreateTensorBufferFromHostMemory:             fn("LiteRtCreateTensorBufferFromHostMemory", s32, ptr, ptr, u64, ptr, ptr),
		DuplicateTensorBuffer:                        fn("LiteRtDuplicateTensorBuffer", s32, ptr),
		GetTensorBufferType:                          fn("LiteRtGetTensorBufferType", s32, ptr, ptr),
		GetTensorBufferTensorType:                    fn("LiteRtGetTensorBufferTensorType", s32, ptr, ptr),
		GetTensorBufferSize:                          fn("LiteRtGetTensorBufferSize", s32, ptr, ptr),
		GetTensorBufferPackedSize:                    fn("LiteRtGetTensorBufferPackedSize", s32, ptr, ptr),
		LockTensorBuffer:                             fn("LiteRtLockTensorBuffer", s32, ptr, ptr, s32),
		UnlockTensorBuffer:                           fn("LiteRtUnlockTensorBuffer", s32, ptr),
		ClearTensorBuffer:                            fn("LiteRtClearTensorBuffer", s32, ptr),
		HasTensorBufferEvent:                         fn("LiteRtHasTensorBufferEvent", s32, ptr, ptr),
		GetTensorBufferEvent:                         fn("LiteRtGetTensorBufferEvent", s32, ptr, ptr),
		ClearTensorBufferEvent:                       fn("LiteRtClearTensorBufferEvent", s32, ptr),
		DestroyTensorBuffer:                          fn("LiteRtDestroyTensorBuffer", void, ptr),
		WaitEvent:                                    fn("LiteRtWaitEvent", s32, ptr, s64),
		IsEventSignaled:                              fn("LiteRtIsEventSignaled", s32, ptr, ptr),
		DestroyEvent:                                 fn("LiteRtDestroyEvent", void, ptr),
	}
}
