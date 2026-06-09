package litert

import "github.com/jupiterrider/ffi"

// One *lazyFun per bound LiteRt C entry point. Pointer handles and out-params
// are TypePointer; LiteRtStatus / enums are TypeSint32; size_t / LiteRtParamIndex
// are TypeUint64. Each resolves its symbol on first Call.

var (
	// ---- Environment ----
	createEnvironmentFunc = newLazyFun(
		"LiteRtCreateEnvironment",
		&ffi.TypeSint32,
		&ffi.TypeSint32,  // int num_options
		&ffi.TypePointer, // const LiteRtEnvOption* options
		&ffi.TypePointer, // LiteRtEnvironment* environment
	)
	destroyEnvironmentFunc = newLazyFun(
		"LiteRtDestroyEnvironment",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- Options ----
	createOptionsFunc = newLazyFun(
		"LiteRtCreateOptions",
		&ffi.TypeSint32, &ffi.TypePointer)
	setOptionsHardwareAcceleratorsFunc = newLazyFun(
		"LiteRtSetOptionsHardwareAccelerators",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	destroyOptionsFunc = newLazyFun(
		"LiteRtDestroyOptions",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- Opaque (accelerator) options ----
	createOpaqueOptionsFunc = newLazyFun(
		"LiteRtCreateOpaqueOptions",
		&ffi.TypeSint32,
		&ffi.TypePointer, // const char* payload_identifier
		&ffi.TypePointer, // void* payload_data
		&ffi.TypePointer, // void (*payload_destructor)(void*)
		&ffi.TypePointer, // LiteRtOpaqueOptions* options
	)
	addOpaqueOptionsFunc = newLazyFun(
		"LiteRtAddOpaqueOptions",
		&ffi.TypeSint32,
		&ffi.TypePointer, // LiteRtOptions options
		&ffi.TypePointer, // LiteRtOpaqueOptions opaque_options
	)

	// ---- Model load ----
	createModelFromFileFunc = newLazyFun(
		"LiteRtCreateModelFromFile",
		&ffi.TypeSint32,
		&ffi.TypePointer, // LiteRtEnvironment
		&ffi.TypePointer, // const char* filename
		&ffi.TypePointer, // LiteRtModel* model
	)
	// LiteRtCreateModelFromBuffer(buffer, size, model*). The distributed LiteRT
	// 2.1.5 prebuilt has no leading LiteRtEnvironment argument.
	createModelFromBufferFunc = newLazyFun(
		"LiteRtCreateModelFromBuffer",
		&ffi.TypeSint32,
		&ffi.TypePointer, // const void* buffer_addr
		&ffi.TypeUint64,  // size_t buffer_size
		&ffi.TypePointer, // LiteRtModel* model
	)
	destroyModelFunc = newLazyFun(
		"LiteRtDestroyModel",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- Signature introspection ----
	getNumModelSignaturesFunc = newLazyFun(
		"LiteRtGetNumModelSignatures",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	getModelSignatureFunc = newLazyFun(
		"LiteRtGetModelSignature",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint64, &ffi.TypePointer)
	getSignatureKeyFunc = newLazyFun(
		"LiteRtGetSignatureKey",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	getNumSignatureInputsFunc = newLazyFun(
		"LiteRtGetNumSignatureInputs",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	getSignatureInputNameFunc = newLazyFun(
		"LiteRtGetSignatureInputName",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint64, &ffi.TypePointer)
	getNumSignatureOutputsFunc = newLazyFun(
		"LiteRtGetNumSignatureOutputs",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	getSignatureOutputNameFunc = newLazyFun(
		"LiteRtGetSignatureOutputName",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint64, &ffi.TypePointer)
	getSignatureInputTensorFunc = newLazyFun(
		"LiteRtGetSignatureInputTensor",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	getSignatureOutputTensorFunc = newLazyFun(
		"LiteRtGetSignatureOutputTensor",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	getRankedTensorTypeFunc = newLazyFun(
		"LiteRtGetRankedTensorType",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Compiled model ----
	createCompiledModelFunc = newLazyFun(
		"LiteRtCreateCompiledModel",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	compiledModelIsFullyAcceleratedFunc = newLazyFun(
		"LiteRtCompiledModelIsFullyAccelerated",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	compiledModelResizeInputTensorFunc = newLazyFun(
		"LiteRtCompiledModelResizeInputTensorNonStrict",
		&ffi.TypeSint32,
		&ffi.TypePointer, // LiteRtCompiledModel
		&ffi.TypeUint64,  // signature_index
		&ffi.TypeUint64,  // input_index
		&ffi.TypePointer, // const int* dims
		&ffi.TypeUint64,  // dims_size
	)
	getCompiledModelInputBufferRequirementsFunc = newLazyFun(
		"LiteRtGetCompiledModelInputBufferRequirements",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint64, &ffi.TypeUint64, &ffi.TypePointer)
	getCompiledModelOutputBufferRequirementsFunc = newLazyFun(
		"LiteRtGetCompiledModelOutputBufferRequirements",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint64, &ffi.TypeUint64, &ffi.TypePointer)
	runCompiledModelFunc = newLazyFun(
		"LiteRtRunCompiledModel",
		&ffi.TypeSint32,
		&ffi.TypePointer, // LiteRtCompiledModel
		&ffi.TypeUint64,  // signature_index
		&ffi.TypeUint64,  // num_input_buffers
		&ffi.TypePointer, // LiteRtTensorBuffer* input_buffers
		&ffi.TypeUint64,  // num_output_buffers
		&ffi.TypePointer, // LiteRtTensorBuffer* output_buffers
	)
	runCompiledModelAsyncFunc = newLazyFun(
		"LiteRtRunCompiledModelAsync",
		&ffi.TypeSint32,
		&ffi.TypePointer, // LiteRtCompiledModel
		&ffi.TypeUint64,  // signature_index
		&ffi.TypeUint64,  // num_input_buffers
		&ffi.TypePointer, // LiteRtTensorBuffer* input_buffers
		&ffi.TypeUint64,  // num_output_buffers
		&ffi.TypePointer, // LiteRtTensorBuffer* output_buffers
		&ffi.TypePointer, // bool* async
	)
	destroyCompiledModelFunc = newLazyFun(
		"LiteRtDestroyCompiledModel",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- Tensor buffers ----
	createManagedTensorBufferFunc = newLazyFun(
		"LiteRtCreateManagedTensorBuffer",
		&ffi.TypeSint32,
		&ffi.TypePointer, // LiteRtEnvironment
		&ffi.TypeSint32,  // LiteRtTensorBufferType
		&ffi.TypePointer, // const LiteRtRankedTensorType*
		&ffi.TypeUint64,  // size_t buffer_size
		&ffi.TypePointer, // LiteRtTensorBuffer* buffer
	)
	lockTensorBufferFunc = newLazyFun(
		"LiteRtLockTensorBuffer",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32)
	unlockTensorBufferFunc = newLazyFun(
		"LiteRtUnlockTensorBuffer",
		&ffi.TypeSint32, &ffi.TypePointer)
	destroyTensorBufferFunc = newLazyFun(
		"LiteRtDestroyTensorBuffer",
		&ffi.TypeVoid, &ffi.TypePointer)
	hasTensorBufferEventFunc = newLazyFun(
		"LiteRtHasTensorBufferEvent",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	getTensorBufferEventFunc = newLazyFun(
		"LiteRtGetTensorBufferEvent",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	waitEventFunc = newLazyFun(
		"LiteRtWaitEvent",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint64)
	clearTensorBufferEventFunc = newLazyFun(
		"LiteRtClearTensorBufferEvent",
		&ffi.TypeSint32, &ffi.TypePointer)

	// ---- Tensor buffer requirements ----
	getTensorBufferRequirementsBufferSizeFunc = newLazyFun(
		"LiteRtGetTensorBufferRequirementsBufferSize",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	getTensorBufferRequirementsSupportedTensorBufferTypeFunc = newLazyFun(
		"LiteRtGetTensorBufferRequirementsSupportedTensorBufferType",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypePointer)
	getNumTensorBufferRequirementsSupportedTypesFunc = newLazyFun(
		"LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
)
