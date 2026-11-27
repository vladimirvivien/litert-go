package litert

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/vladimirvivien/litert-go/internal/capi"
)

var (
	opaqueNoopOnce sync.Once
	opaqueNoopPtr  uintptr
)

func getOpaqueNoopFree() uintptr {
	opaqueNoopOnce.Do(func() {
		opaqueNoopPtr = purego.NewCallback(func(uintptr) uintptr { return 0 })
	})
	return opaqueNoopPtr
}

// Options owns compilation options and accelerator opaque option payloads.
type Options struct {
	syms     *capi.Symbols
	handle   uintptr
	pin      runtime.Pinner
	retained [][]byte
	closed   bool
}

// NewOptions creates compilation options targeting the specified accelerators.
func NewOptions(accel Accelerator) (*Options, error) {
	syms, err := ensureInit("")
	if err != nil {
		return nil, err
	}
	return newOptionsWithSymbols(syms, accel)
}

func newOptionsWithSymbols(syms *capi.Symbols, accel Accelerator) (*Options, error) {
	if syms == nil {
		return nil, &Error{Op: "NewOptions", Sentinel: ErrClosed}
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	var handle uintptr
	st, callErr := syms.CreateOptions.CallStatus(&pin, capi.Out(&pin, &handle))
	if err := statusErr("LiteRtCreateOptions", st, callErr); err != nil {
		return nil, err
	}

	opts := &Options{syms: syms, handle: handle}
	if err := opts.SetHardwareAccelerators(accel); err != nil {
		_ = opts.Close()
		return nil, err
	}
	return opts, nil
}

func (o *Options) valid() bool {
	return o != nil && !o.closed && o.handle != 0 && o.syms != nil
}

// SetHardwareAccelerators updates the target hardware accelerators bitmask.
func (o *Options) SetHardwareAccelerators(accel Accelerator) error {
	if !o.valid() {
		return ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := o.handle
	a := int32(accel)
	st, callErr := o.syms.SetOptionsHardwareAccelerators.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Ptr(&pin, &a),
	)
	return statusErr("LiteRtSetOptionsHardwareAccelerators", st, callErr)
}

// HardwareAccelerators returns the configured hardware accelerators bitmask.
func (o *Options) HardwareAccelerators() (Accelerator, error) {
	if !o.valid() {
		return 0, ErrClosed
	}
	var pin runtime.Pinner
	defer pin.Unpin()

	h := o.handle
	var out int32
	st, callErr := o.syms.GetOptionsHardwareAccelerators.CallStatus(
		&pin,
		capi.Ptr(&pin, &h),
		capi.Out(&pin, &out),
	)
	if err := statusErr("LiteRtGetOptionsHardwareAccelerators", st, callErr); err != nil {
		return 0, err
	}
	return Accelerator(out), nil
}

// AddOpaqueOption attaches a typed opaque payload under identifier. The
// identifier and payload bytes are pinned and owned by Options until Close.
func (o *Options) AddOpaqueOption(identifier, payload string) error {
	if !o.valid() {
		return ErrClosed
	}
	if identifier == "" {
		return &Error{Op: "AddOpaqueOption", Sentinel: ErrInvalidArgument, Msg: "empty identifier"}
	}

	idBytes := capi.CBytes(identifier)
	payloadBytes := capi.CBytes(payload)
	o.pin.Pin(&idBytes[0])
	o.pin.Pin(&payloadBytes[0])
	o.retained = append(o.retained, idBytes, payloadBytes)

	var callPin runtime.Pinner
	defer callPin.Unpin()

	idPtr := unsafe.Pointer(&idBytes[0])
	payloadPtr := unsafe.Pointer(&payloadBytes[0])
	destructor := getOpaqueNoopFree()
	var opaque uintptr

	st, callErr := o.syms.CreateOpaqueOptions.CallStatus(
		&callPin,
		capi.Ptr(&callPin, &idPtr),
		capi.Ptr(&callPin, &payloadPtr),
		capi.Ptr(&callPin, &destructor),
		capi.Out(&callPin, &opaque),
	)
	if err := statusErr("LiteRtCreateOpaqueOptions", st, callErr); err != nil {
		return err
	}

	h := o.handle
	st, callErr = o.syms.AddOpaqueOptions.CallStatus(
		&callPin,
		capi.Ptr(&callPin, &h),
		capi.Ptr(&callPin, &opaque),
	)
	if err := statusErr("LiteRtAddOpaqueOptions", st, callErr); err != nil {
		// Ownership was not transferred on failure; clean up created opaque option.
		_ = o.syms.DestroyOpaqueOptions.CallVoid(capi.Ptr(&callPin, &opaque))
		return err
	}
	return nil
}

// Close destroys the Options handle and unpins retained option payloads.
func (o *Options) Close() error {
	if o == nil || o.closed {
		return nil
	}
	o.closed = true

	var err error
	if o.handle != 0 && o.syms != nil {
		var pin runtime.Pinner
		h := o.handle
		err = statusErr("LiteRtDestroyOptions", 0, o.syms.DestroyOptions.CallVoid(capi.Ptr(&pin, &h)))
		pin.Unpin()
		o.handle = 0
	}
	o.pin.Unpin()
	o.retained = nil
	return err
}

// CPUKernelMode selects how CPU operations execute.
type CPUKernelMode string

const (
	KernelXNNPACK   CPUKernelMode = "xnnpack"
	KernelBuiltin   CPUKernelMode = "builtin"
	KernelReference CPUKernelMode = "reference"
)

// CPUOptions configures the XNNPACK/CPU accelerator opaque payload ("xnnpack").
type CPUOptions struct {
	KernelMode          CPUKernelMode
	NumThreads          int
	Flags               *uint32
	HintSingleDelegate  *bool
	WeightCachePath     string
	WeightCacheFileDesc *int
}

// Validate verifies CPUOptions constraints.
func (c CPUOptions) Validate() error {
	if c.KernelMode != "" && c.KernelMode != KernelXNNPACK && c.KernelMode != KernelBuiltin && c.KernelMode != KernelReference {
		return &Error{Op: "CPUOptions", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("invalid kernel mode %q", c.KernelMode)}
	}
	if c.NumThreads < 0 {
		return &Error{Op: "CPUOptions", Sentinel: ErrInvalidArgument, Msg: "num_threads cannot be negative"}
	}
	if c.WeightCachePath != "" && c.WeightCacheFileDesc != nil {
		return &Error{Op: "CPUOptions", Sentinel: ErrInvalidArgument, Msg: "cannot set both WeightCachePath and WeightCacheFileDesc"}
	}
	return nil
}

// ToTOML serializes CPUOptions using LiteRT's "xnnpack" TOML schema.
func (c CPUOptions) ToTOML() (string, error) {
	if err := c.Validate(); err != nil {
		return "", err
	}
	var b strings.Builder
	if c.KernelMode != "" {
		fmt.Fprintf(&b, "kernel_mode = %s\n", strconv.Quote(string(c.KernelMode)))
	}
	if c.NumThreads > 0 {
		fmt.Fprintf(&b, "num_threads = %d\n", c.NumThreads)
	}
	if c.Flags != nil {
		fmt.Fprintf(&b, "flags = %d\n", *c.Flags)
	}
	if c.HintSingleDelegate != nil {
		fmt.Fprintf(&b, "hint_fully_delegated_to_single_delegate = %t\n", *c.HintSingleDelegate)
	}
	if c.WeightCachePath != "" {
		fmt.Fprintf(&b, "weight_cache_file_path = %s\n", strconv.Quote(c.WeightCachePath))
	}
	if c.WeightCacheFileDesc != nil {
		fmt.Fprintf(&b, "weight_cache_file_descriptor = %d\n", *c.WeightCacheFileDesc)
	}
	return b.String(), nil
}

// AddCPUOptions attaches validated CPU options under identifier "xnnpack".
func (o *Options) AddCPUOptions(cfg CPUOptions) error {
	toml, err := cfg.ToTOML()
	if err != nil {
		return err
	}
	if toml == "" {
		return nil
	}
	return o.AddOpaqueOption("xnnpack", toml)
}

// GPUBackend selects a GPU execution backend.
type GPUBackend int

const (
	GPUBackendAutomatic GPUBackend = 0
	GPUBackendOpenCL    GPUBackend = 1
	GPUBackendWebGPU    GPUBackend = 2
	GPUBackendOpenGL    GPUBackend = 3
)

// GPUPrecision selects delegate floating-point precision.
type GPUPrecision int

const (
	GPUPrecisionDefault GPUPrecision = 0
	GPUPrecisionFP16    GPUPrecision = 1
	GPUPrecisionFP32    GPUPrecision = 2
)

// GPUPriority selects GPU queue priority.
type GPUPriority int

const (
	GPUPriorityDefault GPUPriority = 0
	GPUPriorityLow     GPUPriority = 1
	GPUPriorityNormal  GPUPriority = 2
	GPUPriorityHigh    GPUPriority = 3
)

// GPUWaitType selects synchronous wait behavior.
type GPUWaitType int

const (
	GPUWaitDefault   GPUWaitType = 0
	GPUWaitPassive   GPUWaitType = 1
	GPUWaitActive    GPUWaitType = 2
	GPUWaitDoNotWait GPUWaitType = 3
)

// GPUBufferStorageType selects delegate storage preference.
type GPUBufferStorageType int

const (
	GPUStorageDefault   GPUBufferStorageType = 0
	GPUStorageBuffer    GPUBufferStorageType = 1
	GPUStorageTexture2D GPUBufferStorageType = 2
)

// GPUOptions configures the GPU accelerator opaque payload ("gpu_options").
type GPUOptions struct {
	Backend                      *GPUBackend
	Precision                    *GPUPrecision
	Priority                     *GPUPriority
	WaitType                     *GPUWaitType
	BufferStorageType            *GPUBufferStorageType
	ConstantTensorsSharing       *bool
	InfiniteFloatCapping         *bool
	BenchmarkMode                *bool
	AllowSrcQuantizedFcConvOps   *bool
	HintWaitingForCompletion     *bool
	KernelBatchSize              *int
	PreferTextureWeights         *bool
	SerializationDir             string
	ModelCacheKey                string
	SerializeProgramCache        *bool
	CacheCompiledProgramsOnly    *bool
	SerializeExternalTensors     *bool
	ExternalTensorsMode          *bool
	ExternalTensorPatterns       []string
	MadviseOriginalSharedTensors *bool
	DisableShaderOptimization    *bool
	NumStepsCommandBufferPrep    *int
	UseMetalArgumentBuffers      *bool
	PreferredDeviceSubstr        string
	HintSingleDelegate           *bool
	NumThreadsToUpload           *int
	NumThreadsToCompile          *int
	ConvertWeightsOnGPU          *bool
	WaitForWeightsConversion     *bool
	ProgramCacheFD               *int
	WeightCacheFD                *int
}

// Validate checks GPUOptions constraints.
func (g GPUOptions) Validate() error {
	if g.Backend != nil && (*g.Backend < GPUBackendAutomatic || *g.Backend > GPUBackendOpenGL) {
		return &Error{Op: "GPUOptions", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("invalid GPU backend %d", *g.Backend)}
	}
	if g.Precision != nil && (*g.Precision < GPUPrecisionDefault || *g.Precision > GPUPrecisionFP32) {
		return &Error{Op: "GPUOptions", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("invalid GPU precision %d", *g.Precision)}
	}
	if g.KernelBatchSize != nil && *g.KernelBatchSize < 0 {
		return &Error{Op: "GPUOptions", Sentinel: ErrInvalidArgument, Msg: "kernel_batch_size cannot be negative"}
	}
	if g.NumStepsCommandBufferPrep != nil && *g.NumStepsCommandBufferPrep < 0 {
		return &Error{Op: "GPUOptions", Sentinel: ErrInvalidArgument, Msg: "num_steps_of_command_buffer_preparations cannot be negative"}
	}
	return nil
}

// ToTOML serializes GPUOptions using LiteRT's "gpu_options" TOML schema.
func (g GPUOptions) ToTOML() (string, error) {
	if err := g.Validate(); err != nil {
		return "", err
	}
	var b strings.Builder
	writeTOMLBool(&b, "enable_constant_tensors_sharing", g.ConstantTensorsSharing)
	writeTOMLBool(&b, "enable_infinite_float_capping", g.InfiniteFloatCapping)
	writeTOMLBool(&b, "benchmark_mode", g.BenchmarkMode)
	writeTOMLBool(&b, "allow_src_quantized_fc_conv_ops", g.AllowSrcQuantizedFcConvOps)
	writeTOMLBool(&b, "hint_waiting_for_completion", g.HintWaitingForCompletion)
	writeTOMLInt(&b, "kernel_batch_size", g.KernelBatchSize)
	if g.Precision != nil {
		fmt.Fprintf(&b, "precision = %d\n", int(*g.Precision))
	}
	if g.BufferStorageType != nil {
		fmt.Fprintf(&b, "buffer_storage_type = %d\n", int(*g.BufferStorageType))
	}
	writeTOMLBool(&b, "prefer_texture_weights", g.PreferTextureWeights)
	writeTOMLString(&b, "serialization_dir", g.SerializationDir)
	writeTOMLString(&b, "model_cache_key", g.ModelCacheKey)
	writeTOMLBool(&b, "serialize_program_cache", g.SerializeProgramCache)
	writeTOMLBool(&b, "serialize_external_tensors", g.SerializeExternalTensors)
	writeTOMLBool(&b, "external_tensors_mode", g.ExternalTensorsMode)
	if len(g.ExternalTensorPatterns) > 0 {
		b.WriteString("external_tensor_patterns = [")
		for i, pat := range g.ExternalTensorPatterns {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(strconv.Quote(pat))
		}
		b.WriteString("]\n")
	}
	if g.Backend != nil {
		fmt.Fprintf(&b, "backend = %d\n", int(*g.Backend))
	}
	if g.Priority != nil {
		fmt.Fprintf(&b, "priority = %d\n", int(*g.Priority))
	}
	writeTOMLBool(&b, "madvise_original_shared_tensors", g.MadviseOriginalSharedTensors)
	writeTOMLInt(&b, "num_steps_of_command_buffer_preparations", g.NumStepsCommandBufferPrep)
	writeTOMLBool(&b, "use_metal_argument_buffers", g.UseMetalArgumentBuffers)
	if g.WaitType != nil {
		fmt.Fprintf(&b, "wait_type = %d\n", int(*g.WaitType))
	}
	writeTOMLString(&b, "preferred_device_substr", g.PreferredDeviceSubstr)
	writeTOMLBool(&b, "hint_fully_delegated_to_single_delegate", g.HintSingleDelegate)
	writeTOMLInt(&b, "num_threads_to_upload", g.NumThreadsToUpload)
	writeTOMLInt(&b, "num_threads_to_compile", g.NumThreadsToCompile)
	writeTOMLBool(&b, "convert_weights_on_gpu", g.ConvertWeightsOnGPU)
	writeTOMLBool(&b, "wait_for_weights_conversion_complete", g.WaitForWeightsConversion)
	writeTOMLBool(&b, "disable_shader_optimization", g.DisableShaderOptimization)
	writeTOMLInt(&b, "program_cache_fd", g.ProgramCacheFD)
	writeTOMLInt(&b, "weight_cache_fd", g.WeightCacheFD)
	writeTOMLBool(&b, "cache_only_compiled_programs", g.CacheCompiledProgramsOnly)
	return b.String(), nil
}

// AddGPUOptions attaches validated GPU options under identifier "gpu_options".
func (o *Options) AddGPUOptions(cfg GPUOptions) error {
	toml, err := cfg.ToTOML()
	if err != nil {
		return err
	}
	if toml == "" {
		return nil
	}
	return o.AddOpaqueOption("gpu_options", toml)
}

// ErrorReporterMode selects how CompiledModel collects diagnostic messages.
type ErrorReporterMode int

const (
	ErrorReporterNone   ErrorReporterMode = 0
	ErrorReporterStderr ErrorReporterMode = 1
	ErrorReporterBuffer ErrorReporterMode = 2
)

// RuntimeOptions configures general runtime behavior ("runtime_options_string").
type RuntimeOptions struct {
	EnableProfiling                *bool
	ErrorReporterMode              *ErrorReporterMode
	CompressQuantizationZeroPoints *bool
	DisableDelegateClustering      *bool
}

// ToTOML serializes RuntimeOptions using LiteRT's "runtime_options_string" schema.
func (r RuntimeOptions) ToTOML() (string, error) {
	if r.ErrorReporterMode != nil && (*r.ErrorReporterMode < ErrorReporterNone || *r.ErrorReporterMode > ErrorReporterBuffer) {
		return "", &Error{Op: "RuntimeOptions", Sentinel: ErrInvalidArgument, Msg: fmt.Sprintf("invalid error reporter mode %d", *r.ErrorReporterMode)}
	}
	var b strings.Builder
	writeTOMLBool(&b, "enable_profiling", r.EnableProfiling)
	if r.ErrorReporterMode != nil {
		fmt.Fprintf(&b, "error_reporter_mode = %d\n", int(*r.ErrorReporterMode))
	}
	writeTOMLBool(&b, "compress_quantization_zero_points", r.CompressQuantizationZeroPoints)
	writeTOMLBool(&b, "disable_delegate_clustering", r.DisableDelegateClustering)
	return b.String(), nil
}

// AddRuntimeOptions attaches runtime options under "runtime_options_string".
func (o *Options) AddRuntimeOptions(cfg RuntimeOptions) error {
	toml, err := cfg.ToTOML()
	if err != nil {
		return err
	}
	if toml == "" {
		return nil
	}
	return o.AddOpaqueOption("runtime_options_string", toml)
}

func writeTOMLBool(b *strings.Builder, key string, v *bool) {
	if v != nil {
		fmt.Fprintf(b, "%s = %t\n", key, *v)
	}
}

func writeTOMLInt(b *strings.Builder, key string, v *int) {
	if v != nil {
		fmt.Fprintf(b, "%s = %d\n", key, *v)
	}
}

func writeTOMLString(b *strings.Builder, key, v string) {
	if v != "" {
		fmt.Fprintf(b, "%s = %s\n", key, strconv.Quote(v))
	}
}
