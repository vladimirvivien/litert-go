package litert

import (
	"strings"
	"testing"
)

func TestCPUOptionsTOML(t *testing.T) {
	opts := CPUOptions{
		KernelMode:         KernelXNNPACK,
		NumThreads:         4,
		Flags:              new(uint32(1)),
		HintSingleDelegate: new(true),
		WeightCachePath:    "/tmp/xnn.cache",
	}
	got, err := opts.ToTOML()
	if err != nil {
		t.Fatalf("ToTOML: %v", err)
	}
	for _, want := range []string{
		`kernel_mode = "xnnpack"`,
		`num_threads = 4`,
		`flags = 1`,
		`hint_fully_delegated_to_single_delegate = true`,
		`weight_cache_file_path = "/tmp/xnn.cache"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in TOML:\n%s", want, got)
		}
	}

	badFDAndPath := CPUOptions{WeightCachePath: "/tmp/a", WeightCacheFileDesc: new(3)}
	if _, err := badFDAndPath.ToTOML(); err == nil {
		t.Fatal("expected error when both WeightCachePath and WeightCacheFileDesc are set")
	}
	if _, err := (CPUOptions{NumThreads: -1}).ToTOML(); err == nil {
		t.Fatal("expected error for negative NumThreads")
	}
	if _, err := (CPUOptions{KernelMode: "invalid"}).ToTOML(); err == nil {
		t.Fatal("expected error for invalid KernelMode")
	}
}

func TestGPUOptionsTOML(t *testing.T) {
	opts := GPUOptions{
		Backend:                   new(GPUBackendWebGPU),
		Precision:                 new(GPUPrecisionFP16),
		ConstantTensorsSharing:    new(true),
		InfiniteFloatCapping:      new(true),
		SerializationDir:          "/tmp/gpu_cache",
		ModelCacheKey:             "mobilenet_v2",
		SerializeProgramCache:     new(true),
		SerializeExternalTensors:  new(true),
		ExternalTensorPatterns:    []string{"state_", "hidden_"},
		NumStepsCommandBufferPrep: new(2),
	}
	got, err := opts.ToTOML()
	if err != nil {
		t.Fatalf("ToTOML: %v", err)
	}
	for _, want := range []string{
		`backend = 2`,
		`precision = 1`,
		`enable_constant_tensors_sharing = true`,
		`enable_infinite_float_capping = true`,
		`serialization_dir = "/tmp/gpu_cache"`,
		`model_cache_key = "mobilenet_v2"`,
		`serialize_program_cache = true`,
		`serialize_external_tensors = true`,
		`external_tensor_patterns = ["state_", "hidden_"]`,
		`num_steps_of_command_buffer_preparations = 2`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in GPU TOML:\n%s", want, got)
		}
	}

	if _, err := (GPUOptions{KernelBatchSize: new(-1)}).ToTOML(); err == nil {
		t.Fatal("expected error for negative KernelBatchSize")
	}
}

func TestRuntimeOptionsTOML(t *testing.T) {
	opts := RuntimeOptions{
		EnableProfiling:                new(true),
		ErrorReporterMode:              new(ErrorReporterBuffer),
		CompressQuantizationZeroPoints: new(true),
		DisableDelegateClustering:      new(false),
	}
	got, err := opts.ToTOML()
	if err != nil {
		t.Fatalf("ToTOML: %v", err)
	}
	for _, want := range []string{
		`enable_profiling = true`,
		`error_reporter_mode = 2`,
		`compress_quantization_zero_points = true`,
		`disable_delegate_clustering = false`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in Runtime TOML:\n%s", want, got)
		}
	}
}
