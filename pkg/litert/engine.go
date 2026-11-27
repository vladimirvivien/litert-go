package litert

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/vladimirvivien/litert-go/internal/capi"
	"github.com/vladimirvivien/litert-go/pkg/libfetch"
)

// NamedInputs maps model input tensor names to typed slices ([]float32,
// []uint8, []int8, []int16, etc.), raw packed bytes ([]byte), or caller-managed
// *TensorBuffer values.
type NamedInputs map[string]any

// NamedOutputs maps model output tensor names to typed Go slices (for example
// []float32, []uint8, []int8, []int16) or raw []byte for sub-byte/unmapped types.
type NamedOutputs map[string]any

type openConfig struct {
	libDir         string
	autoFetch      bool
	fetchOpts      []libfetch.Option
	accel          Accelerator
	hasCPU         bool
	cpuOpts        CPUOptions
	hasGPU         bool
	gpuOpts        GPUOptions
	hasRuntime     bool
	runtimeOpts    RuntimeOptions
	envOpts        []EnvOption
	hasLogSeverity bool
	logSeverity    LogSeverity
	cacheDir       string
}

// OpenOption configures Open and OpenBytes.
type OpenOption func(*openConfig)

// WithLibDir sets the directory searched for libLiteRt before LITERT_LIB.
func WithLibDir(dir string) OpenOption {
	return func(c *openConfig) { c.libDir = dir }
}

// WithAutoFetch enables automatic download of LiteRT 2.4.1 shared libraries
// via pkg/libfetch if Init cannot locate an existing installation.
func WithAutoFetch(opts ...libfetch.Option) OpenOption {
	return func(c *openConfig) {
		c.autoFetch = true
		c.fetchOpts = append(c.fetchOpts, opts...)
	}
}

// WithAccelerator sets the hardware accelerator selection bitmask (defaults to AccelCPU).
func WithAccelerator(accel Accelerator) OpenOption {
	return func(c *openConfig) { c.accel = accel }
}

// WithCPUOptions configures the CPU/XNNPACK accelerator delegate options.
func WithCPUOptions(cfg CPUOptions) OpenOption {
	return func(c *openConfig) {
		c.hasCPU = true
		c.cpuOpts = cfg
	}
}

// WithGPUOptions configures GPU accelerator delegate options.
func WithGPUOptions(cfg GPUOptions) OpenOption {
	return func(c *openConfig) {
		c.hasGPU = true
		c.gpuOpts = cfg
	}
}

// WithRuntimeOptions configures general runtime options.
func WithRuntimeOptions(cfg RuntimeOptions) OpenOption {
	return func(c *openConfig) {
		c.hasRuntime = true
		c.runtimeOpts = cfg
	}
}

// WithEnvOptions appends raw Environment configuration options.
func WithEnvOptions(opts ...EnvOption) OpenOption {
	return func(c *openConfig) { c.envOpts = append(c.envOpts, opts...) }
}

// WithLogSeverity sets the minimum LiteRT runtime logging threshold.
func WithLogSeverity(sev LogSeverity) OpenOption {
	return func(c *openConfig) {
		c.hasLogSeverity = true
		c.logSeverity = sev
	}
}

// WithCacheDir configures the LiteRT compiler cache directory.
func WithCacheDir(dir string) OpenOption {
	return func(c *openConfig) { c.cacheDir = dir }
}

type sigSelector struct {
	hasKey   bool
	key      string
	hasIndex bool
	index    int
}

// RunOption configures Engine.Run.
type RunOption func(*runConfig)

type runConfig struct {
	sig sigSelector
}

// WithSignature selects the named signature to execute in Engine.Run.
func WithSignature(key string) RunOption {
	return func(c *runConfig) {
		c.sig.hasKey = true
		c.sig.key = key
	}
}

// WithSignatureIndex selects the zero-based signature index to execute in Engine.Run.
func WithSignatureIndex(index int) RunOption {
	return func(c *runConfig) {
		c.sig.hasIndex = true
		c.sig.index = index
	}
}

// RunnerOption configures Engine.NewRunner.
type RunnerOption func(*runnerConfig)

type runnerConfig struct {
	sig         sigSelector
	inputShapes map[string][]int32
}

// WithRunnerSignature selects the named signature bound by Engine.NewRunner.
func WithRunnerSignature(key string) RunnerOption {
	return func(c *runnerConfig) {
		c.sig.hasKey = true
		c.sig.key = key
	}
}

// WithRunnerSignatureIndex selects the zero-based signature index bound by Engine.NewRunner.
func WithRunnerSignatureIndex(index int) RunnerOption {
	return func(c *runnerConfig) {
		c.sig.hasIndex = true
		c.sig.index = index
	}
}

// WithInputShapes resizes named input tensors before allocating Runner buffers.
func WithInputShapes(shapes map[string][]int32) RunnerOption {
	return func(c *runnerConfig) {
		if len(shapes) == 0 {
			return
		}
		if c.inputShapes == nil {
			c.inputShapes = make(map[string][]int32, len(shapes))
		}
		for k, v := range shapes {
			c.inputShapes[k] = slices.Clone(v)
		}
	}
}

// Engine manages a loaded LiteRT Environment, Model, CompiledModel, and signature metadata.
type Engine struct {
	env        *Environment
	model      *Model
	cm         *CompiledModel
	sigs       []*Signature
	defaultSig *Signature
	closed     bool
}

// Open loads a .tflite file from modelPath, compiles it with opts, and returns
// an owning Engine ready for one-shot Run or reusable NewRunner execution.
func Open(ctx context.Context, modelPath string, opts ...OpenOption) (*Engine, error) {
	return openEngine(ctx, func(env *Environment) (*Model, error) {
		return OpenModelFromFile(env, modelPath)
	}, opts...)
}

// OpenBytes loads an in-memory .tflite model buffer, compiles it with opts,
// and returns an owning Engine. Callers must not mutate data while the Engine
// remains open.
func OpenBytes(ctx context.Context, data []byte, opts ...OpenOption) (*Engine, error) {
	return openEngine(ctx, func(env *Environment) (*Model, error) {
		return OpenModelFromBuffer(env, data)
	}, opts...)
}

func openEngine(ctx context.Context, loadModel func(*Environment) (*Model, error), opts ...OpenOption) (*Engine, error) {
	if ctx != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
	}

	cfg := openConfig{accel: AccelCPU}
	for _, o := range opts {
		if o != nil {
			o(&cfg)
		}
	}

	syms, err := ensureRuntime(ctx, &cfg)
	if err != nil {
		return nil, err
	}

	envOpts := buildEnvOptions(&cfg)
	env, err := newEnvironmentWithSymbols(syms, envOpts...)
	if err != nil {
		return nil, err
	}

	model, err := loadModel(env)
	if err != nil {
		_ = env.Close()
		return nil, err
	}

	compileOpts, err := buildCompileOptions(syms, &cfg)
	if err != nil {
		_ = model.Close()
		_ = env.Close()
		return nil, err
	}

	cm, compileErr := Compile(env, model, compileOpts)
	_ = compileOpts.Close()
	if compileErr != nil {
		_ = model.Close()
		_ = env.Close()
		return nil, compileErr
	}

	sigs, err := model.Signatures()
	if err != nil {
		_ = cm.Close()
		_ = model.Close()
		_ = env.Close()
		return nil, err
	}
	defaultSig, err := model.DefaultSignature()
	if err != nil {
		_ = cm.Close()
		_ = model.Close()
		_ = env.Close()
		return nil, err
	}

	return &Engine{
		env:        env,
		model:      model,
		cm:         cm,
		sigs:       sigs,
		defaultSig: defaultSig,
	}, nil
}

func ensureRuntime(ctx context.Context, cfg *openConfig) (*capi.Symbols, error) {
	syms, err := ensureInit(cfg.libDir)
	if err == nil {
		return syms, nil
	}
	if !cfg.autoFetch || cfg.libDir != "" {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	dir, fetchErr := libfetch.Fetch(ctx, cfg.fetchOpts...)
	if fetchErr != nil {
		return nil, &Error{Op: "Open", Sentinel: ErrDynamicLoading, Cause: fetchErr}
	}
	return ensureInit(dir)
}

func buildEnvOptions(cfg *openConfig) []EnvOption {
	var all []EnvOption
	all = append(all, cfg.envOpts...)
	if cfg.hasLogSeverity {
		all = append(all, EnvIntOption(EnvMinLoggerSeverity, int64(cfg.logSeverity)))
	}
	if cfg.cacheDir != "" {
		all = append(all, EnvStringOption(EnvCompilerCacheDir, cfg.cacheDir))
	}
	return all
}

func buildCompileOptions(syms *capi.Symbols, cfg *openConfig) (*Options, error) {
	o, err := newOptionsWithSymbols(syms, cfg.accel)
	if err != nil {
		return nil, err
	}
	if cfg.hasCPU {
		if addErr := o.AddCPUOptions(cfg.cpuOpts); addErr != nil {
			_ = o.Close()
			return nil, addErr
		}
	}
	if cfg.hasGPU {
		if addErr := o.AddGPUOptions(cfg.gpuOpts); addErr != nil {
			_ = o.Close()
			return nil, addErr
		}
	}
	if cfg.hasRuntime {
		if addErr := o.AddRuntimeOptions(cfg.runtimeOpts); addErr != nil {
			_ = o.Close()
			return nil, addErr
		}
	}
	return o, nil
}

func (e *Engine) valid() bool {
	return e != nil && !e.closed && e.cm.valid() && e.model.valid() && e.env.valid()
}

// Close destroys all owned runtime resources in LIFO order.
func (e *Engine) Close() error {
	if e == nil || e.closed {
		return nil
	}
	e.closed = true
	return errors.Join(
		e.cm.Close(),
		e.model.Close(),
		e.env.Close(),
	)
}

// DefaultSignature returns the model's primary signature or main subgraph metadata.
func (e *Engine) DefaultSignature() *Signature {
	if !e.valid() {
		return nil
	}
	return e.defaultSig
}

// Signatures returns the list of signatures exposed by the model.
func (e *Engine) Signatures() []*Signature {
	if !e.valid() {
		return nil
	}
	out := make([]*Signature, len(e.sigs))
	copy(out, e.sigs)
	return out
}

// Signature looks up a signature by key.
func (e *Engine) Signature(key string) (*Signature, error) {
	if !e.valid() {
		return nil, ErrClosed
	}
	for _, s := range e.sigs {
		if s.Key() == key {
			return s, nil
		}
	}
	if e.defaultSig != nil && e.defaultSig.Key() == key {
		return e.defaultSig, nil
	}
	return nil, &Error{Op: "Engine.Signature", Sentinel: ErrNotFound, Msg: fmt.Sprintf("signature %q not found", key)}
}

// Metadata reads an embedded model metadata entry by key.
func (e *Engine) Metadata(key string) ([]byte, error) {
	if !e.valid() {
		return nil, ErrClosed
	}
	return e.model.Metadata(key)
}

// Model returns the underlying loaded Model, or nil if closed.
func (e *Engine) Model() *Model {
	if !e.valid() {
		return nil
	}
	return e.model
}

// CompiledModel returns the underlying CompiledModel, or nil if closed.
func (e *Engine) CompiledModel() *CompiledModel {
	if !e.valid() {
		return nil
	}
	return e.cm
}

// Environment returns the underlying Environment, or nil if closed.
func (e *Engine) Environment() *Environment {
	if !e.valid() {
		return nil
	}
	return e.env
}

func (e *Engine) resolveSignature(op string, sel sigSelector) (*Signature, error) {
	if sel.hasKey && sel.hasIndex {
		return nil, &Error{Op: op, Sentinel: ErrInvalidArgument, Msg: "cannot specify both signature key and index"}
	}
	if sel.hasKey {
		sig, err := e.Signature(sel.key)
		if err != nil {
			return nil, err
		}
		return sig, nil
	}
	if sel.hasIndex {
		if sel.index < 0 || (len(e.sigs) > 0 && sel.index >= len(e.sigs)) || (len(e.sigs) == 0 && (sel.index != 0 || e.defaultSig == nil)) {
			return nil, &Error{Op: op, Sentinel: ErrIndexOOB, Msg: fmt.Sprintf("signature index %d out of range", sel.index)}
		}
		if len(e.sigs) == 0 {
			return e.defaultSig, nil
		}
		return e.sigs[sel.index], nil
	}
	if e.defaultSig == nil {
		return nil, &Error{Op: op, Sentinel: ErrNotFound, Msg: "model has no default signature"}
	}
	return e.defaultSig, nil
}

// Run executes inference once against named inputs and returns decoded typed
// output slices keyed by output tensor name.
func (e *Engine) Run(ctx context.Context, inputs NamedInputs, opts ...RunOption) (NamedOutputs, error) {
	if !e.valid() {
		return nil, ErrClosed
	}
	var cfg runConfig
	for _, o := range opts {
		if o != nil {
			o(&cfg)
		}
	}
	sig, err := e.resolveSignature("Engine.Run", cfg.sig)
	if err != nil {
		return nil, err
	}

	inSpecs := sig.Inputs()
	outSpecs := sig.Outputs()

	expectedInputs := make(map[string]struct{}, len(inSpecs))
	for _, sp := range inSpecs {
		expectedInputs[sp.Name] = struct{}{}
	}
	for name := range inputs {
		if _, ok := expectedInputs[name]; !ok {
			return nil, &Error{
				Op:       "Engine.Run",
				Sentinel: ErrInvalidArgument,
				Msg:      fmt.Sprintf("unknown input tensor %q", name),
			}
		}
	}

	var owned []*TensorBuffer
	defer func() { closeBuffers(owned) }()

	inBufs := make([]*TensorBuffer, len(inSpecs))
	for i, sp := range inSpecs {
		val, ok := inputs[sp.Name]
		if !ok {
			return nil, &Error{
				Op:       "Engine.Run",
				Sentinel: ErrInvalidArgument,
				Msg:      fmt.Sprintf("missing input tensor %q", sp.Name),
			}
		}
		if tb, isBuf := val.(*TensorBuffer); isBuf {
			if !tb.valid() {
				return nil, ErrClosed
			}
			inBufs[i] = tb
			continue
		}
		b, createErr := e.cm.CreateInputBuffer(sig.Index(), sp, i)
		if createErr != nil {
			return nil, createErr
		}
		owned = append(owned, b)
		if wErr := writeNamedInput(b, sp.Name, val); wErr != nil {
			return nil, wErr
		}
		inBufs[i] = b
	}

	outBufs := make([]*TensorBuffer, len(outSpecs))
	for i, sp := range outSpecs {
		b, createErr := e.cm.CreateOutputBuffer(sig.Index(), sp, i)
		if createErr != nil {
			return nil, createErr
		}
		owned = append(owned, b)
		outBufs[i] = b
	}

	if runErr := e.cm.Run(ctx, sig.Index(), inBufs, outBufs); runErr != nil {
		return nil, runErr
	}

	outputs := make(NamedOutputs, len(outSpecs))
	for i, sp := range outSpecs {
		val, readErr := readTypedOutput(outBufs[i], sp)
		if readErr != nil {
			return nil, readErr
		}
		outputs[sp.Name] = val
	}
	return outputs, nil
}

// NewRunner pre-allocates owned input and output TensorBuffers for repeated
// execution with zero steady-state heap allocations.
func (e *Engine) NewRunner(opts ...RunnerOption) (*Runner, error) {
	if !e.valid() {
		return nil, ErrClosed
	}
	var cfg runnerConfig
	for _, o := range opts {
		if o != nil {
			o(&cfg)
		}
	}
	sig, err := e.resolveSignature("Engine.NewRunner", cfg.sig)
	if err != nil {
		return nil, err
	}

	inSpecs := sig.Inputs()
	outSpecs := sig.Outputs()

	if len(cfg.inputShapes) > 0 {
		inIdxByName := make(map[string]int, len(inSpecs))
		for i, sp := range inSpecs {
			inIdxByName[sp.Name] = i
		}
		for name, dims := range cfg.inputShapes {
			idx, ok := inIdxByName[name]
			if !ok {
				return nil, &Error{
					Op:       "Engine.NewRunner",
					Sentinel: ErrInvalidArgument,
					Msg:      fmt.Sprintf("unknown input tensor %q in WithInputShapes", name),
				}
			}
			if rErr := e.cm.ResizeInputNonStrict(sig.Index(), idx, dims); rErr != nil {
				return nil, rErr
			}
			inSpecs[idx].Type.Shape = slices.Clone(dims)
		}
		if len(outSpecs) > 0 {
			layouts, lErr := e.cm.OutputLayouts(sig.Index(), len(outSpecs), true)
			if lErr != nil {
				return nil, lErr
			}
			for i := range outSpecs {
				outSpecs[i].Type.Shape = layouts[i].Dimensions
				outSpecs[i].Type.HasStrides = layouts[i].HasStrides
				outSpecs[i].Type.Strides = layouts[i].Strides
			}
		}
	}

	var allocated []*TensorBuffer
	inBufs := make([]*TensorBuffer, len(inSpecs))
	inNames := make([]string, len(inSpecs))
	for i, sp := range inSpecs {
		b, createErr := e.cm.CreateInputBuffer(sig.Index(), sp, i)
		if createErr != nil {
			closeBuffers(allocated)
			return nil, createErr
		}
		allocated = append(allocated, b)
		inBufs[i] = b
		inNames[i] = sp.Name
	}

	outBufs := make([]*TensorBuffer, len(outSpecs))
	outNames := make([]string, len(outSpecs))
	for i, sp := range outSpecs {
		b, createErr := e.cm.CreateOutputBuffer(sig.Index(), sp, i)
		if createErr != nil {
			closeBuffers(allocated)
			return nil, createErr
		}
		allocated = append(allocated, b)
		outBufs[i] = b
		outNames[i] = sp.Name
	}

	r, err := NewRunner(e.cm, sig.Index(), inBufs, outBufs)
	if err != nil {
		closeBuffers(allocated)
		return nil, err
	}
	r.ownedBufs = true
	r.setTensorNames(inNames, outNames)
	return r, nil
}

func closeBuffers(bufs []*TensorBuffer) {
	for _, b := range bufs {
		_ = b.Close()
	}
}

func writeNamedInput(buf *TensorBuffer, name string, val any) error {
	switch v := val.(type) {
	case []float32:
		return WriteSlice(buf, v)
	case []float64:
		return WriteSlice(buf, v)
	case []uint8:
		return buf.WriteBytes(v)
	case []int8:
		return WriteSlice(buf, v)
	case []int16:
		return WriteSlice(buf, v)
	case []uint16:
		return WriteSlice(buf, v)
	case []int32:
		return WriteSlice(buf, v)
	case []uint32:
		return WriteSlice(buf, v)
	case []int64:
		return WriteSlice(buf, v)
	case []uint64:
		return WriteSlice(buf, v)
	case []bool:
		return WriteSlice(buf, v)
	default:
		return &Error{
			Op:       "Engine.Run",
			Sentinel: ErrInvalidArgument,
			Msg:      fmt.Sprintf("unsupported input slice type %T for %q", val, name),
		}
	}
}

func readTypedOutput(buf *TensorBuffer, sp TensorSpec) (any, error) {
	switch sp.Type.ElementType {
	case ElementFloat32:
		return ReadSlice[float32](buf)
	case ElementFloat64:
		return ReadSlice[float64](buf)
	case ElementUInt8:
		return ReadSlice[uint8](buf)
	case ElementInt8:
		return ReadSlice[int8](buf)
	case ElementInt16:
		return ReadSlice[int16](buf)
	case ElementUInt16:
		return ReadSlice[uint16](buf)
	case ElementInt32:
		return ReadSlice[int32](buf)
	case ElementUInt32:
		return ReadSlice[uint32](buf)
	case ElementInt64:
		return ReadSlice[int64](buf)
	case ElementUInt64:
		return ReadSlice[uint64](buf)
	case ElementBool:
		return ReadSlice[bool](buf)
	default:
		return buf.ReadAllBytes()
	}
}
