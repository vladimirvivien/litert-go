package lm

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/vladimirvivien/litert-go/libfetch"
	"github.com/vladimirvivien/litert-go/litert"
)

// Option configures Open.
type Option func(*openConfig)

type openConfig struct {
	libDir       string
	accel        litert.HwAccelerator
	gpuCacheDir  string
	metrics      func(DecodeStats)
	fetchVersion string
}

// WithLibDir sets the directory holding libLiteRt and its accelerator
// libraries. When unset, Open resolves the directory from the LITERT_LIB
// environment variable, then from libfetch's default download location.
func WithLibDir(dir string) Option { return func(c *openConfig) { c.libDir = dir } }

// WithAccelerator selects the compilation backend. The default is CPU.
func WithAccelerator(a litert.HwAccelerator) Option {
	return func(c *openConfig) { c.accel = a }
}

// WithGPUCacheDir sets the directory for persisting compiled GPU programs
// across runs. The default is <user cache dir>/litert-go/gpu-cache.
func WithGPUCacheDir(dir string) Option { return func(c *openConfig) { c.gpuCacheDir = dir } }

// WithMetrics registers a callback that receives decode statistics after each
// generation or conversation turn. The callback runs on the calling goroutine.
func WithMetrics(f func(DecodeStats)) Option { return func(c *openConfig) { c.metrics = f } }

// WithFetch downloads the runtime libraries (libfetch.Fetch with the given
// version, e.g. libfetch.DefaultVersion) when no library directory resolves —
// the only case where Open touches the network. Without this option, Open
// never downloads.
func WithFetch(version string) Option { return func(c *openConfig) { c.fetchVersion = version } }

// DecodeStats describes one generation's decode loop.
type DecodeStats struct {
	Tokens int           // generated tokens
	Decode time.Duration // wall time of the decode loop
}

// TokensPerSecond returns the decode throughput, or 0 for an empty run.
func (s DecodeStats) TokensPerSecond() float64 {
	if s.Tokens == 0 || s.Decode <= 0 {
		return 0
	}
	return float64(s.Tokens) / s.Decode.Seconds()
}

// PerformanceMetrics describes the performance characteristics of a generation run or turn.
type PerformanceMetrics struct {
	PrefillDuration  time.Duration // Time spent on prompt evaluation (prefill stage)
	DecodeDuration   time.Duration // Time spent generating tokens (decode stage)
	TimeToFirstToken time.Duration // Time from prefill start to first decoded token
	PrefillTokens    int           // Number of tokens prefilled
	DecodeTokens     int           // Number of tokens generated
	TokensPerSecond  float64       // Decode throughput rate
	CacheHits        int           // KV cache reused tokens
}

// resolveLibDir picks the runtime-library directory: the explicit option,
// then LITERT_LIB, then libfetch's default download location if it exists,
// then — only with WithFetch — a fresh download.
func resolveLibDir(ctx context.Context, c *openConfig) (string, error) {
	if c.libDir != "" {
		return c.libDir, nil
	}
	if dir := os.Getenv(litert.EnvVar); dir != "" {
		return dir, nil
	}
	if dir, err := libfetch.DefaultDir(); err == nil {
		if _, serr := os.Stat(dir); serr == nil {
			return dir, nil
		}
	}
	if c.fetchVersion != "" {
		return libfetch.Fetch(ctx, libfetch.WithVersion(c.fetchVersion))
	}
	return "", fmt.Errorf("lm: no runtime libraries found: pass WithLibDir, set %s, or opt in to download with WithFetch", litert.EnvVar)
}
