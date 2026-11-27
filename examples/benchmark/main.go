// Command benchmark measures steady-state inference latency using a
// pre-allocated litert.Runner and reports accelerator delegation metrics.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/vladimirvivien/litert-go/pkg/litert"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "benchmark: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to .tflite model file")
	warmup := flag.Int("warmup", 5, "number of warm-up runs")
	iters := flag.Int("iters", 50, "number of timed inference runs")
	threads := flag.Int("threads", 4, "CPU thread count")
	useGPU := flag.Bool("gpu", false, "enable GPU accelerator with CPU fallback")
	libDir := flag.String("lib", "", "directory containing libLiteRt shared library")
	flag.Parse()

	if *modelPath == "" || *iters <= 0 {
		return fmt.Errorf("usage: benchmark -model model.tflite [-warmup 5 -iters 50 -threads 4 -gpu]")
	}

	accel := litert.AccelCPU
	if *useGPU {
		accel |= litert.AccelGPU
	}
	opts := []litert.OpenOption{
		litert.WithAccelerator(accel),
		litert.WithCPUOptions(litert.CPUOptions{NumThreads: *threads}),
		litert.WithLogSeverity(litert.LogWarning),
	}
	if *libDir != "" {
		opts = append(opts, litert.WithLibDir(*libDir))
	} else {
		opts = append(opts, litert.WithAutoFetch())
	}

	ctx := context.Background()
	eng, err := litert.Open(ctx, *modelPath, opts...)
	if err != nil {
		return err
	}
	defer func() { _ = eng.Close() }()

	runner, err := eng.NewRunner()
	if err != nil {
		return err
	}
	defer func() { _ = runner.Close() }()

	for i := range runner.NumInputs() {
		if cErr := runner.Input(i).Clear(); cErr != nil {
			sz, _ := runner.Input(i).PackedSize()
			_ = runner.Input(i).WriteBytes(make([]byte, sz))
		}
	}

	for range *warmup {
		if rErr := runner.Run(ctx); rErr != nil {
			return fmt.Errorf("warmup: %w", rErr)
		}
	}

	samples := make([]time.Duration, *iters)
	var total time.Duration
	for i := range *iters {
		t0 := time.Now()
		if rErr := runner.Run(ctx); rErr != nil {
			return fmt.Errorf("run[%d]: %w", i, rErr)
		}
		d := time.Since(t0)
		samples[i] = d
		total += d
	}

	slices.Sort(samples)
	mean := total / time.Duration(*iters)
	p50 := samples[len(samples)/2]
	p95 := samples[(len(samples)*95)/100]

	fmt.Printf("Iterations: %d (warmup %d)\n", *iters, *warmup)
	fmt.Printf("Latency:    min=%v p50=%v mean=%v p95=%v max=%v\n",
		samples[0], p50, mean, p95, samples[len(samples)-1])
	fmt.Printf("Throughput: %.1f inf/sec\n", float64(*iters)/total.Seconds())

	if full, fErr := eng.CompiledModel().FullyAccelerated(); fErr == nil {
		fmt.Printf("FullyAccelerated: %v\n", full)
	}
	if dm, dErr := eng.CompiledModel().DelegationMetrics(); dErr == nil {
		fmt.Printf("DelegatedNodes: total=%d accelerators=%d\n", dm.TotalNodeCount, len(dm.Accelerators))
	}
	return nil
}
