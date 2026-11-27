package libfetch

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/md5"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

func TestPlatformAndDefaultDir(t *testing.T) {
	p, err := Platform()
	if err != nil {
		t.Skipf("unsupported host: %v", err)
	}
	switch p {
	case "windows_x86_64", "windows_arm64", "linux_x86_64", "linux_arm64", "macos_arm64", "android_arm64":
	default:
		t.Fatalf("unexpected platform %q", p)
	}

	dir, err := DefaultDir()
	if err != nil {
		t.Fatalf("DefaultDir: %v", err)
	}
	if filepath.Base(dir) != p {
		t.Fatalf("DefaultDir %q does not end with %q", dir, p)
	}
}

func TestIsSharedLibFilter(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"libLiteRt.so", true},
		{"libLiteRt.dylib", true},
		{"libLiteRt.dll", true},
		{"libLiteRtWebGpuAccelerator.so", true},
		{"LiteRtCompilerPlugin_IntelOpenvino.dll", true},
		{"benchmark_model.exe", false},
		{"run_model", false},
		{"other.dll", false},
	}
	for _, tc := range tests {
		if got := isSharedLib(tc.name); got != tc.want {
			t.Errorf("isSharedLib(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func buildTestDXCZip(t *testing.T, compilerData, dxilData []byte) ([]byte, string) {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, content := range map[string][]byte{
		"bin/x64/dxcompiler.dll": compilerData,
		"bin/x64/dxil.dll":       dxilData,
	} {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(content); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(buf.Bytes())
	return buf.Bytes(), hex.EncodeToString(sum[:])
}

func TestFetchStreamingAndCacheHit(t *testing.T) {
	libBytes := []byte("fake-litert-library")
	pluginBytes := []byte("fake-webgpu-plugin")
	libSum := md5.Sum(libBytes)
	pluginSum := md5.Sum(pluginBytes)
	dxcZip, dxcSHA := buildTestDXCZip(t, []byte("dxc-compiler"), []byte("dxc-il"))

	var libHits atomic.Int32
	var dxcHits atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/storage/v1/b/litert/o":
			items := []bucketObject{
				{Name: "binaries/2.4.1/windows_x86_64/libLiteRt.dll", MD5Hash: base64.StdEncoding.EncodeToString(libSum[:])},
				{Name: "binaries/2.4.1/windows_x86_64/libLiteRtWebGpuAccelerator.dll", MD5Hash: base64.StdEncoding.EncodeToString(pluginSum[:])},
				{Name: "binaries/2.4.1/windows_x86_64/benchmark_model.exe", MD5Hash: base64.StdEncoding.EncodeToString(libSum[:])},
				{Name: "binaries/2.4.1/windows_x86_64/nested/libLiteRtNested.dll", MD5Hash: base64.StdEncoding.EncodeToString(libSum[:])},
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"items": items})
		case "/binaries/2.4.1/windows_x86_64/libLiteRt.dll":
			libHits.Add(1)
			_, _ = w.Write(libBytes)
		case "/binaries/2.4.1/windows_x86_64/libLiteRtWebGpuAccelerator.dll":
			_, _ = w.Write(pluginBytes)
		case "/dxc.zip":
			dxcHits.Add(1)
			_, _ = w.Write(dxcZip)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	dstDir := t.TempDir()
	testOpts := []Option{
		WithVersion("2.4.1"),
		WithPlatform("windows_x86_64"),
		WithDir(dstDir),
		WithHTTPClient(srv.Client()),
		func(c *config) {
			c.bucketBaseURL = srv.URL
			c.storageAPIBase = srv.URL
			c.dxcURL = srv.URL + "/dxc.zip"
			c.dxcSHA256 = dxcSHA
		},
	}

	gotDir, err := Fetch(context.Background(), testOpts...)
	if err != nil {
		t.Fatalf("first Fetch: %v", err)
	}
	if gotDir != dstDir {
		t.Fatalf("gotDir = %q, want %q", gotDir, dstDir)
	}
	if _, statErr := os.Stat(filepath.Join(dstDir, "benchmark_model.exe")); !os.IsNotExist(statErr) {
		t.Fatal("non-library exe should not have been downloaded")
	}
	if _, statErr := os.Stat(filepath.Join(dstDir, "libLiteRtNested.dll")); !os.IsNotExist(statErr) {
		t.Fatal("nested object should not have been downloaded")
	}

	// Second Fetch should hit local checksum cache without re-downloading.
	if _, err = Fetch(context.Background(), testOpts...); err != nil {
		t.Fatalf("second Fetch: %v", err)
	}
	if libHits.Load() != 1 || dxcHits.Load() != 1 {
		t.Fatalf("expected 1 download each, got libHits=%d dxcHits=%d", libHits.Load(), dxcHits.Load())
	}

	// Corrupting dxil.dll must trigger re-verification failure and re-download.
	if err = os.WriteFile(filepath.Join(dstDir, "dxil.dll"), []byte("corrupted"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err = Fetch(context.Background(), testOpts...); err != nil {
		t.Fatalf("recovery Fetch: %v", err)
	}
	if dxcHits.Load() != 2 {
		t.Fatalf("expected DXC re-download after corruption, got dxcHits=%d", dxcHits.Load())
	}
	gotDXIL, err := os.ReadFile(filepath.Join(dstDir, "dxil.dll"))
	if err != nil || string(gotDXIL) != "dxc-il" {
		t.Fatalf("dxil.dll after recovery = %q, err=%v", gotDXIL, err)
	}
}

func TestFetchMD5MismatchAndCancellation(t *testing.T) {
	wantSum := md5.Sum([]byte("expected"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/storage/v1/b/litert/o":
			items := []bucketObject{
				{Name: "binaries/2.4.1/linux_x86_64/libLiteRt.so", MD5Hash: base64.StdEncoding.EncodeToString(wantSum[:])},
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"items": items})
		case "/binaries/2.4.1/linux_x86_64/libLiteRt.so":
			_, _ = w.Write([]byte("corrupted-payload"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	dstDir := t.TempDir()
	_, err := Fetch(context.Background(),
		WithVersion("2.4.1"),
		WithPlatform("linux_x86_64"),
		WithDir(dstDir),
		WithHTTPClient(srv.Client()),
		func(c *config) {
			c.bucketBaseURL = srv.URL
			c.storageAPIBase = srv.URL
		},
	)
	if err == nil {
		t.Fatal("expected md5 mismatch error")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = Fetch(ctx,
		WithVersion("2.4.1"),
		WithPlatform("linux_x86_64"),
		WithDir(dstDir),
		WithHTTPClient(srv.Client()),
		func(c *config) {
			c.bucketBaseURL = srv.URL
			c.storageAPIBase = srv.URL
		},
	)
	if err == nil {
		t.Fatal("expected cancellation error")
	}
}

func TestFetchConcurrentSafety(t *testing.T) {
	libBytes := []byte("concurrent-lib")
	libSum := md5.Sum(libBytes)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/storage/v1/b/litert/o":
			items := []bucketObject{
				{Name: "binaries/2.4.1/linux_x86_64/libLiteRt.so", MD5Hash: base64.StdEncoding.EncodeToString(libSum[:])},
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"items": items})
		case "/binaries/2.4.1/linux_x86_64/libLiteRt.so":
			_, _ = w.Write(libBytes)
		}
	}))
	defer srv.Close()

	dstDir := t.TempDir()
	var wg sync.WaitGroup
	errs := make(chan error, 6)
	for range 6 {
		wg.Go(func() {
			_, err := Fetch(context.Background(),
				WithVersion("2.4.1"),
				WithPlatform("linux_x86_64"),
				WithDir(dstDir),
				WithHTTPClient(srv.Client()),
				func(c *config) {
					c.bucketBaseURL = srv.URL
					c.storageAPIBase = srv.URL
				},
			)
			errs <- err
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Fetch error: %v", err)
		}
	}
}
