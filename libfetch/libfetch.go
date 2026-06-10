// Package libfetch downloads the LiteRT runtime libraries litert-go loads at
// runtime: libLiteRt and the platform's accelerator from the LiteRT prebuilt
// release bucket, plus — on Windows — the DirectX Shader Compiler the WebGPU
// accelerator requires. The result directory is usable as litert.Load's dir
// argument or the LITERT_LIB environment variable.
//
// Fetch is idempotent: files already present with the published checksum are
// not downloaded again.
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
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// DefaultVersion is the LiteRT prebuilt release litert-go is validated
// against. The bucket also serves "latest", an alias for the newest release.
const DefaultVersion = "2.1.5"

const bucketURL = "https://storage.googleapis.com/litert"

// The WebGPU accelerator compiles shaders through the DirectX Shader Compiler
// on Windows; Microsoft distributes it separately from LiteRT.
const (
	dxcURL    = "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2602.24/dxc_2026_05_27.zip"
	dxcSHA256 = "cf658aacf070d3045e31b8f1f8a696c2945f37c1095019481ef7c513368db3b4"
)

type config struct {
	version  string
	dir      string
	platform string
	logf     func(format string, args ...any)
}

// Option configures Fetch.
type Option func(*config)

// WithVersion selects the prebuilt release (e.g. "2.1.5", "latest").
func WithVersion(v string) Option { return func(c *config) { c.version = v } }

// WithDir sets the destination directory. The default is
// <user cache dir>/litert-go/lib/<version>/<platform>.
func WithDir(dir string) Option { return func(c *config) { c.dir = dir } }

// WithPlatform overrides the bucket platform name (e.g. "linux_x86_64") for
// fetching libraries for a machine other than the current one.
func WithPlatform(p string) Option { return func(c *config) { c.platform = p } }

// WithLogf reports per-file progress. The default discards it.
func WithLogf(f func(format string, args ...any)) Option {
	return func(c *config) { c.logf = f }
}

// Platform returns the prebuilt bucket's platform name for the current
// GOOS/GOARCH.
func Platform() (string, error) {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "windows/amd64":
		return "windows_x86_64", nil
	case "linux/amd64":
		return "linux_x86_64", nil
	case "linux/arm64":
		return "linux_arm64", nil
	case "darwin/arm64":
		return "macos_arm64", nil
	case "android/arm64":
		return "android_arm64", nil
	}
	return "", fmt.Errorf("libfetch: no LiteRT prebuilts for %s/%s", runtime.GOOS, runtime.GOARCH)
}

// Fetch downloads the runtime libraries and returns the directory holding
// them.
func Fetch(ctx context.Context, opts ...Option) (string, error) {
	c := config{version: DefaultVersion, logf: func(string, ...any) {}}
	for _, o := range opts {
		o(&c)
	}
	if c.platform == "" {
		p, err := Platform()
		if err != nil {
			return "", err
		}
		c.platform = p
	}
	if c.dir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return "", fmt.Errorf("libfetch: no cache dir: %w", err)
		}
		c.dir = filepath.Join(base, "litert-go", "lib", c.version, c.platform)
	}
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return "", err
	}

	files, err := listRelease(ctx, c.version, c.platform)
	if err != nil {
		return "", err
	}
	libs := 0
	for _, f := range files {
		if !strings.HasPrefix(f.base(), "libLiteRt") {
			continue
		}
		libs++
		if err := fetchObject(ctx, &c, f); err != nil {
			return "", err
		}
	}
	if libs == 0 {
		return "", fmt.Errorf("libfetch: no libLiteRt libraries under binaries/%s/%s", c.version, c.platform)
	}

	if c.platform == "windows_x86_64" {
		if err := fetchDXC(ctx, &c); err != nil {
			return "", err
		}
	}
	return c.dir, nil
}

// object is one bucket entry from the GCS JSON listing.
type object struct {
	Name    string `json:"name"`
	Size    string `json:"size"`
	MD5Hash string `json:"md5Hash"` // base64
}

func (o object) base() string { return o.Name[strings.LastIndexByte(o.Name, '/')+1:] }

// listRelease enumerates binaries/<version>/<platform>/ in the release
// bucket.
func listRelease(ctx context.Context, version, platform string) ([]object, error) {
	url := fmt.Sprintf("%s/storage/v1/b/litert/o?prefix=binaries/%s/%s/",
		"https://storage.googleapis.com", version, platform)
	body, err := get(ctx, url)
	if err != nil {
		return nil, err
	}
	var listing struct {
		Items []object `json:"items"`
	}
	if err := json.Unmarshal(body, &listing); err != nil {
		return nil, fmt.Errorf("libfetch: bucket listing: %w", err)
	}
	if len(listing.Items) == 0 {
		return nil, fmt.Errorf("libfetch: no prebuilts under binaries/%s/%s (unknown version or platform)", version, platform)
	}
	return listing.Items, nil
}

// fetchObject downloads one bucket object into c.dir, verifying its MD5
// against the listing. A present file that already matches is kept.
func fetchObject(ctx context.Context, c *config, f object) error {
	want, err := base64.StdEncoding.DecodeString(f.MD5Hash)
	if err != nil {
		return fmt.Errorf("libfetch: %s: bad md5 in listing: %w", f.base(), err)
	}
	dst := filepath.Join(c.dir, f.base())
	if fileMD5Matches(dst, want) {
		c.logf("%s: up to date", f.base())
		return nil
	}
	c.logf("%s: downloading", f.base())
	data, err := get(ctx, bucketURL+"/"+f.Name)
	if err != nil {
		return err
	}
	sum := md5.Sum(data)
	if !bytes.Equal(sum[:], want) {
		return fmt.Errorf("libfetch: %s: md5 mismatch after download", f.base())
	}
	return writeFile(dst, data)
}

// fetchDXC extracts dxcompiler.dll and dxil.dll from the pinned DirectX
// Shader Compiler release.
func fetchDXC(ctx context.Context, c *config) error {
	need := []string{"dxcompiler.dll", "dxil.dll"}
	missing := false
	for _, n := range need {
		if _, err := os.Stat(filepath.Join(c.dir, n)); err != nil {
			missing = true
		}
	}
	if !missing {
		c.logf("dxc: up to date")
		return nil
	}
	c.logf("dxc: downloading %s", dxcURL)
	data, err := get(ctx, dxcURL)
	if err != nil {
		return err
	}
	if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != dxcSHA256 {
		return fmt.Errorf("libfetch: dxc archive sha256 mismatch")
	}
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return fmt.Errorf("libfetch: dxc archive: %w", err)
	}
	for _, n := range need {
		entry := "bin/x64/" + n
		zf, err := zr.Open(entry)
		if err != nil {
			return fmt.Errorf("libfetch: dxc archive has no %s: %w", entry, err)
		}
		content, err := io.ReadAll(zf)
		zf.Close()
		if err != nil {
			return err
		}
		if err := writeFile(filepath.Join(c.dir, n), content); err != nil {
			return err
		}
	}
	return nil
}

func get(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("libfetch: GET %s: %s", url, resp.Status)
	}
	return io.ReadAll(resp.Body)
}

func fileMD5Matches(path string, want []byte) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	sum := md5.Sum(data)
	return bytes.Equal(sum[:], want)
}

// writeFile writes through a temp file and renames, so an interrupted fetch
// never leaves a truncated library behind.
func writeFile(dst string, data []byte) error {
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, data, 0o755); err != nil {
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}
