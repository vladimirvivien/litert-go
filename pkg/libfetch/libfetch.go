// Package libfetch downloads the LiteRT prebuilt shared libraries for the
// target platform (libLiteRt, accelerators, and on Windows x64, the DirectX
// Shader Compiler DLLs) using streaming I/O and cryptographic verification.
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
	"sync"
	"time"
)

var (
	dirLocksMu sync.Mutex
	dirLocks   = make(map[string]*sync.Mutex)
)

func lockDir(dir string) func() {
	key, err := filepath.Abs(dir)
	if err != nil {
		key = filepath.Clean(dir)
	}
	dirLocksMu.Lock()
	mu, ok := dirLocks[key]
	if !ok {
		mu = &sync.Mutex{}
		dirLocks[key] = mu
	}
	dirLocksMu.Unlock()

	mu.Lock()
	return mu.Unlock
}

// DefaultVersion is the LiteRT prebuilt release targeted by litert-go.
const DefaultVersion = "2.4.1"

const (
	defaultBucketBaseURL  = "https://storage.googleapis.com/litert"
	defaultStorageAPIBase = "https://storage.googleapis.com"
	defaultDXCURL         = "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2602.24/dxc_2026_05_27.zip"
	defaultDXCSHA256      = "cf658aacf070d3045e31b8f1f8a696c2945f37c1095019481ef7c513368db3b4"
	dxcManifestName       = ".dxc.sha256"
)

type config struct {
	version        string
	dir            string
	platform       string
	bucketBaseURL  string
	storageAPIBase string
	dxcURL         string
	dxcSHA256      string
	includeDXC     bool
	client         *http.Client
	logf           func(format string, args ...any)
}

// Option configures Fetch.
type Option func(*config)

// WithVersion selects the LiteRT release version (e.g. "2.4.1" or "latest").
func WithVersion(v string) Option { return func(c *config) { c.version = v } }

// WithDir sets the output directory for downloaded libraries.
func WithDir(dir string) Option { return func(c *config) { c.dir = dir } }

// WithPlatform overrides the target bucket platform (e.g. "linux_x86_64").
func WithPlatform(p string) Option { return func(c *config) { c.platform = p } }

// WithHTTPClient supplies a custom HTTP client for downloads.
func WithHTTPClient(hc *http.Client) Option { return func(c *config) { c.client = hc } }

// WithLogf configures progress logging.
func WithLogf(fn func(format string, args ...any)) Option {
	return func(c *config) { c.logf = fn }
}

// WithIncludeDXC toggles downloading DirectX Shader Compiler DLLs on
// windows_x86_64 (enabled by default).
func WithIncludeDXC(include bool) Option {
	return func(c *config) { c.includeDXC = include }
}

// DefaultDir returns the default cache directory for the host platform and
// DefaultVersion without touching the network.
func DefaultDir() (string, error) {
	p, err := Platform()
	if err != nil {
		return "", err
	}
	return DefaultDirFor(DefaultVersion, p)
}

// DefaultDirFor returns the default cache directory for version and platform.
func DefaultDirFor(version, platform string) (string, error) {
	if version == "" {
		version = DefaultVersion
	}
	if platform == "" {
		var err error
		platform, err = Platform()
		if err != nil {
			return "", err
		}
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("libfetch: cache dir: %w", err)
	}
	return filepath.Join(base, "litert-go", "lib", version, platform), nil
}

// Platform returns the LiteRT release bucket platform name for the current OS
// and architecture.
func Platform() (string, error) {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "windows/amd64":
		return "windows_x86_64", nil
	case "windows/arm64":
		return "windows_arm64", nil
	case "linux/amd64":
		return "linux_x86_64", nil
	case "linux/arm64":
		return "linux_arm64", nil
	case "darwin/arm64":
		return "macos_arm64", nil
	case "android/arm64":
		return "android_arm64", nil
	default:
		return "", fmt.Errorf("libfetch: unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

func defaultHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			TLSHandshakeTimeout:   15 * time.Second,
			ResponseHeaderTimeout: 30 * time.Second,
			IdleConnTimeout:       60 * time.Second,
		},
	}
}

// Fetch downloads and verifies the LiteRT shared libraries into c.dir (or the
// default cache directory) and returns the resulting directory path.
func Fetch(ctx context.Context, opts ...Option) (string, error) {
	c := config{
		version:        DefaultVersion,
		bucketBaseURL:  defaultBucketBaseURL,
		storageAPIBase: defaultStorageAPIBase,
		dxcURL:         defaultDXCURL,
		dxcSHA256:      defaultDXCSHA256,
		includeDXC:     true,
		logf:           func(string, ...any) {},
	}
	for _, o := range opts {
		o(&c)
	}
	if c.client == nil {
		c.client = defaultHTTPClient()
	}
	if c.platform == "" {
		p, err := Platform()
		if err != nil {
			return "", err
		}
		c.platform = p
	}
	if c.dir == "" {
		d, err := DefaultDirFor(c.version, c.platform)
		if err != nil {
			return "", err
		}
		c.dir = d
	}
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return "", fmt.Errorf("libfetch: mkdir %q: %w", c.dir, err)
	}
	unlock := lockDir(c.dir)
	defer unlock()

	prefix := fmt.Sprintf("binaries/%s/%s/", c.version, c.platform)
	items, err := listRelease(ctx, &c, prefix)
	if err != nil {
		return "", err
	}

	libs := 0
	for _, f := range items {
		rel := strings.TrimPrefix(f.Name, prefix)
		if rel == "" || strings.ContainsRune(rel, '/') || !isSharedLib(rel) {
			continue
		}
		libs++
		if err := fetchObject(ctx, &c, rel, f); err != nil {
			return "", err
		}
	}
	if libs == 0 {
		return "", fmt.Errorf("libfetch: no LiteRT shared libraries under %s", prefix)
	}

	if c.includeDXC && c.platform == "windows_x86_64" {
		if err := fetchDXC(ctx, &c); err != nil {
			return "", err
		}
	}
	return c.dir, nil
}

func isSharedLib(name string) bool {
	if !strings.HasPrefix(name, "libLiteRt") && !strings.HasPrefix(name, "LiteRt") {
		return false
	}
	ext := filepath.Ext(name)
	return ext == ".so" || ext == ".dylib" || ext == ".dll" || strings.Contains(name, ".so.")
}

type bucketObject struct {
	Name    string `json:"name"`
	Size    string `json:"size"`
	MD5Hash string `json:"md5Hash"`
}

func listRelease(ctx context.Context, c *config, prefix string) ([]bucketObject, error) {
	listURL := fmt.Sprintf("%s/storage/v1/b/litert/o?prefix=%s", strings.TrimRight(c.storageAPIBase, "/"), prefix)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, listURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("libfetch: list %s: %w", listURL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("libfetch: GET %s: %s", listURL, resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, fmt.Errorf("libfetch: read listing: %w", err)
	}
	var listing struct {
		Items []bucketObject `json:"items"`
	}
	if err := json.Unmarshal(body, &listing); err != nil {
		return nil, fmt.Errorf("libfetch: decode listing: %w", err)
	}
	if len(listing.Items) == 0 {
		return nil, fmt.Errorf("libfetch: no artifacts under %s", prefix)
	}
	return listing.Items, nil
}

func fetchObject(ctx context.Context, c *config, name string, f bucketObject) error {
	want, err := base64.StdEncoding.DecodeString(f.MD5Hash)
	if err != nil {
		return fmt.Errorf("libfetch: %s: decode md5: %w", name, err)
	}
	dst := filepath.Join(c.dir, name)
	if ok, matchErr := fileMD5Matches(dst, want); matchErr == nil && ok {
		c.logf("%s: up to date", name)
		return nil
	}

	c.logf("%s: downloading", name)
	objURL := strings.TrimRight(c.bucketBaseURL, "/") + "/" + strings.TrimLeft(f.Name, "/")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, objURL, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("libfetch: GET %s: %w", objURL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("libfetch: GET %s: %s", objURL, resp.Status)
	}

	sum, err := streamToAtomicFile(c.dir, dst, resp.Body, md5.New(), 0o755)
	if err != nil {
		return fmt.Errorf("libfetch: %s: %w", name, err)
	}
	if !bytes.Equal(sum, want) {
		_ = os.Remove(dst)
		return fmt.Errorf("libfetch: %s: md5 mismatch", name)
	}
	return nil
}

func fetchDXC(ctx context.Context, c *config) error {
	need := []string{"dxcompiler.dll", "dxil.dll"}
	if dxcCacheValid(c.dir, c.dxcSHA256, need) {
		c.logf("dxc: up to date")
		return nil
	}

	c.logf("dxc: downloading %s", c.dxcURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.dxcURL, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("libfetch: dxc GET: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("libfetch: dxc GET %s: %s", c.dxcURL, resp.Status)
	}

	tmpZip, err := os.CreateTemp(c.dir, ".dxc-*.zip")
	if err != nil {
		return fmt.Errorf("libfetch: dxc temp file: %w", err)
	}
	tmpZipPath := tmpZip.Name()
	defer func() { _ = os.Remove(tmpZipPath) }()

	h := sha256.New()
	if _, copyErr := io.Copy(io.MultiWriter(tmpZip, h), resp.Body); copyErr != nil {
		_ = tmpZip.Close()
		return fmt.Errorf("libfetch: dxc stream: %w", copyErr)
	}
	if closeErr := tmpZip.Close(); closeErr != nil {
		return fmt.Errorf("libfetch: dxc close temp: %w", closeErr)
	}
	if hex.EncodeToString(h.Sum(nil)) != strings.ToLower(c.dxcSHA256) {
		return fmt.Errorf("libfetch: dxc archive sha256 mismatch")
	}

	zr, err := zip.OpenReader(tmpZipPath)
	if err != nil {
		return fmt.Errorf("libfetch: dxc open zip: %w", err)
	}
	defer func() { _ = zr.Close() }()

	var manifest strings.Builder
	fmt.Fprintf(&manifest, "archive=%s\n", strings.ToLower(c.dxcSHA256))
	for _, n := range need {
		entry := "bin/x64/" + n
		rc, err := zr.Open(entry)
		if err != nil {
			return fmt.Errorf("libfetch: dxc missing %s: %w", entry, err)
		}
		dst := filepath.Join(c.dir, n)
		dllSum, copyErr := streamToAtomicFile(c.dir, dst, rc, sha256.New(), 0o755)
		_ = rc.Close()
		if copyErr != nil {
			return fmt.Errorf("libfetch: dxc extract %s: %w", n, copyErr)
		}
		fmt.Fprintf(&manifest, "%s=%s\n", n, hex.EncodeToString(dllSum))
	}

	manifestPath := filepath.Join(c.dir, dxcManifestName)
	if _, err := streamToAtomicFile(c.dir, manifestPath, strings.NewReader(manifest.String()), sha256.New(), 0o644); err != nil {
		return fmt.Errorf("libfetch: dxc manifest: %w", err)
	}
	return nil
}

func dxcCacheValid(dir, expectedArchiveSHA string, files []string) bool {
	data, err := os.ReadFile(filepath.Join(dir, dxcManifestName))
	if err != nil {
		return false
	}
	wantHashes := make(map[string]string, len(files)+1)
	for line := range strings.SplitSeq(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			return false
		}
		wantHashes[k] = v
	}
	if wantHashes["archive"] != strings.ToLower(expectedArchiveSHA) {
		return false
	}
	for _, f := range files {
		wantHex, ok := wantHashes[f]
		if !ok {
			return false
		}
		wantBytes, err := hex.DecodeString(wantHex)
		if err != nil {
			return false
		}
		match, err := fileSHA256Matches(filepath.Join(dir, f), wantBytes)
		if err != nil || !match {
			return false
		}
	}
	return true
}

func fileMD5Matches(path string, want []byte) (bool, error) {
	return fileHashMatches(path, md5.New(), want)
}

func fileSHA256Matches(path string, want []byte) (bool, error) {
	return fileHashMatches(path, sha256.New(), want)
}

func fileHashMatches(path string, h io.Writer, want []byte) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer func() { _ = f.Close() }()

	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	sum := h.(interface{ Sum([]byte) []byte }).Sum(nil)
	return bytes.Equal(sum, want), nil
}

func streamToAtomicFile(dir, dst string, src io.Reader, h io.Writer, mode os.FileMode) ([]byte, error) {
	tmp, err := os.CreateTemp(dir, ".litert-fetch-*.tmp")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	renamed := false
	defer func() {
		if !renamed {
			_ = os.Remove(tmpPath)
		}
	}()

	if _, err := io.Copy(io.MultiWriter(tmp, h), src); err != nil {
		_ = tmp.Close()
		return nil, err
	}
	if err := tmp.Chmod(mode); err != nil && runtime.GOOS != "windows" {
		_ = tmp.Close()
		return nil, err
	}
	if err := tmp.Close(); err != nil {
		return nil, err
	}
	if err := os.Rename(tmpPath, dst); err != nil {
		return nil, err
	}
	renamed = true
	return h.(interface{ Sum([]byte) []byte }).Sum(nil), nil
}
