package libfetch

import (
	"crypto/md5"
	"os"
	"path/filepath"
	"testing"
)

func TestPlatform(t *testing.T) {
	p, err := Platform()
	if err != nil {
		t.Skipf("unsupported host: %v", err)
	}
	switch p {
	case "windows_x86_64", "linux_x86_64", "linux_arm64", "macos_arm64", "android_arm64":
	default:
		t.Fatalf("unexpected platform %q", p)
	}
}

func TestObjectBase(t *testing.T) {
	o := object{Name: "binaries/2.1.5/windows_x86_64/libLiteRt.dll"}
	if got := o.base(); got != "libLiteRt.dll" {
		t.Fatalf("base = %q", got)
	}
}

func TestFileMD5Matches(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f")
	data := []byte("hello")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	sum := md5.Sum(data)
	if !fileMD5Matches(path, sum[:]) {
		t.Fatal("matching file reported as mismatch")
	}
	if fileMD5Matches(path, make([]byte, md5.Size)) {
		t.Fatal("mismatched digest reported as match")
	}
	if fileMD5Matches(filepath.Join(dir, "absent"), sum[:]) {
		t.Fatal("missing file reported as match")
	}
}

func TestWriteFileAtomic(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "lib.dll")
	if err := writeFile(dst, []byte("v1")); err != nil {
		t.Fatal(err)
	}
	if err := writeFile(dst, []byte("v2")); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "v2" {
		t.Fatalf("content = %q", got)
	}
	if _, err := os.Stat(dst + ".tmp"); !os.IsNotExist(err) {
		t.Fatal("temp file left behind")
	}
}
