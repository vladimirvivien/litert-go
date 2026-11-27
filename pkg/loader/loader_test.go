package loader

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestLibraryFilename(t *testing.T) {
	got := LibraryFilename()
	switch runtime.GOOS {
	case "windows":
		if got != "libLiteRt.dll" {
			t.Fatalf("got %q, want libLiteRt.dll", got)
		}
	case "darwin":
		if got != "libLiteRt.dylib" {
			t.Fatalf("got %q, want libLiteRt.dylib", got)
		}
	default:
		if got != "libLiteRt.so" {
			t.Fatalf("got %q, want libLiteRt.so", got)
		}
	}
}

func TestResolve(t *testing.T) {
	dir := t.TempDir()
	libPath := filepath.Join(dir, LibraryFilename())
	if err := os.WriteFile(libPath, []byte("stub"), 0o644); err != nil {
		t.Fatal(err)
	}

	gotPath, gotDir, err := Resolve(dir)
	if err != nil {
		t.Fatalf("Resolve(dir): %v", err)
	}
	if gotPath != libPath || gotDir != dir {
		t.Fatalf("got (%q, %q), want (%q, %q)", gotPath, gotDir, libPath, dir)
	}

	gotPath, gotDir, err = Resolve(libPath)
	if err != nil {
		t.Fatalf("Resolve(file): %v", err)
	}
	if gotPath != libPath || gotDir != dir {
		t.Fatalf("got (%q, %q), want (%q, %q)", gotPath, gotDir, libPath, dir)
	}

	t.Setenv(EnvVar, dir)
	gotPath, gotDir, err = Resolve("")
	if err != nil {
		t.Fatalf("Resolve(env): %v", err)
	}
	if gotPath != libPath || gotDir != dir {
		t.Fatalf("got (%q, %q), want (%q, %q)", gotPath, gotDir, libPath, dir)
	}
}

func TestResolveErrors(t *testing.T) {
	t.Setenv(EnvVar, "")
	if _, _, err := Resolve(""); err == nil {
		t.Fatal("expected error when dir and env are empty")
	}

	emptyDir := t.TempDir()
	if _, _, err := Resolve(emptyDir); err == nil {
		t.Fatal("expected error when library file is absent")
	}

	if _, _, err := Load(filepath.Join(emptyDir, "missing")); err == nil {
		t.Fatal("expected error loading nonexistent path")
	}
}
