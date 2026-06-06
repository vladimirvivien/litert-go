package litertlm

import (
	"os"
	"testing"
)

func TestIsContainer(t *testing.T) {
	if IsContainer([]byte("nope")) {
		t.Error("short non-container reported as container")
	}
	if !IsContainer([]byte("LITERTLM\x00\x00\x00\x00")) {
		t.Error("magic not recognized")
	}
}

func TestSectionsRejectsNonContainer(t *testing.T) {
	if _, err := Sections([]byte("not a litertlm file at all")); err == nil {
		t.Error("expected error for non-container input")
	}
}

func TestParseMetadata(t *testing.T) {
	// LlmMetadata { max_num_tokens=2048 (field 5);
	//               llm_model_type { gemma4 (oneof field 8) } (field 6) }.
	oneof := []byte{8<<3 | 2, 0x00} // field 8, length-delimited, empty Gemma4
	var b []byte
	b = append(b, 5<<3|0) // field 5, varint
	b = appendVarint(b, 2048)
	b = append(b, 6<<3|2, byte(len(oneof))) // field 6, length-delimited
	b = append(b, oneof...)

	md := parseMetadata(b)
	if md.MaxNumTokens != 2048 {
		t.Errorf("MaxNumTokens = %d, want 2048", md.MaxNumTokens)
	}
	if md.ModelType != ModelGemma4 {
		t.Errorf("ModelType = %s, want gemma4", md.ModelType)
	}
}

func TestParseMetadataAbsentModelType(t *testing.T) {
	// max_num_tokens only, no llm_model_type — the gemma3 / qwen3 case.
	b := appendVarint([]byte{5<<3 | 0}, 4096)
	md := parseMetadata(b)
	if md.MaxNumTokens != 4096 {
		t.Errorf("MaxNumTokens = %d, want 4096", md.MaxNumTokens)
	}
	if md.ModelType != ModelUnknown {
		t.Errorf("ModelType = %s, want unknown", md.ModelType)
	}
}

func appendVarint(b []byte, v uint64) []byte {
	for v >= 0x80 {
		b = append(b, byte(v)|0x80)
		v >>= 7
	}
	return append(b, byte(v))
}

// benchData loads the .litertlm file named by LITERTLM_BENCH_FILE. The
// benchmarks skip when the variable is unset so the suite stays portable.
func benchData(b *testing.B) (string, []byte) {
	b.Helper()
	path := os.Getenv("LITERTLM_BENCH_FILE")
	if path == "" {
		b.Skip("set LITERTLM_BENCH_FILE to a .litertlm path to run")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatalf("read %s: %v", path, err)
	}
	return path, data
}

// BenchmarkSections measures parsing the section table from an in-memory
// container (parser cost only, no I/O).
func BenchmarkSections(b *testing.B) {
	_, data := benchData(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Sections(data); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSectionTFLite measures locating the TFLite section in memory. The
// returned slice aliases the input, so this is parse plus a sub-slice.
func BenchmarkSectionTFLite(b *testing.B) {
	_, data := benchData(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SectionTFLite(data); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkReadTFLite measures the full loader: read the file from disk plus
// parse and locate the TFLite section. Throughput is reported over file size.
func BenchmarkReadTFLite(b *testing.B) {
	path, data := benchData(b)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ReadTFLite(path); err != nil {
			b.Fatal(err)
		}
	}
}
