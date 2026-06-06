// Package litertlm reads the .litertlm container format (LiteRT-LM's wrapper
// around a .tflite model, a tokenizer, and metadata) far enough to extract the
// embedded TFLite model.
//
// File layout (schema/core/litertlm_header_schema.fbs in LiteRT-LM):
//
//	[0:8]    magic "LITERTLM"
//	[8:20]   major/minor/patch version (3x uint32 LE)
//	[20:24]  padding
//	[24:32]  header_end_offset (uint64 LE)
//	[32:end] FlatBuffer LiteRTLMMetaData (root_type)
//
// The metadata's section_metadata.objects[] each carry begin_offset/end_offset
// (absolute file offsets) and a data_type. The TFLite model is the section whose
// data_type is TFLiteModel.
package litertlm

import (
	"encoding/binary"
	"fmt"
	"os"
)

// AnySectionDataType values, in declaration order from the schema enum.
const (
	SectionNone uint8 = iota
	SectionGenericBinary
	SectionDeprecated
	SectionTFLiteModel
	SectionSPTokenizer
	SectionLlmMetadata
	SectionHFTokenizerZlib
	SectionTFLiteWeights
)

const magic = "LITERTLM"

// Section locates one object within a .litertlm container.
type Section struct {
	Type       uint8
	Begin, End uint64
}

// TypeName returns a short name for the section's data type.
func (s Section) TypeName() string {
	switch s.Type {
	case SectionGenericBinary:
		return "GenericBinaryData"
	case SectionTFLiteModel:
		return "TFLiteModel"
	case SectionSPTokenizer:
		return "SP_Tokenizer"
	case SectionLlmMetadata:
		return "LlmMetadataProto"
	case SectionHFTokenizerZlib:
		return "HF_Tokenizer_Zlib"
	case SectionTFLiteWeights:
		return "TFLiteWeights"
	default:
		return fmt.Sprintf("type(%d)", s.Type)
	}
}

// IsContainer reports whether b begins with the .litertlm magic.
func IsContainer(b []byte) bool {
	return len(b) >= len(magic) && string(b[:len(magic)]) == magic
}

// ReadTFLite reads a .litertlm file and returns the bytes of its first
// TFLiteModel section.
func ReadTFLite(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return SectionTFLite(data)
}

// SectionTFLite returns the first TFLiteModel section from an in-memory
// .litertlm container.
func SectionTFLite(file []byte) ([]byte, error) {
	return SectionBytes(file, SectionTFLiteModel)
}

// Sections parses the container header and returns its section table.
func Sections(file []byte) ([]Section, error) {
	if !IsContainer(file) {
		return nil, fmt.Errorf("litertlm: missing %q magic", magic)
	}
	if len(file) < 32 {
		return nil, fmt.Errorf("litertlm: file too short (%d bytes)", len(file))
	}
	headerEnd := binary.LittleEndian.Uint64(file[24:32])
	if headerEnd < 32 || headerEnd > uint64(len(file)) {
		return nil, fmt.Errorf("litertlm: bad header_end_offset %d", headerEnd)
	}

	meta := fb{b: file[32:headerEnd]}
	root := meta.indirect(0)

	// LiteRTLMMetaData.section_metadata = field 1.
	smField := meta.field(root, 1)
	if smField == 0 {
		return nil, fmt.Errorf("litertlm: no section_metadata")
	}
	sm := meta.indirect(smField)

	// SectionMetadata.objects = field 0 (vector of SectionObject).
	objsField := meta.field(sm, 0)
	if objsField == 0 {
		return nil, fmt.Errorf("litertlm: no section objects")
	}
	objs := meta.indirect(objsField)
	n := meta.u32(objs)

	sections := make([]Section, 0, n)
	for i := uint32(0); i < n; i++ {
		obj := meta.indirect(objs + 4 + 4*i)
		// SectionObject: begin_offset=field 1, end_offset=field 2, data_type=field 3.
		var s Section
		if off := meta.field(obj, 1); off != 0 {
			s.Begin = meta.u64(off)
		}
		if off := meta.field(obj, 2); off != 0 {
			s.End = meta.u64(off)
		}
		if off := meta.field(obj, 3); off != 0 {
			s.Type = meta.b[off]
		}
		sections = append(sections, s)
	}
	return sections, nil
}

// fb is a minimal little-endian FlatBuffers reader for the fixed traversal this
// package needs: tables (via vtable) and a vector of tables. It implements no
// more of the wire format than that.
type fb struct{ b []byte }

func (f fb) u16(off uint32) uint16 { return binary.LittleEndian.Uint16(f.b[off:]) }
func (f fb) u32(off uint32) uint32 { return binary.LittleEndian.Uint32(f.b[off:]) }
func (f fb) i32(off uint32) int32  { return int32(f.u32(off)) }
func (f fb) u64(off uint32) uint64 { return binary.LittleEndian.Uint64(f.b[off:]) }

// indirect follows the uoffset stored at off to the absolute position it names.
func (f fb) indirect(off uint32) uint32 { return off + f.u32(off) }

// field returns the absolute offset of field id within the table at tablePos,
// or 0 when the field is absent (default).
func (f fb) field(tablePos uint32, id int) uint32 {
	vtable := uint32(int32(tablePos) - f.i32(tablePos))
	vtableSize := uint32(f.u16(vtable))
	slot := uint32(4 + 2*id)
	if slot >= vtableSize {
		return 0
	}
	rel := uint32(f.u16(vtable + slot))
	if rel == 0 {
		return 0
	}
	return tablePos + rel
}
