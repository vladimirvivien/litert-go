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
// (absolute file offsets), a data_type, and optional items (key/value hints). A
// container may hold several TFLiteModel sections (Gemma 3n/4, MTP drafter,
// adapters); each is tagged by an items "model_type" hint
// (tf_lite_prefill_decode, tf_lite_embedder, tf_lite_mtp_drafter, …). The main
// generation graph is the tf_lite_prefill_decode section.
package litertlm

import (
	"encoding/binary"
	"fmt"
	"os"
	"strings"
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

// HintModelType is the SectionObject items key whose value tags a TFLiteModel
// section's role within a multi-section container.
const HintModelType = "model_type"

// model_type hint values (SectionObject items["model_type"]).
const (
	TFLitePrefillDecode    = "tf_lite_prefill_decode" // the main generation graph
	TFLiteEmbedder         = "tf_lite_embedder"
	TFLitePerLayerEmbedder = "tf_lite_per_layer_embedder"
	TFLiteMTPDrafter       = "tf_lite_mtp_drafter"
	TFLiteVisionEncoder    = "tf_lite_vision_encoder"
	TFLiteVisionAdapter    = "tf_lite_vision_adapter"
)

// vDataStringValue is the VData union tag for StringValue (schema enum order,
// NONE=0). Section item parsing reads only string-valued hints.
const vDataStringValue = 9

// Section locates one object within a .litertlm container.
type Section struct {
	Type       uint8
	Begin, End uint64
	Items      map[string]string // string-valued SectionObject items (hints)
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

// SectionTFLite returns the main generation graph: the TFLiteModel section
// tagged model_type=tf_lite_prefill_decode, or — for single-section or unhinted
// containers — the first TFLiteModel section with no model_type hint. Sections
// hinted as other roles (embedder, MTP drafter, adapters) are skipped.
func SectionTFLite(file []byte) ([]byte, error) {
	sections, err := Sections(file)
	if err != nil {
		return nil, err
	}
	for _, s := range sections {
		if s.Type != SectionTFLiteModel {
			continue
		}
		if mt := s.Items[HintModelType]; mt == "" || strings.EqualFold(mt, TFLitePrefillDecode) {
			return sectionData(file, s)
		}
	}
	return nil, fmt.Errorf("litertlm: no prefill/decode TFLiteModel section")
}

// SectionTFLiteModelType returns the TFLiteModel section whose model_type hint
// matches modelType (case-insensitive) — e.g. TFLiteMTPDrafter for the
// speculative-decoding draft model.
func SectionTFLiteModelType(file []byte, modelType string) ([]byte, error) {
	sections, err := Sections(file)
	if err != nil {
		return nil, err
	}
	for _, s := range sections {
		if s.Type == SectionTFLiteModel && strings.EqualFold(s.Items[HintModelType], modelType) {
			return sectionData(file, s)
		}
	}
	return nil, fmt.Errorf("litertlm: no TFLiteModel section with model_type %q", modelType)
}

func sectionData(file []byte, s Section) ([]byte, error) {
	if s.End > uint64(len(file)) || s.Begin > s.End {
		return nil, fmt.Errorf("litertlm: bad section range [%d,%d)", s.Begin, s.End)
	}
	return file[s.Begin:s.End], nil
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
		// SectionObject: items=field 0, begin_offset=field 1, end_offset=field 2,
		// data_type=field 3.
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
		s.Items = meta.sectionItems(obj)
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

// str reads the string whose uoffset is stored at fieldOff (as returned by
// field). A FlatBuffers string is a uoffset to a u32 length followed by bytes.
func (f fb) str(fieldOff uint32) string {
	pos := f.indirect(fieldOff)
	if pos+4 > uint32(len(f.b)) {
		return ""
	}
	n := f.u32(pos)
	end := pos + 4 + n
	if end > uint32(len(f.b)) {
		return ""
	}
	return string(f.b[pos+4 : end])
}

// sectionItems reads SectionObject.items (field 0), a vector of KeyValuePair,
// collecting only the string-valued entries into a map. KeyValuePair fields:
// key=0 (string), value_type=1 (ubyte VData tag), value=2 (union table); the
// StringValue member holds its string at field 0.
func (f fb) sectionItems(obj uint32) map[string]string {
	itemsField := f.field(obj, 0)
	if itemsField == 0 {
		return nil
	}
	vec := f.indirect(itemsField)
	n := f.u32(vec)
	items := make(map[string]string, n)
	for i := uint32(0); i < n; i++ {
		kvp := f.indirect(vec + 4 + 4*i)
		keyField := f.field(kvp, 0)
		vtField := f.field(kvp, 1)
		valField := f.field(kvp, 2)
		if keyField == 0 || vtField == 0 || valField == 0 {
			continue
		}
		if f.b[vtField] != vDataStringValue {
			continue
		}
		valTable := f.indirect(valField)
		if sField := f.field(valTable, 0); sField != 0 {
			items[f.str(keyField)] = f.str(sField)
		}
	}
	if len(items) == 0 {
		return nil
	}
	return items
}
