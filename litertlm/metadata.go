package litertlm

import "fmt"

// ModelType is the model family carried by LlmMetadata.llm_model_type, a
// protobuf oneof. The value identifies which data processor / chat template the
// model expects.
type ModelType int

// Field numbers of the LlmModelType oneof (runtime/proto/llm_model_type.proto).
const (
	ModelUnknown       ModelType = 0
	ModelGeneric       ModelType = 1
	ModelGemma3N       ModelType = 2
	ModelFunctionGemma ModelType = 3
	ModelGemma3        ModelType = 4
	ModelQwen3         ModelType = 5
	ModelQwen2p5       ModelType = 7
	ModelGemma4        ModelType = 8
	ModelFastVLM       ModelType = 9
)

func (m ModelType) String() string {
	switch m {
	case ModelGeneric:
		return "generic"
	case ModelGemma3N:
		return "gemma3n"
	case ModelFunctionGemma:
		return "function_gemma"
	case ModelGemma3:
		return "gemma3"
	case ModelQwen3:
		return "qwen3"
	case ModelQwen2p5:
		return "qwen2p5"
	case ModelGemma4:
		return "gemma4"
	case ModelFastVLM:
		return "fastvlm"
	default:
		return "unknown"
	}
}

// Metadata holds the LlmMetadata fields the executor needs to format prompts.
type Metadata struct {
	ModelType    ModelType
	MaxNumTokens int
}

// ReadMetadata parses the LlmMetadataProto section of a .litertlm container.
// It returns an error if the container has no metadata section. ModelType is
// Unknown and MaxNumTokens is 0 when the metadata omits those fields — common
// for gemma3 / qwen3 containers, which do not set llm_model_type.
func ReadMetadata(file []byte) (Metadata, error) {
	sec, err := SectionBytes(file, SectionLlmMetadata)
	if err != nil {
		return Metadata{}, err
	}
	return parseMetadata(sec), nil
}

// parseMetadata scans an LlmMetadata message: max_num_tokens = field 5 (varint),
// llm_model_type = field 6 (sub-message holding the family oneof).
func parseMetadata(sec []byte) Metadata {
	var m Metadata
	r := pb{b: sec}
	for {
		field, wire, v, data, ok := r.next()
		if !ok {
			break
		}
		switch {
		case field == 5 && wire == 0:
			m.MaxNumTokens = int(v)
		case field == 6 && wire == 2:
			m.ModelType = modelTypeFromOneof(data)
		}
	}
	return m
}

// modelTypeFromOneof returns the set field of the LlmModelType oneof. Exactly
// one field is set, so the first one encountered identifies the family.
func modelTypeFromOneof(b []byte) ModelType {
	r := pb{b: b}
	if field, _, _, _, ok := r.next(); ok {
		return ModelType(field)
	}
	return ModelUnknown
}

// SectionBytes returns the bytes of the first section of the given data type.
func SectionBytes(file []byte, dataType uint8) ([]byte, error) {
	sections, err := Sections(file)
	if err != nil {
		return nil, err
	}
	for _, s := range sections {
		if s.Type == dataType {
			if s.End > uint64(len(file)) || s.Begin > s.End {
				return nil, fmt.Errorf("litertlm: bad section range [%d,%d)", s.Begin, s.End)
			}
			return file[s.Begin:s.End], nil
		}
	}
	return nil, fmt.Errorf("litertlm: no section of type %d", dataType)
}

// pb is a minimal protobuf wire-format reader for top-level field scanning. It
// does not descend into sub-messages (the caller re-scans those), which keeps
// field matching unambiguous.
type pb struct {
	b   []byte
	pos int
}

func (r *pb) varint() (uint64, bool) {
	var x uint64
	var s uint
	for r.pos < len(r.b) {
		c := r.b[r.pos]
		r.pos++
		x |= uint64(c&0x7f) << s
		if c < 0x80 {
			return x, true
		}
		s += 7
	}
	return 0, false
}

// next decodes one field. For varint fields the value is in v; for
// length-delimited fields the payload is in data. ok is false at end of input.
func (r *pb) next() (field, wire int, v uint64, data []byte, ok bool) {
	tag, ok := r.varint()
	if !ok {
		return 0, 0, 0, nil, false
	}
	field = int(tag >> 3)
	wire = int(tag & 7)
	switch wire {
	case 0: // varint
		v, ok = r.varint()
	case 1: // 64-bit
		if r.pos+8 > len(r.b) {
			return 0, 0, 0, nil, false
		}
		data = r.b[r.pos : r.pos+8]
		r.pos += 8
	case 2: // length-delimited
		n, vok := r.varint()
		if !vok || r.pos+int(n) > len(r.b) {
			return 0, 0, 0, nil, false
		}
		data = r.b[r.pos : r.pos+int(n)]
		r.pos += int(n)
	case 5: // 32-bit
		if r.pos+4 > len(r.b) {
			return 0, 0, 0, nil, false
		}
		data = r.b[r.pos : r.pos+4]
		r.pos += 4
	default:
		return 0, 0, 0, nil, false
	}
	return field, wire, v, data, true
}
