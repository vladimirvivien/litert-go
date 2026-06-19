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

// String returns the model_type hint as recorded in the container.
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

// Affixes wraps a role's content for a chat turn: Prefix is prepended and
// Suffix appended. Mirrors LlmMetadata.PromptAffixes.
type Affixes struct {
	Prefix string
	Suffix string
}

// PromptTemplates holds the per-role chat affixes from
// LlmMetadata.prompt_templates.
type PromptTemplates struct {
	User   Affixes
	Model  Affixes
	System Affixes
}

// Token references a token either by explicit IDs or by a string to be looked
// up in the tokenizer. Mirrors LlmMetadata.TokenUnion: control tokens (e.g. a
// BOS that has no round-tripping string) are carried as IDs, user-defined
// markers as strings. Exactly one form is populated.
type Token struct {
	IDs []int32
	Str string
}

// Metadata holds the LlmMetadata fields the executor needs to format prompts.
type Metadata struct {
	ModelType    ModelType
	MaxNumTokens int
	StartToken   Token           // field 1: prepended to the input sequence
	StopTokens   []Token         // field 2: end-of-output markers
	Prompts      PromptTemplates // field 3
	HasPrompts   bool            // whether prompt_templates was present
}

// Templates returns the chat affixes for the model: the container's
// prompt_templates when present, otherwise the documented per-family fallback
// for containers that ship only a Jinja template (e.g. gemma4). ok is false
// when neither is available.
func (m Metadata) Templates() (PromptTemplates, bool) {
	if m.HasPrompts {
		return m.Prompts, true
	}
	return FallbackTemplates(m.ModelType)
}

// FallbackTemplates returns the fixed chat format for a family whose container
// carries only a Jinja template (no prompt_templates field). Gemma 4 uses the
// <|turn> / <turn|> markers (a fixed structure per the Gemma 4 prompt-format
// docs, not Jinja).
func FallbackTemplates(mt ModelType) (PromptTemplates, bool) {
	switch mt {
	case ModelGemma3, ModelFunctionGemma:
		return PromptTemplates{
			User:  Affixes{Prefix: "<start_of_turn>user\n", Suffix: "<end_of_turn>\n"},
			Model: Affixes{Prefix: "<start_of_turn>model\n", Suffix: "<end_of_turn>\n"},
		}, true
	case ModelGemma4:
		return PromptTemplates{
			User:   Affixes{Prefix: "<|turn>user\n", Suffix: "<turn|>\n"},
			Model:  Affixes{Prefix: "<|turn>model\n", Suffix: "<turn|>\n"},
			System: Affixes{Prefix: "<|turn>system\n", Suffix: "<turn|>\n"},
		}, true
	}
	return PromptTemplates{}, false
}

// ToolTemplates describes a family's function-calling syntax: the
// markers that wrap tool declarations in the system turn, tool calls
// in model output, and tool responses sent back, plus the token that
// quotes string values in the FC expression syntax.
type ToolTemplates struct {
	DeclStart, DeclEnd string
	CallStart, CallEnd string
	RespStart, RespEnd string
	Quote              string
}

// ToolTemplatesFor returns the family's tool syntax. ok is false for
// families without validated tool support (as of LiteRT-LM v0.13.1
// the C++ engine renders tools only for the Gemma 4 family; its qwen3
// path drops tool specs and tool-result content).
func ToolTemplatesFor(mt ModelType) (ToolTemplates, bool) {
	switch mt {
	case ModelGemma4:
		return ToolTemplates{
			DeclStart: "<|tool>", DeclEnd: "<tool|>",
			CallStart: "<|tool_call>", CallEnd: "<tool_call|>",
			RespStart: "<|tool_response>", RespEnd: "<tool_response|>",
			Quote: `<|"|>`,
		}, true
	}
	return ToolTemplates{}, false
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

// parseMetadata scans an LlmMetadata message: start_token = field 1, stop_tokens
// = field 2 (repeated), prompt_templates = field 3, max_num_tokens = field 5
// (varint), llm_model_type = field 6 (sub-message holding the family oneof).
func parseMetadata(sec []byte) Metadata {
	var m Metadata
	r := pb{b: sec}
	for {
		field, wire, v, data, ok := r.next()
		if !ok {
			break
		}
		switch {
		case field == 1 && wire == 2:
			m.StartToken = parseTokenUnion(data)
		case field == 2 && wire == 2:
			m.StopTokens = append(m.StopTokens, parseTokenUnion(data))
		case field == 3 && wire == 2:
			m.Prompts = parsePromptTemplates(data)
			m.HasPrompts = true
		case field == 5 && wire == 0:
			m.MaxNumTokens = int(v)
		case field == 6 && wire == 2:
			m.ModelType = modelTypeFromOneof(data)
		}
	}
	return m
}

// parseTokenUnion reads a TokenUnion: token_ids = field 1 (a TokenIds
// sub-message), token_str = field 2 (string).
func parseTokenUnion(b []byte) Token {
	var t Token
	r := pb{b: b}
	for {
		field, wire, _, data, ok := r.next()
		if !ok {
			break
		}
		switch {
		case field == 1 && wire == 2:
			t.IDs = parseTokenIDs(data)
		case field == 2 && wire == 2:
			t.Str = string(data)
		}
	}
	return t
}

// parseTokenIDs reads TokenIds.ids = field 1 (repeated int32), accepting both
// the packed (proto3 default) and unpacked encodings.
func parseTokenIDs(b []byte) []int32 {
	var ids []int32
	r := pb{b: b}
	for {
		field, wire, v, data, ok := r.next()
		if !ok {
			break
		}
		if field != 1 {
			continue
		}
		switch wire {
		case 0:
			ids = append(ids, int32(v))
		case 2:
			p := pb{b: data}
			for {
				x, vok := p.varint()
				if !vok {
					break
				}
				ids = append(ids, int32(x))
			}
		}
	}
	return ids
}

func parsePromptTemplates(b []byte) PromptTemplates {
	var pt PromptTemplates
	r := pb{b: b}
	for {
		field, wire, _, data, ok := r.next()
		if !ok {
			break
		}
		if wire != 2 {
			continue
		}
		switch field {
		case 1:
			pt.User = parseAffixes(data)
		case 2:
			pt.Model = parseAffixes(data)
		case 3:
			pt.System = parseAffixes(data)
		}
	}
	return pt
}

func parseAffixes(b []byte) Affixes {
	var a Affixes
	r := pb{b: b}
	for {
		field, wire, _, data, ok := r.next()
		if !ok {
			break
		}
		if wire != 2 {
			continue
		}
		switch field {
		case 1:
			a.Prefix = string(data)
		case 2:
			a.Suffix = string(data)
		}
	}
	return a
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
