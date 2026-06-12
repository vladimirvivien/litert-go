package lm

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/vladimirvivien/litert-go/litertlm"
)

// Function calling uses the FC expression syntax the C++ LiteRT-LM
// engine speaks for the Gemma 4 family: tool declarations are rendered
// into the system turn as `declaration:name{...}` blocks, the model
// emits `call:name{...}` blocks between the family's call markers, and
// tool results return as `response:name{...}` blocks. FC objects use
// unquoted keys sorted alphabetically, family quote tokens around
// strings, and uppercased JSON-Schema type names in declarations.

// ToolCall is one function invocation parsed from model output.
type ToolCall struct {
	Name string
	Args map[string]any
}

// ToolResult is one function result sent back to the model.
type ToolResult struct {
	Name     string
	Response any
}

// ToolSender is implemented by conversations that support tool
// calling (KV-reuse sessions over tool-capable families). Use a type
// assertion on the Conversation returned by NewConversation.
type ToolSender interface {
	// SendToolResults delivers function results and decodes the
	// model's follow-up turn.
	SendToolResults(ctx context.Context, results []ToolResult) (string, error)
	// SendToolResultsStream is SendToolResults with incremental
	// output.
	SendToolResultsStream(ctx context.Context, results []ToolResult, onPiece func(string)) (string, error)
}

// conversationStart renders what precedes the first turn: the system
// prompt, plus the tool declarations when GenOptions carries tool
// specs. Tools force a system turn even with an empty system prompt,
// matching the C++ engine's render.
func conversationStart(e *Engine, tpl litertlm.PromptTemplates, o GenOptions) (string, error) {
	if o.ToolsJSON == "" {
		return renderSystem(tpl, o.System), nil
	}
	tt, ok := litertlm.ToolTemplatesFor(e.md.ModelType)
	if !ok {
		return "", fmt.Errorf("lm: tools are not supported for model type %q", e.md.ModelType)
	}
	decls, err := fcDeclarations(o.ToolsJSON, tt)
	if err != nil {
		return "", err
	}
	return tpl.System.Prefix + o.System + "\n\n" + decls + tpl.System.Suffix, nil
}

// toolResultsTurn renders function results as one conversation turn:
// the family's response blocks, with no role affixes (the C++ engine
// ingests tool responses outside the user/model turn structure).
func toolResultsTurn(e *Engine, results []ToolResult) (string, error) {
	tt, ok := litertlm.ToolTemplatesFor(e.md.ModelType)
	if !ok {
		return "", fmt.Errorf("lm: tools are not supported for model type %q", e.md.ModelType)
	}
	return fcToolResponses(results, tt)
}

// ExtractToolCalls splits a model reply into its text and the tool
// calls it contains, using the engine's family markers. Replies from
// families without tool support pass through unchanged.
func (e *Engine) ExtractToolCalls(reply string) (string, []ToolCall, error) {
	tt, ok := litertlm.ToolTemplatesFor(e.md.ModelType)
	if !ok {
		return reply, nil, nil
	}
	return extractToolCalls(reply, tt)
}

// extractToolCalls removes every CallStart..CallEnd block from text
// and parses each block's `call:name{args}` FC expression.
func extractToolCalls(text string, tt litertlm.ToolTemplates) (string, []ToolCall, error) {
	var calls []ToolCall
	var clean strings.Builder
	rest := text
	for {
		i := strings.Index(rest, tt.CallStart)
		if i < 0 {
			clean.WriteString(rest)
			break
		}
		clean.WriteString(rest[:i])
		rest = rest[i+len(tt.CallStart):]
		j := strings.Index(rest, tt.CallEnd)
		if j < 0 {
			// Unterminated block: treat the marker as text.
			clean.WriteString(tt.CallStart)
			clean.WriteString(rest)
			break
		}
		block := rest[:j]
		rest = rest[j+len(tt.CallEnd):]
		blockCalls, err := parseFcCalls(block, tt.Quote)
		if err != nil {
			return "", nil, fmt.Errorf("lm: parse tool call %q: %w", block, err)
		}
		calls = append(calls, blockCalls...)
	}
	return strings.TrimSpace(clean.String()), calls, nil
}

// parseFcCalls parses one fenced block: one or more `call:name{args}`
// expressions (args optional).
func parseFcCalls(block, quote string) ([]ToolCall, error) {
	p := &fcParser{s: strings.TrimSpace(block), quote: quote}
	var calls []ToolCall
	for !p.eof() {
		p.skipSpace()
		if !p.consume("call:") {
			return nil, fmt.Errorf("expected call: at %q", p.rest())
		}
		name := p.ident()
		if name == "" {
			return nil, fmt.Errorf("missing function name at %q", p.rest())
		}
		args := map[string]any{}
		p.skipSpace()
		if p.peek() == '{' {
			v, err := p.value()
			if err != nil {
				return nil, err
			}
			obj, ok := v.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("arguments of %s are not an object", name)
			}
			args = obj
		}
		calls = append(calls, ToolCall{Name: name, Args: args})
		p.skipSpace()
	}
	return calls, nil
}

// fcParser scans the FC expression syntax: objects with unquoted keys,
// strings delimited by the family quote token, bare numbers and bools,
// and nested arrays/objects.
type fcParser struct {
	s     string
	pos   int
	quote string
}

func (p *fcParser) eof() bool    { return p.pos >= len(p.s) }
func (p *fcParser) rest() string { return p.s[p.pos:] }
func (p *fcParser) peek() byte {
	if p.eof() {
		return 0
	}
	return p.s[p.pos]
}

func (p *fcParser) skipSpace() {
	for !p.eof() && (p.s[p.pos] == ' ' || p.s[p.pos] == '\n' || p.s[p.pos] == '\t' || p.s[p.pos] == '\r') {
		p.pos++
	}
}

func (p *fcParser) consume(tok string) bool {
	if strings.HasPrefix(p.rest(), tok) {
		p.pos += len(tok)
		return true
	}
	return false
}

// ident reads an unquoted identifier (function or key name).
func (p *fcParser) ident() string {
	start := p.pos
	for !p.eof() {
		c := p.s[p.pos]
		if c == ':' || c == '{' || c == '}' || c == ',' || c == ' ' || c == '\n' {
			break
		}
		p.pos++
	}
	return p.s[start:p.pos]
}

func (p *fcParser) value() (any, error) {
	p.skipSpace()
	switch {
	case p.consume(p.quote):
		end := strings.Index(p.rest(), p.quote)
		if end < 0 {
			return nil, fmt.Errorf("unterminated string at %q", p.rest())
		}
		s := p.rest()[:end]
		p.pos += end + len(p.quote)
		return s, nil
	case p.peek() == '{':
		p.pos++
		obj := map[string]any{}
		p.skipSpace()
		if p.peek() == '}' {
			p.pos++
			return obj, nil
		}
		for {
			p.skipSpace()
			key := p.ident()
			if key == "" {
				return nil, fmt.Errorf("missing key at %q", p.rest())
			}
			p.skipSpace()
			if !p.consume(":") {
				return nil, fmt.Errorf("missing : after %q", key)
			}
			v, err := p.value()
			if err != nil {
				return nil, err
			}
			obj[key] = v
			p.skipSpace()
			if p.consume(",") {
				continue
			}
			if p.consume("}") {
				return obj, nil
			}
			return nil, fmt.Errorf("missing , or } at %q", p.rest())
		}
	case p.peek() == '[':
		p.pos++
		arr := []any{}
		p.skipSpace()
		if p.peek() == ']' {
			p.pos++
			return arr, nil
		}
		for {
			v, err := p.value()
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
			p.skipSpace()
			if p.consume(",") {
				continue
			}
			if p.consume("]") {
				return arr, nil
			}
			return nil, fmt.Errorf("missing , or ] at %q", p.rest())
		}
	case p.consume("true"):
		return true, nil
	case p.consume("false"):
		return false, nil
	case p.consume("null"):
		return nil, nil
	default:
		start := p.pos
		for !p.eof() {
			c := p.s[p.pos]
			if c == ',' || c == '}' || c == ']' || c == ' ' || c == '\n' {
				break
			}
			p.pos++
		}
		lit := p.s[start:p.pos]
		if lit == "" {
			return nil, fmt.Errorf("empty value at %q", p.rest())
		}
		if i, err := strconv.ParseInt(lit, 10, 64); err == nil {
			return float64(i), nil
		}
		f, err := strconv.ParseFloat(lit, 64)
		if err != nil {
			return nil, fmt.Errorf("bad literal %q", lit)
		}
		return f, nil
	}
}

// fcDeclarations renders OpenAI-style tool specs (a JSON array of
// {"type":"function","function":{name,description,parameters}}) into
// the family's declaration blocks, concatenated without separators.
func fcDeclarations(toolsJSON string, tt litertlm.ToolTemplates) (string, error) {
	var tools []struct {
		Function json.RawMessage `json:"function"`
	}
	if err := json.Unmarshal([]byte(toolsJSON), &tools); err != nil {
		return "", fmt.Errorf("lm: parse tools JSON: %w", err)
	}
	var b strings.Builder
	for _, tool := range tools {
		var fn map[string]any
		if err := json.Unmarshal(tool.Function, &fn); err != nil {
			return "", fmt.Errorf("lm: parse tool function: %w", err)
		}
		name, _ := fn["name"].(string)
		if name == "" {
			return "", fmt.Errorf("lm: tool with empty name")
		}
		delete(fn, "name")
		b.WriteString(tt.DeclStart)
		b.WriteString("declaration:")
		b.WriteString(name)
		writeFcValue(&b, fn, tt.Quote, true)
		b.WriteString(tt.DeclEnd)
	}
	return b.String(), nil
}

// fcToolResponses renders tool results into the family's response
// blocks, concatenated without separators.
func fcToolResponses(results []ToolResult, tt litertlm.ToolTemplates) (string, error) {
	var b strings.Builder
	for _, r := range results {
		if r.Name == "" {
			return "", fmt.Errorf("lm: tool result with empty name")
		}
		// Round-trip through JSON so arbitrary Go values reduce to
		// the maps/slices/scalars the FC writer handles.
		raw, err := json.Marshal(r.Response)
		if err != nil {
			return "", fmt.Errorf("lm: marshal tool response: %w", err)
		}
		var v any
		if err := json.Unmarshal(raw, &v); err != nil {
			return "", fmt.Errorf("lm: tool response: %w", err)
		}
		b.WriteString(tt.RespStart)
		b.WriteString("response:")
		b.WriteString(r.Name)
		writeFcValue(&b, v, tt.Quote, false)
		b.WriteString(tt.RespEnd)
	}
	return b.String(), nil
}

// writeFcValue writes one value in FC syntax: objects with
// alphabetically sorted unquoted keys, quote-token strings, bare
// numbers and bools. upperTypes uppercases the values of "type" keys
// (JSON-Schema type names in declarations).
func writeFcValue(b *strings.Builder, v any, quote string, upperTypes bool) {
	writeFcValueKeyed(b, v, quote, upperTypes, "")
}

func writeFcValueKeyed(b *strings.Builder, v any, quote string, upperTypes bool, key string) {
	switch x := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(k)
			b.WriteByte(':')
			writeFcValueKeyed(b, x[k], quote, upperTypes, k)
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			writeFcValueKeyed(b, e, quote, upperTypes, "")
		}
		b.WriteByte(']')
	case string:
		if upperTypes && key == "type" {
			x = strings.ToUpper(x)
		}
		b.WriteString(quote)
		b.WriteString(x)
		b.WriteString(quote)
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case float64:
		b.WriteString(strconv.FormatFloat(x, 'g', -1, 64))
	case nil:
		b.WriteString("null")
	default:
		b.WriteString(fmt.Sprint(x))
	}
}
