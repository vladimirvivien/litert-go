package lm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
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

// ToolDefinition is the common contract every tool attached to a chat/generation session satisfies.
type ToolDefinition interface {
	Name() string
	Description() string
	Parameters() map[string]any
}

// dispatchable is implemented by *ManagedTool[I, O] regardless of its type parameters.
type dispatchable interface {
	invoke(ctx context.Context, argsJSON []byte) (any, error)
	policy() ToolPolicy
}

type ToolPolicy int

const (
	ToolPolicyReturnOnError ToolPolicy = iota
	ToolPolicyInformOnError
)

type ToolOption func(*toolConfig)

type toolConfig struct {
	policy ToolPolicy
}

func WithToolPolicy(p ToolPolicy) ToolOption {
	return func(c *toolConfig) { c.policy = p }
}

type RawTool struct {
	name        string
	description string
	parameters  map[string]any
}

func NewRawTool(name, description string, parameters map[string]any) *RawTool {
	return &RawTool{
		name:        name,
		description: description,
		parameters:  parameters,
	}
}

func (r *RawTool) Name() string { return r.name }
func (r *RawTool) Description() string { return r.description }
func (r *RawTool) Parameters() map[string]any { return r.parameters }

type ManagedTool[I, O any] struct {
	name        string
	description string
	handler     func(context.Context, I) (O, error)
	parameters  map[string]any
	errPolicy   ToolPolicy
}

func (t *ManagedTool[I, O]) Name() string { return t.name }
func (t *ManagedTool[I, O]) Description() string { return t.description }
func (t *ManagedTool[I, O]) Parameters() map[string]any { return t.parameters }
func (t *ManagedTool[I, O]) policy() ToolPolicy { return t.errPolicy }

func (t *ManagedTool[I, O]) invoke(ctx context.Context, argsJSON []byte) (any, error) {
	var in I
	if len(argsJSON) > 0 && string(argsJSON) != "null" {
		if err := json.Unmarshal(argsJSON, &in); err != nil {
			return nil, fmt.Errorf("lm: tool %q: unmarshal args: %w", t.name, err)
		}
	}
	out, err := t.handler(ctx, in)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func RegisterTool[I, O any](
	name, description string,
	handler func(context.Context, I) (O, error),
	opts ...ToolOption,
) (*ManagedTool[I, O], error) {
	if name == "" {
		return nil, fmt.Errorf("lm: RegisterTool: empty name")
	}
	if handler == nil {
		return nil, fmt.Errorf("lm: RegisterTool %q: nil handler", name)
	}

	params, err := paramsSchemaOf(reflect.TypeFor[I]())
	if err != nil {
		return nil, fmt.Errorf("lm: RegisterTool %q: %w", name, err)
	}

	cfg := toolConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	tool := &ManagedTool[I, O]{
		name:        name,
		description: description,
		handler:     handler,
		parameters:  params,
		errPolicy:   cfg.policy,
	}
	return tool, nil
}

func paramsSchemaOf(t reflect.Type) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("paramsSchemaOf: type %s must be a struct or pointer to struct", t)
	}
	return paramsSchemaStruct(t, 0)
}

func paramsSchemaStruct(t reflect.Type, depth int) (map[string]any, error) {
	if depth > 32 {
		return nil, fmt.Errorf("paramsSchemaOf: type %s nests deeper than 32 levels", t)
	}

	properties := map[string]any{}
	var required []string

	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		name := jsonFieldName(f)
		if name == "-" {
			continue
		}

		fieldSchema, err := paramsSchemaForType(f.Type, depth+1)
		if err != nil {
			return nil, err
		}
		if desc := f.Tag.Get("description"); desc != "" {
			fieldSchema["description"] = desc
		}
		properties[name] = fieldSchema

		if f.Type.Kind() != reflect.Pointer {
			required = append(required, name)
		}
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema, nil
}

func paramsSchemaForType(t reflect.Type, depth int) (map[string]any, error) {
	if depth > 32 {
		return nil, fmt.Errorf("paramsSchemaOf: type %s nests deeper than 32 levels", t)
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Slice, reflect.Array:
		items, err := paramsSchemaForType(t.Elem(), depth+1)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Struct:
		return paramsSchemaStruct(t, depth+1)
	default:
		return nil, fmt.Errorf("paramsSchemaOf: unsupported kind %s for type %s", t.Kind(), t)
	}
}

func jsonFieldName(f reflect.StructField) string {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "-"
	}
	if tag == "" {
		return strings.ToLower(f.Name)
	}
	if comma := strings.IndexByte(tag, ','); comma >= 0 {
		tag = tag[:comma]
	}
	if tag == "" {
		return strings.ToLower(f.Name)
	}
	return tag
}

// sendWithDispatch drives the auto-dispatch loop for Session
func (s *Session) sendWithDispatch(ctx context.Context, firstTurn string, onPiece func(string)) (string, error) {
	cap := s.o.MaxToolHops
	if cap <= 0 {
		cap = 5
	}
	registry := make(map[string]ToolDefinition, len(s.o.Tools))
	for _, t := range s.o.Tools {
		registry[t.Name()] = t
	}

	reply, err := s.sendTurn(ctx, firstTurn, onPiece)
	if err != nil {
		return "", err
	}

	for hop := 1; ; hop++ {
		cleanText, calls, err := s.e.ExtractToolCalls(reply)
		if err != nil {
			return "", err
		}
		if len(calls) == 0 {
			return cleanText, nil
		}

		allDispatchable := true
		for _, call := range calls {
			if _, ok := registry[call.Name]; !ok {
				allDispatchable = false
				break
			}
		}

		if !allDispatchable {
			return reply, nil
		}

		if hop > cap {
			return reply, fmt.Errorf("lm: tool execution exceeded max tool hops limit (%d)", cap)
		}

		results, err := invokeAll(ctx, calls, registry)
		if err != nil {
			return "", err
		}

		reply, err = s.SendToolResultsStream(ctx, results, onPiece)
		if err != nil {
			return "", err
		}
	}
}

// sendWithDispatch drives the auto-dispatch loop for embedSession
func (s *embedSession) sendWithDispatch(ctx context.Context, parts []Part, onPiece func(string)) (string, error) {
	cap := s.o.MaxToolHops
	if cap <= 0 {
		cap = 5
	}
	registry := make(map[string]ToolDefinition, len(s.o.Tools))
	for _, t := range s.o.Tools {
		registry[t.Name()] = t
	}

	reply, err := s.sendTurn(ctx, parts, false, onPiece)
	if err != nil {
		return "", err
	}

	for hop := 1; ; hop++ {
		cleanText, calls, err := s.e.ExtractToolCalls(reply)
		if err != nil {
			return "", err
		}
		if len(calls) == 0 {
			return cleanText, nil
		}

		allDispatchable := true
		for _, call := range calls {
			if _, ok := registry[call.Name]; !ok {
				allDispatchable = false
				break
			}
		}

		if !allDispatchable {
			return reply, nil
		}

		if hop > cap {
			return reply, fmt.Errorf("lm: tool execution exceeded max tool hops limit (%d)", cap)
		}

		results, err := invokeAll(ctx, calls, registry)
		if err != nil {
			return "", err
		}

		reply, err = s.SendToolResultsStream(ctx, results, onPiece)
		if err != nil {
			return "", err
		}
	}
}

func invokeAll(ctx context.Context, calls []ToolCall, registry map[string]ToolDefinition) ([]ToolResult, error) {
	results := make([]ToolResult, len(calls))
	for i, call := range calls {
		def, ok := registry[call.Name]
		if !ok {
			return nil, fmt.Errorf("lm: tool %q not registered", call.Name)
		}
		d, ok := def.(dispatchable)
		if !ok {
			return nil, fmt.Errorf("lm: tool %q is not auto-dispatchable", call.Name)
		}

		argsJSON, err := json.Marshal(call.Args)
		if err != nil {
			return nil, fmt.Errorf("lm: tool %q: marshal args: %w", call.Name, err)
		}

		out, err := d.invoke(ctx, argsJSON)
		if err != nil {
			if d.policy() == ToolPolicyInformOnError {
				results[i] = ToolResult{
					Name:     call.Name,
					Response: map[string]any{"error": err.Error()},
				}
				continue
			}
			return nil, fmt.Errorf("lm: tool %q: %w", call.Name, err)
		}
		results[i] = ToolResult{Name: call.Name, Response: out}
	}
	return results, nil
}

func buildToolsJSON(tools []ToolDefinition) (string, error) {
	var list []map[string]any
	for _, t := range tools {
		list = append(list, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name(),
				"description": t.Description(),
				"parameters":  t.Parameters(),
			},
		})
	}
	b, err := json.Marshal(list)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// conversationStart renders what precedes the first turn: the system
// prompt, plus the tool declarations when GenOptions carries tool
// specs. Tools force a system turn even with an empty system prompt,
// matching the C++ engine's render.
func conversationStart(e *Engine, tpl litertlm.PromptTemplates, o GenOptions) (string, error) {
	toolsJSON := o.ToolsJSON
	if toolsJSON == "" && len(o.Tools) > 0 {
		var err error
		toolsJSON, err = buildToolsJSON(o.Tools)
		if err != nil {
			return "", err
		}
	}
	if toolsJSON == "" {
		return renderSystem(tpl, o.System), nil
	}
	tt, ok := litertlm.ToolTemplatesFor(e.md.ModelType)
	if !ok {
		return "", fmt.Errorf("lm: tools are not supported for model type %q", e.md.ModelType)
	}
	decls, err := fcDeclarations(toolsJSON, tt)
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
