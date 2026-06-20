package lm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"reflect"
	"sync"
)

// captureKey[T] is the context.Value key the synthesized capture tool
// reads at dispatch time to find the per-call destination pointer.
// Generic over T so distinct types in the same engine don't collide.
type captureKey[T any] struct{}

// errCaptureToolUnsuitable signals that T cannot be expressed as a
// JSON-Schema object (e.g. T is a slice, map, or scalar). The
// synthesized-tool path is unavailable; the caller falls through to
// the prompt-engineered path.
var errCaptureToolUnsuitable = errors.New("lm: capture tool unsuitable for type")

const reservedToolNamePrefix = "__lm_"

var captureTools sync.Map

// defaultSchemaInstruction is a Printf format string with one %s
// placeholder for the shape hint, used by the fallback path. Imperative
// and short; explicitly forbids markdown fences.
const defaultSchemaInstruction = "Respond with valid JSON only — no commentary, no markdown fences.\n" +
	"The output must match this shape:\n%s"

// captureDirective prepends to the user's prompt on the tool-call
// path. Directs the model to deliver the answer as the synthesized
// tool's arguments rather than as free-form text.
const captureDirective = "Respond by calling the available tool with the structured value as its arguments. " +
	"Do not write any text outside the tool call.\n\n"

// captureToolName returns a deterministic, reserved-prefixed tool name
// derived from T's reflect.Type.String(). The hash is FNV-1a 64-bit;
// the name is short, alphanumeric, and stable across runs.
func captureToolName[T any]() string {
	h := fnv.New64a()
	h.Write([]byte(reflect.TypeFor[T]().String()))
	return fmt.Sprintf("%scapture_%016x", reservedToolNamePrefix, h.Sum64())
}

// captureDescription is the model-facing tool description. The wording
// is a direct instruction so the model invokes the tool exactly once
// with the structured value as its arguments.
func captureDescription[T any]() string {
	return fmt.Sprintf(
		"Deliver the requested structured value as the arguments to this tool. "+
			"Call this tool exactly once. Do not call any other tool. "+
			"Do not emit any text outside the tool call. "+
			"The arguments object conforms to the JSON-Schema for type %s.",
		reflect.TypeFor[T]().String())
}

// getOrSynthesizeCaptureTool returns the capture tool for T.
//
// Returns errCaptureToolUnsuitable when T is not a struct (and thus
// has no JSON-Schema object representation). Callers should treat
// that sentinel as a signal to fall back to the prompt-engineered
// GenerateData path.
func getOrSynthesizeCaptureTool[T any]() (*ManagedTool[T, struct{}], error) {
	typ := reflect.TypeFor[T]()
	if val, ok := captureTools.Load(typ); ok {
		return val.(*ManagedTool[T, struct{}]), nil
	}

	t := typ
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("%w: %s", errCaptureToolUnsuitable, typ.String())
	}

	name := captureToolName[T]()
	desc := captureDescription[T]()
	handler := captureHandler[T]()

	tool, err := RegisterTool[T, struct{}](name, desc, handler)
	if err != nil {
		return nil, err
	}

	actual, _ := captureTools.LoadOrStore(typ, tool)
	return actual.(*ManagedTool[T, struct{}]), nil
}

// captureHandler is the shared closure all GenerateData[T] calls of a
// given T route through. At dispatch time it pulls the per-call
// destination pointer from ctx.Value and writes the typed input.
// Concurrent calls with distinct contexts each see their own slot.
func captureHandler[T any]() func(context.Context, T) (struct{}, error) {
	return func(ctx context.Context, in T) (struct{}, error) {
		slot, ok := ctx.Value(captureKey[T]{}).(**T)
		if !ok || slot == nil {
			return struct{}{}, nil
		}
		v := in
		*slot = &v
		return struct{}{}, nil
	}
}

// GenerateData is the text-only convenience wrapper over
// GenerateDataMulti.
//
// On parse-path failure after the final attempt the caller receives a
// *GenerateDataError; use errors.As to inspect Phase / Raw / Attempts.
// Generate-phase errors (ctx cancellation, FFI failure) propagate
// immediately and do not trigger retries.
func GenerateData[T any](ctx context.Context, e *Engine, prompt string, o GenOptions) (*T, error) {
	return GenerateDataMulti[T](ctx, e, []Part{{Kind: "text", Text: prompt}}, o)
}

// GenerateDataMulti is the multimodal sibling of GenerateData. parts
// may carry text, image, and audio segments.
//
// Each attempt tries the synthesized-tool path first, then falls
// through to the prompt-engineered path if the tool-call did not
// deliver a value. Retries (defined via o.Retries) repeat the full sequence
// up to 1+o.Retries times. Generate-phase errors propagate immediately.
func GenerateDataMulti[T any](ctx context.Context, e *Engine, parts []Part, o GenOptions) (*T, error) {
	if e == nil {
		return nil, fmt.Errorf("lm: GenerateDataMulti: nil engine")
	}

	var lastErr error
	for attempt := 1; attempt <= 1+o.Retries; attempt++ {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}

		if result := tryCaptureToolSilent[T](ctx, e, parts, o); result != nil {
			return result, nil
		}
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}

		result, err := generateDataPromptEngineered[T](ctx, e, parts, o, attempt)
		if err == nil {
			return result, nil
		}
		var gdErr *GenerateDataError
		if errors.As(err, &gdErr) && gdErr.Phase == "generate" {
			return nil, err
		}
		lastErr = err
	}
	return nil, lastErr
}

// tryCaptureToolSilent runs the synthesized-tool capture path for T.
// Returns the captured value on success, nil otherwise. All failures
// (errCaptureToolUnsuitable, NewConversation failure, transport error, the
// model declining to call the tool) are silent so the caller can fall
// through to the prompt-engineered path. ctx cancellation is observed
// via the caller's subsequent ctx.Err() check.
func tryCaptureToolSilent[T any](ctx context.Context, e *Engine, parts []Part, o GenOptions) *T {
	if e == nil {
		return nil
	}
	tool, err := getOrSynthesizeCaptureTool[T]()
	if err != nil {
		return nil
	}

	var captured *T
	callCtx := context.WithValue(ctx, captureKey[T]{}, &captured)

	toolOpts := o
	toolOpts.Tools = []ToolDefinition{tool}
	toolOpts.MaxToolHops = 1

	conv, err := e.NewConversation(toolOpts)
	if err != nil {
		return nil
	}
	defer conv.Close()

	if _, err := conv.Send(callCtx, augmentForToolUse(parts)...); err != nil {
		return nil
	}
	return captured
}

// generateDataPromptEngineered is the fallback: augment the prompt
// with a JSON-shape instruction, generate, and tolerantly parse the
// response. attempt is 1-indexed and used solely for error reporting;
// the outer retry loop in GenerateDataMulti drives iteration.
func generateDataPromptEngineered[T any](
	ctx context.Context, e *Engine, parts []Part, o GenOptions, attempt int,
) (*T, error) {
	t := reflect.TypeFor[T]()
	shape, err := shapeOf(t)
	if err != nil {
		return nil, fmt.Errorf("lm: GenerateDataMulti: %w", err)
	}

	instruction := o.SchemaInstruction
	if instruction == "" {
		instruction = defaultSchemaInstruction
	}
	augmentedParts := injectSchema(parts, fmt.Sprintf(instruction, shape))
	wantArray := isArrayType(t)

	conv, err := e.NewConversation(o)
	if err != nil {
		return nil, &GenerateDataError{
			Phase:    "generate",
			Err:      err,
			Attempts: attempt,
		}
	}
	defer conv.Close()

	text, err := conv.Send(ctx, augmentedParts...)
	if err != nil {
		return nil, &GenerateDataError{
			Phase:    "generate",
			Err:      err,
			Attempts: attempt,
		}
	}

	extracted, err := extractJSON(text, wantArray)
	if err != nil {
		return nil, &GenerateDataError{
			Phase:    "parse",
			Err:      err,
			Raw:      text,
			Attempts: attempt,
		}
	}

	out := new(T)
	if err := json.Unmarshal([]byte(extracted), out); err != nil {
		return nil, &GenerateDataError{
			Phase:    "parse",
			Err:      err,
			Raw:      text,
			Attempts: attempt,
		}
	}
	return out, nil
}

// GenerateDataError describes a structured-output failure. Phase
// distinguishes "generate" (the underlying model call failed) from
// "parse" (the model produced text but it could not be unmarshalled
// into T). Raw holds the model output for parse-phase failures so
// callers can log or display the offending response.
type GenerateDataError struct {
	Phase    string // "generate" or "parse"
	Err      error
	Raw      string // populated on parse-phase failures
	Attempts int    // 1-indexed
}

func (e *GenerateDataError) Error() string {
	if e == nil {
		return "<nil GenerateDataError>"
	}
	return fmt.Sprintf("lm: GenerateData %s phase failed after %d attempt(s): %v",
		e.Phase, e.Attempts, e.Err)
}

func (e *GenerateDataError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// injectSchema returns a copy of parts with instruction prepended to
// the LAST text Part. When no text part exists, a new text Part is
// appended at the end.
func injectSchema(parts []Part, instruction string) []Part {
	out := append([]Part(nil), parts...)
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Kind == "text" || out[i].Kind == "" {
			out[i].Text = instruction + "\n\n" + out[i].Text
			return out
		}
	}
	return append(out, Part{Kind: "text", Text: instruction})
}

// augmentForToolUse returns a copy of parts with captureDirective
// prepended to the LAST text Part, mirroring injectSchema's placement
// rules so the directive lands close to the user's content. Appended
// as a new Text part when none exists.
func augmentForToolUse(parts []Part) []Part {
	out := append([]Part(nil), parts...)
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Kind == "text" || out[i].Kind == "" {
			out[i].Text = captureDirective + out[i].Text
			return out
		}
	}
	return append(out, Part{Kind: "text", Text: captureDirective})
}
