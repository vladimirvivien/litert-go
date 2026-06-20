package lm

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
)

// ---- types used across tests --------------------------------------------

type capturePerson struct {
	Name string `description:"the person's full name"`
	Age  int    `description:"age in years"`
}

type captureAddress struct {
	Street string
	City   string
}

// ---- captureToolName ----------------------------------------------------

func TestCaptureToolName_FormatAndPrefix(t *testing.T) {
	name := captureToolName[capturePerson]()
	if !strings.HasPrefix(name, reservedToolNamePrefix) {
		t.Errorf("captureToolName = %q; want prefix %q", name, reservedToolNamePrefix)
	}
	if !strings.HasPrefix(name, reservedToolNamePrefix+"capture_") {
		t.Errorf("captureToolName = %q; want %qcapture_<hash>", name, reservedToolNamePrefix)
	}
	if got := captureToolName[capturePerson](); got != name {
		t.Errorf("captureToolName not stable: %q vs %q", got, name)
	}
}

func TestCaptureToolName_DistinctPerType(t *testing.T) {
	a := captureToolName[capturePerson]()
	b := captureToolName[captureAddress]()
	if a == b {
		t.Errorf("distinct types produced the same capture name: %q", a)
	}
}

// ---- getOrSynthesizeCaptureTool -----------------------------------------

func TestGetOrSynthesizeCaptureTool_CachesByType(t *testing.T) {
	// Clear the global cache map first for independent test run.
	captureTools = sync.Map{}

	first, err := getOrSynthesizeCaptureTool[capturePerson]()
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	second, err := getOrSynthesizeCaptureTool[capturePerson]()
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if first != second {
		t.Errorf("getOrSynthesize did not cache: %p vs %p", first, second)
	}
}

func TestGetOrSynthesizeCaptureTool_DistinctTypes(t *testing.T) {
	captureTools = sync.Map{}

	p, err := getOrSynthesizeCaptureTool[capturePerson]()
	if err != nil {
		t.Fatalf("person: %v", err)
	}
	a, err := getOrSynthesizeCaptureTool[captureAddress]()
	if err != nil {
		t.Fatalf("address: %v", err)
	}
	if p.Name() == a.Name() {
		t.Errorf("distinct types share tool name %q", p.Name())
	}
}

func TestGetOrSynthesizeCaptureTool_PointerStructAccepted(t *testing.T) {
	captureTools = sync.Map{}

	tool, err := getOrSynthesizeCaptureTool[*capturePerson]()
	if err != nil {
		t.Fatalf("pointer-to-struct should be accepted: %v", err)
	}
	if tool.Parameters()["type"] != "object" {
		t.Errorf("parameters.type = %v, want object", tool.Parameters()["type"])
	}
}

func TestGetOrSynthesizeCaptureTool_RejectsNonStruct(t *testing.T) {
	captureTools = sync.Map{}

	cases := []func() error{
		func() error { _, err := getOrSynthesizeCaptureTool[string](); return err },
		func() error { _, err := getOrSynthesizeCaptureTool[int](); return err },
		func() error { _, err := getOrSynthesizeCaptureTool[[]capturePerson](); return err },
		func() error { _, err := getOrSynthesizeCaptureTool[map[string]capturePerson](); return err },
	}
	for i, fn := range cases {
		err := fn()
		if !errors.Is(err, errCaptureToolUnsuitable) {
			t.Errorf("case %d: err = %v; want errCaptureToolUnsuitable", i, err)
		}
	}
}

func TestGetOrSynthesizeCaptureTool_ConcurrentSameType(t *testing.T) {
	captureTools = sync.Map{}

	const N = 32
	var wg sync.WaitGroup
	results := make([]*ManagedTool[capturePerson, struct{}], N)
	errs := make([]error, N)
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = getOrSynthesizeCaptureTool[capturePerson]()
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("goroutine %d: %v", i, err)
		}
	}
	first := results[0]
	for i, r := range results[1:] {
		if r != first {
			t.Errorf("goroutine %d returned %p, want %p", i+1, r, first)
		}
	}
}

// ---- captureHandler -----------------------------------------------------

func TestCaptureHandler_WritesToCtxSlot(t *testing.T) {
	handler := captureHandler[capturePerson]()
	var slot *capturePerson
	ctx := context.WithValue(context.Background(), captureKey[capturePerson]{}, &slot)
	want := capturePerson{Name: "Alice", Age: 30}
	if _, err := handler(ctx, want); err != nil {
		t.Fatalf("handler: %v", err)
	}
	if slot == nil {
		t.Fatal("slot still nil after handler call")
	}
	if *slot != want {
		t.Errorf("captured = %+v, want %+v", *slot, want)
	}
}

func TestCaptureHandler_NoSlotInCtxIsNoop(t *testing.T) {
	handler := captureHandler[capturePerson]()
	if _, err := handler(context.Background(), capturePerson{Name: "Bob"}); err != nil {
		t.Errorf("handler with no slot returned error: %v", err)
	}
}

func TestCaptureHandler_WrongTypeSlotIgnored(t *testing.T) {
	handler := captureHandler[capturePerson]()
	var wrongSlot *captureAddress
	ctx := context.WithValue(context.Background(), captureKey[captureAddress]{}, &wrongSlot)
	if _, err := handler(ctx, capturePerson{Name: "Carol"}); err != nil {
		t.Errorf("handler should ignore foreign-type slots: %v", err)
	}
	if wrongSlot != nil {
		t.Errorf("foreign-type slot was written: %+v", wrongSlot)
	}
}

func TestCaptureHandler_ConcurrentDistinctSlots(t *testing.T) {
	handler := captureHandler[capturePerson]()
	const N = 16
	var wg sync.WaitGroup
	slots := make([]*capturePerson, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.WithValue(context.Background(), captureKey[capturePerson]{}, &slots[i])
			_, _ = handler(ctx, capturePerson{Name: "p", Age: i})
		}(i)
	}
	wg.Wait()
	for i, s := range slots {
		if s == nil {
			t.Errorf("slot %d not written", i)
			continue
		}
		if s.Age != i {
			t.Errorf("slot %d age = %d, want %d", i, s.Age, i)
		}
	}
}

// ---- GenerateDataError --------------------------------------------------

func TestGenerateDataError_Format(t *testing.T) {
	inner := errors.New("invalid character 'x'")
	e := &GenerateDataError{
		Phase:    "parse",
		Err:      inner,
		Raw:      "xyz",
		Attempts: 3,
	}
	got := e.Error()
	for _, want := range []string{"parse", "3 attempt", "invalid character"} {
		if !strings.Contains(got, want) {
			t.Errorf("Error() = %q, want substring %q", got, want)
		}
	}
}

func TestGenerateDataError_Unwrap(t *testing.T) {
	inner := errors.New("inner")
	e := &GenerateDataError{Phase: "parse", Err: inner}
	if !errors.Is(e, inner) {
		t.Errorf("errors.Is should reach inner via Unwrap")
	}

	wrapped := &GenerateDataError{
		Phase: "generate",
		Err:   inner,
	}
	var gd *GenerateDataError
	if !errors.As(wrapped, &gd) {
		t.Fatalf("errors.As should match *GenerateDataError")
	}
	if gd.Phase != "generate" {
		t.Errorf("As'd error phase = %q, want generate", gd.Phase)
	}
}

func TestGenerateDataError_NilSafe(t *testing.T) {
	var e *GenerateDataError
	if msg := e.Error(); msg == "" {
		t.Errorf("nil *GenerateDataError should still produce a sentinel string")
	}
	if got := e.Unwrap(); got != nil {
		t.Errorf("nil.Unwrap() = %v, want nil", got)
	}
}

// ---- injectSchema ------------------------------------------------------

func TestInjectSchema_PrependsToOnlyTextPart(t *testing.T) {
	parts := []Part{{Kind: "text", Text: "describe the image"}}
	out := injectSchema(parts, "INSTR")
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	want := "INSTR\n\ndescribe the image"
	if out[0].Text != want {
		t.Errorf("text = %q, want %q", out[0].Text, want)
	}
}

func TestInjectSchema_PicksLastTextPart(t *testing.T) {
	parts := []Part{
		{Kind: "text", Text: "system context"},
		{Kind: "image", Data: []byte{0xFF}},
		{Kind: "text", Text: "the user prompt"},
	}
	out := injectSchema(parts, "INSTR")
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[0].Text != "system context" {
		t.Errorf("first text mutated: %q", out[0].Text)
	}
	if !strings.HasPrefix(out[2].Text, "INSTR\n\n") {
		t.Errorf("last text not augmented: %q", out[2].Text)
	}
}

func TestInjectSchema_AppendsWhenNoText(t *testing.T) {
	parts := []Part{{Kind: "image", Data: []byte{0xFF}}, {Kind: "audio", Data: []byte("riff")}}
	out := injectSchema(parts, "INSTR")
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[2].Kind != "text" || out[2].Text != "INSTR" {
		t.Errorf("appended part = %+v, want Text(INSTR)", out[2])
	}
}

func TestInjectSchema_DoesNotMutateInput(t *testing.T) {
	parts := []Part{{Kind: "text", Text: "original"}}
	_ = injectSchema(parts, "INSTR")
	if parts[0].Text != "original" {
		t.Errorf("input mutated: %q", parts[0].Text)
	}
}

// ---- augmentForToolUse --------------------------------------------------

func TestAugmentForToolUse_PrependsToOnlyTextPart(t *testing.T) {
	parts := []Part{{Kind: "text", Text: "extract entities"}}
	out := augmentForToolUse(parts)
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	if !strings.HasPrefix(out[0].Text, captureDirective) {
		t.Errorf("missing directive prefix: %q", out[0].Text)
	}
	if !strings.HasSuffix(out[0].Text, "extract entities") {
		t.Errorf("original prompt missing from output: %q", out[0].Text)
	}
}

func TestAugmentForToolUse_PicksLastTextPart(t *testing.T) {
	parts := []Part{
		{Kind: "text", Text: "system context"},
		{Kind: "image", Data: []byte{0xFF}},
		{Kind: "text", Text: "the user prompt"},
	}
	out := augmentForToolUse(parts)
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[0].Text != "system context" {
		t.Errorf("first text mutated: %q", out[0].Text)
	}
	if !strings.HasPrefix(out[2].Text, captureDirective) {
		t.Errorf("last text not augmented: %q", out[2].Text)
	}
}

func TestAugmentForToolUse_AppendsWhenNoText(t *testing.T) {
	parts := []Part{{Kind: "image", Data: []byte{0xFF}}, {Kind: "audio", Data: []byte("riff")}}
	out := augmentForToolUse(parts)
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[2].Kind != "text" || out[2].Text != captureDirective {
		t.Errorf("appended part = %+v, want Text(captureDirective)", out[2])
	}
}

func TestAugmentForToolUse_DoesNotMutateInput(t *testing.T) {
	parts := []Part{{Kind: "text", Text: "original"}}
	_ = augmentForToolUse(parts)
	if parts[0].Text != "original" {
		t.Errorf("input mutated: %q", parts[0].Text)
	}
}

// ---- tryCaptureToolSilent ------------------------------------------------

func TestTryCaptureToolSilent_UnsuitableType(t *testing.T) {
	e := &Engine{}
	if got := tryCaptureToolSilent[[]capturePerson](context.Background(), e, []Part{{Kind: "text", Text: "x"}}, GenOptions{}); got != nil {
		t.Errorf("unsuitable T returned non-nil: %+v", got)
	}
}

func TestTryCaptureToolSilent_NilEngine(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("tryCaptureToolSilent panicked on nil engine: %v", r)
		}
	}()
	if got := tryCaptureToolSilent[capturePerson](context.Background(), nil, []Part{{Kind: "text", Text: "x"}}, GenOptions{}); got != nil {
		t.Errorf("nil engine returned non-nil: %+v", got)
	}
}
