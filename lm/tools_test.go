package lm

import (
	"reflect"
	"testing"

	"github.com/vladimirvivien/litert-go/litertlm"
)

var gemma4TT, _ = litertlm.ToolTemplatesFor(litertlm.ModelGemma4)

// Ground truth captured from the C++ LiteRT-LM v0.13.1 engine's
// Conversation.RenderMessage on gemma-4-E2B-it.
const (
	weatherTool = `[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name."}},"required":["city"]}}}]`
	weatherDecl = `<|tool>declaration:get_weather{description:<|"|>Get current weather for a city.<|"|>,parameters:{properties:{city:{description:<|"|>City name.<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`

	twoTools = `[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name."}},"required":["city"]}}},{"type":"function","function":{"name":"add","description":"Add two integers.","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}]`
	addDecl  = `<|tool>declaration:add{description:<|"|>Add two integers.<|"|>,parameters:{properties:{a:{type:<|"|>INTEGER<|"|>},b:{type:<|"|>INTEGER<|"|>}},required:[<|"|>a<|"|>,<|"|>b<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`

	weatherResp = `<|tool_response>response:get_weather{sky:<|"|>clear<|"|>,temp_c:21}<tool_response|>`
)

func TestFcDeclarations_MatchesCppRender(t *testing.T) {
	got, err := fcDeclarations(weatherTool, gemma4TT)
	if err != nil {
		t.Fatalf("fcDeclarations: %v", err)
	}
	if got != weatherDecl {
		t.Errorf("declaration mismatch\n got: %s\nwant: %s", got, weatherDecl)
	}

	got, err = fcDeclarations(twoTools, gemma4TT)
	if err != nil {
		t.Fatalf("fcDeclarations(two): %v", err)
	}
	if want := weatherDecl + addDecl; got != want {
		t.Errorf("two-tool mismatch\n got: %s\nwant: %s", got, want)
	}
}

func TestFcToolResponses_MatchesCppRender(t *testing.T) {
	got, err := fcToolResponses([]ToolResult{{
		Name:     "get_weather",
		Response: map[string]any{"temp_c": 21, "sky": "clear"},
	}}, gemma4TT)
	if err != nil {
		t.Fatalf("fcToolResponses: %v", err)
	}
	if got != weatherResp {
		t.Errorf("response mismatch\n got: %s\nwant: %s", got, weatherResp)
	}
}

func TestExtractToolCalls(t *testing.T) {
	tests := []struct {
		name      string
		reply     string
		wantText  string
		wantCalls []ToolCall
	}{
		{
			name:  "single call with string arg",
			reply: `<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>`,
			wantCalls: []ToolCall{
				{Name: "get_weather", Args: map[string]any{"city": "Paris"}},
			},
		},
		{
			name:  "numeric args",
			reply: `<|tool_call>call:add{a:17,b:25}<tool_call|>`,
			wantCalls: []ToolCall{
				{Name: "add", Args: map[string]any{"a": float64(17), "b": float64(25)}},
			},
		},
		{
			name:  "no-arg call",
			reply: `<|tool_call>call:refresh<tool_call|>`,
			wantCalls: []ToolCall{
				{Name: "refresh", Args: map[string]any{}},
			},
		},
		{
			name:     "text around call",
			reply:    "Let me check.\n" + `<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>` + "\n",
			wantText: "Let me check.",
			wantCalls: []ToolCall{
				{Name: "get_weather", Args: map[string]any{"city": "Paris"}},
			},
		},
		{
			name:  "nested object and array",
			reply: `<|tool_call>call:plan{steps:[<|"|>a<|"|>,<|"|>b<|"|>],meta:{depth:2,dry:true}}<tool_call|>`,
			wantCalls: []ToolCall{
				{Name: "plan", Args: map[string]any{
					"steps": []any{"a", "b"},
					"meta":  map[string]any{"depth": float64(2), "dry": true},
				}},
			},
		},
		{
			name:     "plain text passes through",
			reply:    "The answer is 42.",
			wantText: "The answer is 42.",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			text, calls, err := extractToolCalls(tc.reply, gemma4TT)
			if err != nil {
				t.Fatalf("extractToolCalls: %v", err)
			}
			if text != tc.wantText {
				t.Errorf("text = %q, want %q", text, tc.wantText)
			}
			if !reflect.DeepEqual(calls, tc.wantCalls) {
				t.Errorf("calls = %#v, want %#v", calls, tc.wantCalls)
			}
		})
	}
}

func TestExtractToolCalls_RoundTripsResponseQuote(t *testing.T) {
	// A reply whose argument value contains characters the FC writer
	// must round-trip (spaces, punctuation — not the quote token).
	reply := `<|tool_call>call:note{text:<|"|>hello, world: {ok}<|"|>}<tool_call|>`
	_, calls, err := extractToolCalls(reply, gemma4TT)
	if err != nil {
		t.Fatalf("extractToolCalls: %v", err)
	}
	if calls[0].Args["text"] != "hello, world: {ok}" {
		t.Errorf("text arg = %q", calls[0].Args["text"])
	}
}
