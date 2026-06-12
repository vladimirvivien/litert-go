package lm_test

import (
	"context"
	"os"
	"testing"

	"github.com/vladimirvivien/litert-go/lm"
)

// Live function-calling round on a tool-capable model (gemma-4
// family). Requires LITERT_LIB and LITERT_LM_TOOL_MODEL (a gemma-4
// .litertlm); skips otherwise.
func TestToolRound(t *testing.T) {
	lib := os.Getenv("LITERT_LIB")
	model := os.Getenv("LITERT_LM_TOOL_MODEL")
	if lib == "" || model == "" {
		t.Skip("LITERT_LIB / LITERT_LM_TOOL_MODEL not set")
	}

	ctx := context.Background()
	e, err := lm.Open(ctx, model, lm.WithLibDir(lib))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer e.Close()

	const tools = `[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name."}},"required":["city"]}}}]`
	conv, err := e.NewConversation(lm.GenOptions{
		MaxTokens: 256,
		System:    "You are a helpful assistant.",
		ToolsJSON: tools,
	})
	if err != nil {
		t.Fatalf("NewConversation: %v", err)
	}
	defer conv.Close()

	reply, err := conv.Send(ctx, "What is the weather in Paris?")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	text, calls, err := e.ExtractToolCalls(reply)
	if err != nil {
		t.Fatalf("ExtractToolCalls: %v (reply=%q)", err, reply)
	}
	if len(calls) != 1 || calls[0].Name != "get_weather" {
		t.Fatalf("calls = %+v (text=%q), want one get_weather call", calls, text)
	}
	if city, _ := calls[0].Args["city"].(string); city != "Paris" {
		t.Errorf("city arg = %v, want Paris", calls[0].Args["city"])
	}

	sender, ok := conv.(lm.ToolSender)
	if !ok {
		t.Fatalf("conversation %T does not implement ToolSender", conv)
	}
	final, err := sender.SendToolResults(ctx, []lm.ToolResult{{
		Name:     "get_weather",
		Response: map[string]any{"temp_c": 21, "sky": "clear"},
	}})
	if err != nil {
		t.Fatalf("SendToolResults: %v", err)
	}
	t.Logf("final: %q", final)
	if final == "" {
		t.Fatal("empty final reply")
	}
	// Greedy cross-engine anchor: the C++ engine's reply for this
	// exact tool round on gemma-4-E2B-it.
	const anchor = "The weather in Paris is clear with a temperature of 21°C."
	if final != anchor {
		t.Errorf("final = %q, want C++ anchor %q", final, anchor)
	}
}
