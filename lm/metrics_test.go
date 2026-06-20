package lm_test

import (
	"context"
	"testing"

	"github.com/vladimirvivien/litert-go/lm"
)

func TestPerformanceMetrics(t *testing.T) {
	eng := openEngine(t)
	o := lm.GenOptions{MaxTokens: 8}
	
	// Test GenerateWithMetrics
	text, metrics, err := eng.GenerateWithMetrics(context.Background(), "Hello, who are you?", false, o)
	if err != nil {
		t.Fatalf("GenerateWithMetrics: %v", err)
	}
	if text == "" {
		t.Fatal("GenerateWithMetrics returned empty text")
	}
	
	t.Logf("Generated text: %q", text)
	t.Logf("Metrics: %+v", metrics)
	
	if metrics.PrefillDuration <= 0 {
		t.Errorf("expected PrefillDuration > 0, got %v", metrics.PrefillDuration)
	}
	if metrics.DecodeDuration <= 0 {
		t.Errorf("expected DecodeDuration > 0, got %v", metrics.DecodeDuration)
	}
	if metrics.TimeToFirstToken <= 0 {
		t.Errorf("expected TimeToFirstToken > 0, got %v", metrics.TimeToFirstToken)
	}
	if metrics.PrefillTokens <= 0 {
		t.Errorf("expected PrefillTokens > 0, got %d", metrics.PrefillTokens)
	}
	if metrics.DecodeTokens <= 0 {
		t.Errorf("expected DecodeTokens > 0, got %d", metrics.DecodeTokens)
	}
	if metrics.TokensPerSecond <= 0 {
		t.Errorf("expected TokensPerSecond > 0, got %f", metrics.TokensPerSecond)
	}
	
	// Test LastMetrics accessor on Engine
	last := eng.LastMetrics()
	if last.PrefillTokens != metrics.PrefillTokens || last.DecodeTokens != metrics.DecodeTokens {
		t.Errorf("LastMetrics() = %+v, does not match GenerateWithMetrics return %+v", last, metrics)
	}
	
	// Test metrics collection during a Conversation session (KV-reuse)
	if !eng.HasChatTemplate() {
		return
	}
	
	conv, err := eng.NewConversation(lm.GenOptions{MaxTokens: 8})
	if err != nil {
		t.Fatalf("NewConversation: %v", err)
	}
	defer conv.Close()
	
	// First turn (prefill + decode)
	_, err = conv.Send(context.Background(), lm.Part{Text: "Name a primary color."})
	if err != nil {
		t.Fatalf("Conversation turn 1 Send: %v", err)
	}
	metricsTurn1 := eng.LastMetrics()
	t.Logf("Turn 1 Metrics: %+v", metricsTurn1)
	if metricsTurn1.CacheHits != 0 {
		t.Errorf("expected turn 1 CacheHits to be 0, got %d", metricsTurn1.CacheHits)
	}
	if metricsTurn1.PrefillTokens <= 0 {
		t.Errorf("expected turn 1 PrefillTokens > 0, got %d", metricsTurn1.PrefillTokens)
	}
	
	// Second turn (should reuse KV cache, so CacheHits > 0)
	_, err = conv.Send(context.Background(), lm.Part{Text: "Now name another one."})
	if err != nil {
		t.Fatalf("Conversation turn 2 Send: %v", err)
	}
	metricsTurn2 := eng.LastMetrics()
	t.Logf("Turn 2 Metrics: %+v", metricsTurn2)
	if metricsTurn2.CacheHits <= 0 {
		t.Errorf("expected turn 2 CacheHits > 0 due to KV-cache reuse, got %d", metricsTurn2.CacheHits)
	}
}
