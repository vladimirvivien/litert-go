package lm

import "errors"

// Sentinel errors for expected conditions. Call sites wrap them with detail;
// branch with errors.Is.
var (
	// ErrNoTokenizer: the model carries no tokenizer section. Token-level
	// APIs (GenerateIDs) still work.
	ErrNoTokenizer = errors.New("lm: model has no tokenizer")

	// ErrNoChatTemplate: the model's metadata has no chat affixes, so
	// chat-mode generation and conversations cannot render turns.
	ErrNoChatTemplate = errors.New("lm: model has no chat template")

	// ErrNotEmbeddingModel: the operation needs an embedding-input model
	// (gemma 3n/4 class) and the loaded model is token-input.
	ErrNotEmbeddingModel = errors.New("lm: model is not embedding-input")

	// ErrSpecUnsupported: speculative decoding cannot run in this
	// configuration.
	ErrSpecUnsupported = errors.New("lm: speculative decoding not supported")
)
