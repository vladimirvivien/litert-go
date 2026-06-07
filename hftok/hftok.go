// Package hftok is a pure-Go byte-level BPE tokenizer for HuggingFace
// tokenizer.json models (Qwen2/Qwen3, GPT-2 family). It loads the tokenizer
// from a .litertlm HF_Tokenizer_Zlib section (an 8-byte little-endian
// uncompressed-size header followed by a zlib stream) and provides Encode /
// Decode over token IDs.
//
// Pipeline: text is first split on the model's added (special) tokens, which are
// emitted as single IDs; each remaining span is pre-tokenized with the model's
// regex, byte-level encoded (each UTF-8 byte mapped to a printable rune), and
// merged by BPE rank. Decode reverses this.
package hftok

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"

	"github.com/dlclark/regexp2"
)

// Tokenizer is a loaded byte-level BPE tokenizer.
type Tokenizer struct {
	vocab     map[string]int32 // byte-level token string -> id
	idToTok   map[int32]string // id -> byte-level token string
	ranks     map[[2]string]int
	special   map[string]int32 // added-token content -> id
	specialID map[int32]string
	pre       *regexp2.Regexp // pre-tokenizer split pattern
	splitter  *regexp.Regexp  // added-token alternation (nil if none)
	b2r       [256]rune
	r2b       map[rune]byte
}

type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Vocab  map[string]int32  `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
}

// LoadSection decompresses a HF_Tokenizer_Zlib section and loads it. The section
// is an 8-byte little-endian uncompressed size followed by a zlib stream.
func LoadSection(section []byte) (*Tokenizer, error) {
	js, err := decompress(section)
	if err != nil {
		return nil, err
	}
	return Load(js)
}

func decompress(section []byte) ([]byte, error) {
	if len(section) < 10 {
		return nil, fmt.Errorf("hftok: section too short (%d bytes)", len(section))
	}
	want := binary.LittleEndian.Uint64(section[:8])
	zr, err := zlib.NewReader(bytes.NewReader(section[8:]))
	if err != nil {
		return nil, fmt.Errorf("hftok: zlib: %w", err)
	}
	defer zr.Close()
	js, err := io.ReadAll(zr)
	if err != nil {
		return nil, fmt.Errorf("hftok: inflate: %w", err)
	}
	if uint64(len(js)) != want {
		return nil, fmt.Errorf("hftok: size mismatch (header %d, got %d)", want, len(js))
	}
	return js, nil
}

// Load builds a Tokenizer from decompressed tokenizer.json bytes.
func Load(js []byte) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(js, &tj); err != nil {
		return nil, fmt.Errorf("hftok: parse tokenizer.json: %w", err)
	}
	if len(tj.Model.Vocab) == 0 {
		return nil, fmt.Errorf("hftok: empty vocab")
	}

	t := &Tokenizer{
		vocab:     tj.Model.Vocab,
		idToTok:   make(map[int32]string, len(tj.Model.Vocab)),
		ranks:     make(map[[2]string]int, len(tj.Model.Merges)),
		special:   map[string]int32{},
		specialID: map[int32]string{},
		r2b:       make(map[rune]byte, 256),
	}
	for tok, id := range tj.Model.Vocab {
		t.idToTok[id] = tok
	}
	for i, m := range tj.Model.Merges {
		l, r, err := parseMerge(m)
		if err != nil {
			return nil, err
		}
		t.ranks[[2]string{l, r}] = i
	}
	for _, a := range tj.AddedTokens {
		t.special[a.Content] = a.ID
		t.specialID[a.ID] = a.Content
		t.idToTok[a.ID] = a.Content
	}
	t.buildByteMaps()

	pat, err := splitPattern(tj.PreTokenizer)
	if err != nil {
		return nil, err
	}
	if t.pre, err = regexp2.Compile(pat, 0); err != nil {
		return nil, fmt.Errorf("hftok: pre-tokenizer regex: %w", err)
	}
	if sp := t.splitterPattern(); sp != "" {
		t.splitter = regexp.MustCompile(sp)
	}
	return t, nil
}

// parseMerge accepts both HF merge encodings: a ["left","right"] array (current)
// and a "left right" string (legacy).
func parseMerge(raw json.RawMessage) (string, string, error) {
	if len(raw) > 0 && raw[0] == '[' {
		var pair []string
		if err := json.Unmarshal(raw, &pair); err != nil || len(pair) != 2 {
			return "", "", fmt.Errorf("hftok: bad merge %s", raw)
		}
		return pair[0], pair[1], nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", "", fmt.Errorf("hftok: bad merge %s", raw)
	}
	i := strings.IndexByte(s, ' ')
	if i < 0 {
		return "", "", fmt.Errorf("hftok: merge has no space: %q", s)
	}
	return s[:i], s[i+1:], nil
}

// splitPattern extracts the Split regex from the pre_tokenizer (a single Split,
// or the first Split inside a Sequence). The ByteLevel stage is applied in code.
func splitPattern(raw json.RawMessage) (string, error) {
	var probe struct {
		Type          string                 `json:"type"`
		Pattern       struct{ Regex string } `json:"pattern"`
		PreTokenizers []struct {
			Type    string                 `json:"type"`
			Pattern struct{ Regex string } `json:"pattern"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return "", fmt.Errorf("hftok: pre_tokenizer: %w", err)
	}
	if probe.Type == "Split" && probe.Pattern.Regex != "" {
		return probe.Pattern.Regex, nil
	}
	for _, p := range probe.PreTokenizers {
		if p.Type == "Split" && p.Pattern.Regex != "" {
			return p.Pattern.Regex, nil
		}
	}
	return "", fmt.Errorf("hftok: no Split pattern in pre_tokenizer")
}

// splitterPattern builds an alternation of the added-token contents, longest
// first so the longest token wins at any position.
func (t *Tokenizer) splitterPattern() string {
	if len(t.special) == 0 {
		return ""
	}
	keys := make([]string, 0, len(t.special))
	for k := range t.special {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return len(keys[i]) > len(keys[j]) })
	quoted := make([]string, len(keys))
	for i, k := range keys {
		quoted[i] = regexp.QuoteMeta(k)
	}
	return strings.Join(quoted, "|")
}

// buildByteMaps builds the GPT-2 byte<->rune tables: printable bytes map to
// themselves, the rest to runes 256+.
func (t *Tokenizer) buildByteMaps() {
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	n := 0
	for b := 0; b < 256; b++ {
		var r rune
		if printable(b) {
			r = rune(b)
		} else {
			r = rune(256 + n)
			n++
		}
		t.b2r[b] = r
		t.r2b[r] = byte(b)
	}
}

// Encode tokenizes text into token IDs.
func (t *Tokenizer) Encode(text string) []int32 {
	var ids []int32
	t.eachSpan(text, func(span string, special bool) {
		if special {
			ids = append(ids, t.special[span])
			return
		}
		for _, piece := range t.preTokenize(span) {
			ids = append(ids, t.bpeIDs(piece)...)
		}
	})
	return ids
}

// eachSpan splits text into added-token spans (special) and the spans between
// them (normal), in order.
func (t *Tokenizer) eachSpan(text string, fn func(span string, special bool)) {
	if t.splitter == nil {
		fn(text, false)
		return
	}
	last := 0
	for _, loc := range t.splitter.FindAllStringIndex(text, -1) {
		if loc[0] > last {
			fn(text[last:loc[0]], false)
		}
		fn(text[loc[0]:loc[1]], true)
		last = loc[1]
	}
	if last < len(text) {
		fn(text[last:], false)
	}
}

// preTokenize applies the model's split regex, returning the matched pieces.
func (t *Tokenizer) preTokenize(s string) []string {
	var out []string
	m, _ := t.pre.FindStringMatch(s)
	for m != nil {
		if m.Length > 0 {
			out = append(out, m.String())
		}
		m, _ = t.pre.FindNextMatch(m)
	}
	return out
}

// bpeIDs byte-encodes a pre-token piece and applies BPE merges, returning IDs.
func (t *Tokenizer) bpeIDs(piece string) []int32 {
	symbols := make([]string, 0, len(piece))
	for i := 0; i < len(piece); i++ {
		symbols = append(symbols, string(t.b2r[piece[i]]))
	}
	symbols = t.merge(symbols)
	ids := make([]int32, 0, len(symbols))
	for _, s := range symbols {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// merge repeatedly joins the adjacent symbol pair with the lowest merge rank.
func (t *Tokenizer) merge(symbols []string) []string {
	for len(symbols) >= 2 {
		bestRank, bestI := int(^uint(0)>>1), -1
		for i := 0; i+1 < len(symbols); i++ {
			if r, ok := t.ranks[[2]string{symbols[i], symbols[i+1]}]; ok && r < bestRank {
				bestRank, bestI = r, i
			}
		}
		if bestI < 0 {
			break
		}
		symbols[bestI] = symbols[bestI] + symbols[bestI+1]
		symbols = append(symbols[:bestI+1], symbols[bestI+2:]...)
	}
	return symbols
}

// Decode turns token IDs back into text.
func (t *Tokenizer) Decode(ids []int) string {
	var b []byte
	var sb strings.Builder
	flush := func() {
		if len(b) > 0 {
			sb.Write(b)
			b = b[:0]
		}
	}
	for _, id := range ids {
		i32 := int32(id)
		if content, ok := t.specialID[i32]; ok {
			flush()
			sb.WriteString(content)
			continue
		}
		tok, ok := t.idToTok[i32]
		if !ok {
			continue
		}
		for _, r := range tok {
			if by, ok := t.r2b[r]; ok {
				b = append(b, by)
			}
		}
	}
	flush()
	return sb.String()
}
