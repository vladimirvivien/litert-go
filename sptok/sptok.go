package sptok

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

type Tokenizer struct {
	spec        *ModelSpec
	pieces      map[string]int32
	vocab       []SentencePiece
	vocabScore  []float32
	vocabType   []PieceType
	addedTokens map[string]int32
	bytePieces  map[byte]int32
	unkID       int32
	bosID       int32
	eosID       int32
	padID       int32
}

type chunk struct {
	text string
	id   int32
}

type bpeNode struct {
	text string
	id   int32
	prev *bpeNode
	next *bpeNode
}

func bytePiece(b byte) string {
	return fmt.Sprintf("<0x%02X>", b)
}

func New(modelProto []byte) (*Tokenizer, error) {
	spec, err := parseModelSpec(modelProto)
	if err != nil {
		return nil, err
	}
	if spec.ModelType != ModelTypeBPE {
		return nil, fmt.Errorf("sptok: only BPE models (type 2) are supported, got model type %d", spec.ModelType)
	}
	if spec.HasPrecompiledCharsmap {
		return nil, fmt.Errorf("sptok: model uses precompiled charsmap which is unsupported")
	}

	t := &Tokenizer{
		spec:        spec,
		pieces:      make(map[string]int32),
		vocab:       spec.Pieces,
		vocabScore:  make([]float32, len(spec.Pieces)),
		vocabType:   make([]PieceType, len(spec.Pieces)),
		addedTokens: make(map[string]int32),
		bytePieces:  make(map[byte]int32),
		unkID:       int32(spec.UnkID),
		bosID:       int32(spec.BosID),
		eosID:       int32(spec.EosID),
		padID:       int32(spec.PadID),
	}

	for id, p := range spec.Pieces {
		t.pieces[p.Piece] = int32(id)
		t.vocabScore[id] = p.Score
		pType := p.Type
		if pType == 0 {
			pType = PieceTypeNormal
		}
		t.vocabType[id] = pType

		if pType == PieceTypeControl || pType == PieceTypeUserDefined {
			t.addedTokens[p.Piece] = int32(id)
		}
	}

	for i := 0; i < 256; i++ {
		b := byte(i)
		p := bytePiece(b)
		if id, ok := t.pieces[p]; ok {
			t.bytePieces[b] = id
		}
	}

	return t, nil
}

func (t *Tokenizer) BOS() int32 {
	return t.bosID
}

func (t *Tokenizer) EOS() int32 {
	return t.eosID
}

func (t *Tokenizer) splitSpecial(text string) []chunk {
	if len(t.addedTokens) == 0 {
		return []chunk{{text: text, id: -1}}
	}

	var chunks []chunk
	remaining := text
	for len(remaining) > 0 {
		bestIndex := -1
		bestLength := 0
		bestID := int32(-1)

		for token, id := range t.addedTokens {
			idx := strings.Index(remaining, token)
			if idx != -1 {
				if bestIndex == -1 || idx < bestIndex || (idx == bestIndex && len(token) > bestLength) {
					bestIndex = idx
					bestLength = len(token)
					bestID = id
				}
			}
		}

		if bestIndex == -1 {
			chunks = append(chunks, chunk{text: remaining, id: -1})
			break
		}

		if bestIndex > 0 {
			chunks = append(chunks, chunk{text: remaining[:bestIndex], id: -1})
		}
		chunks = append(chunks, chunk{text: remaining[bestIndex : bestIndex+bestLength], id: bestID})
		remaining = remaining[bestIndex+bestLength:]
	}
	return chunks
}

func (t *Tokenizer) normalize(text string) string {
	if t.spec.RemoveExtraWhitespaces {
		var buf bytes.Buffer
		lastWasSpace := false
		for _, r := range text {
			if r == ' ' {
				if !lastWasSpace {
					buf.WriteRune(' ')
					lastWasSpace = true
				}
			} else {
				buf.WriteRune(r)
				lastWasSpace = false
			}
		}
		text = buf.String()
	}

	if t.spec.AddDummyPrefix {
		text = " " + text
	}

	if t.spec.EscapeWhitespaces {
		text = strings.ReplaceAll(text, " ", "▁")
	}

	return text
}

func (t *Tokenizer) Encode(text string) []int32 {
	chunks := t.splitSpecial(text)
	var ids []int32

	for _, c := range chunks {
		if c.id != -1 {
			ids = append(ids, c.id)
			continue
		}

		normalized := t.normalize(c.text)
		if len(normalized) == 0 {
			continue
		}

		runes := []rune(normalized)
		var head, tail *bpeNode

		for _, r := range runes {
			s := string(r)
			id, ok := t.pieces[s]

			if ok {
				node := &bpeNode{text: s, id: id}
				if head == nil {
					head = node
					tail = node
				} else {
					tail.next = node
					node.prev = tail
					tail = node
				}
			} else if t.spec.ByteFallback {
				var buf [4]byte
				n := utf8.EncodeRune(buf[:], r)
				for j := 0; j < n; j++ {
					b := buf[j]
					bId, bOk := t.bytePieces[b]
					var node *bpeNode
					if bOk {
						node = &bpeNode{text: string([]byte{b}), id: bId}
					} else {
						node = &bpeNode{text: string([]byte{b}), id: t.unkID}
					}
					if head == nil {
						head = node
						tail = node
					} else {
						tail.next = node
						node.prev = tail
						tail = node
					}
				}
			} else {
				node := &bpeNode{text: s, id: t.unkID}
				if head == nil {
					head = node
					tail = node
				} else {
					tail.next = node
					node.prev = tail
					tail = node
				}
			}
		}

		// BPE merge loop
		for {
			var bestNode *bpeNode
			var bestScore float32 = -math.MaxFloat32
			var bestID int32 = -1
			var bestText string

			curr := head
			for curr != nil && curr.next != nil {
				pairText := curr.text + curr.next.text
				if id, ok := t.pieces[pairText]; ok {
					pType := t.vocabType[id]
					if pType == PieceTypeNormal || pType == PieceTypeUnknown || pType == PieceTypeByte || pType == PieceTypeUnused {
						score := t.vocabScore[id]
						if score > bestScore {
							bestScore = score
							bestNode = curr
							bestID = id
							bestText = pairText
						}
					}
				}
				curr = curr.next
			}

			if bestNode == nil {
				break
			}

			next := bestNode.next
			bestNode.text = bestText
			bestNode.id = bestID
			bestNode.next = next.next
			if next.next != nil {
				next.next.prev = bestNode
			}
		}

		// Gather IDs
		curr := head
		for curr != nil {
			ids = append(ids, curr.id)
			curr = curr.next
		}
	}

	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	var buf bytes.Buffer
	var byteBuf []byte

	flushBytes := func() {
		if len(byteBuf) > 0 {
			buf.Write(byteBuf)
			byteBuf = byteBuf[:0]
		}
	}

	for _, id := range ids {
		if id < 0 || id >= len(t.vocab) {
			continue
		}
		p := t.vocab[id]
		if p.Type == PieceTypeControl {
			continue
		}
		if p.Type == PieceTypeUnknown {
			flushBytes()
			buf.WriteString("<unk>")
			continue
		}
		if p.Type == PieceTypeByte {
			if len(p.Piece) == 6 && strings.HasPrefix(p.Piece, "<0x") && p.Piece[5] == '>' {
				hexStr := p.Piece[3:5]
				if b, err := strconv.ParseUint(hexStr, 16, 8); err == nil {
					byteBuf = append(byteBuf, byte(b))
					continue
				}
			}
		}

		flushBytes()
		buf.WriteString(p.Piece)
	}
	flushBytes()

	decoded := buf.String()
	if t.spec.EscapeWhitespaces {
		decoded = strings.ReplaceAll(decoded, "▁", " ")
	}

	if t.spec.AddDummyPrefix && len(decoded) > 0 && decoded[0] == ' ' {
		decoded = decoded[1:]
	}

	return decoded
}
