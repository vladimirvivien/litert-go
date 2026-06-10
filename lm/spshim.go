package lm

import (
	"bytes"
	"encoding/binary"
)

// HF-converted BPE SentencePiece models (Qwen3, Phi-4) declare
// escape_whitespaces=false and carry raw-space pieces (" quick").
// go-sentencepiece implements only the ▁ convention: it maps spaces to ▁
// on encode and ▁ to spaces on decode, so on these vocabs every prompt
// space falls to unk. rewriteRawSpacePieces adapts the model instead of the
// library: every space byte inside a piece string becomes "▁" (U+2581),
// after which go-sentencepiece's encode matches the vocab and its decode
// restores the text. Models with escape_whitespaces=true (gemma) and
// unparseable protos are returned unchanged.
//
// sentencepiece_model.proto wire layout used here:
//
//	ModelProto:      1 pieces (msg, repeated), 3 normalizer_spec (msg)
//	SentencePiece:   1 piece (string)
//	NormalizerSpec:  5 escape_whitespaces (bool, default true)
func rewriteRawSpacePieces(sp []byte) []byte {
	if spEscapesWhitespace(sp) {
		return sp
	}
	out := make([]byte, 0, len(sp)+len(sp)/8)
	i := 0
	for i < len(sp) {
		field, wt, val, end, ok := spField(sp, i)
		if !ok {
			return sp
		}
		if field == 1 && wt == 2 && bytes.IndexByte(val, ' ') >= 0 {
			piece, pok := spRewritePiece(val)
			if !pok {
				return sp
			}
			out = append(out, 0x0a) // tag: field 1, wire type 2
			out = binary.AppendUvarint(out, uint64(len(piece)))
			out = append(out, piece...)
		} else {
			out = append(out, sp[i:end]...)
		}
		i = end
	}
	return out
}

// spEscapesWhitespace reads normalizer_spec.escape_whitespaces (proto2
// default: true). Unparseable input reports true so the caller leaves it
// unchanged.
func spEscapesWhitespace(sp []byte) bool {
	i := 0
	for i < len(sp) {
		field, wt, val, end, ok := spField(sp, i)
		if !ok {
			return true
		}
		if field == 3 && wt == 2 {
			j := 0
			for j < len(val) {
				f, w, v, e, ok := spField(val, j)
				if !ok {
					return true
				}
				if f == 5 && w == 0 {
					return len(v) > 0 && v[0] != 0
				}
				j = e
			}
			return true
		}
		i = end
	}
	return true
}

// spRewritePiece re-encodes one SentencePiece submessage, replacing spaces
// in its piece string (field 1) with ▁.
func spRewritePiece(msg []byte) ([]byte, bool) {
	out := make([]byte, 0, len(msg)+8)
	i := 0
	for i < len(msg) {
		field, wt, val, end, ok := spField(msg, i)
		if !ok {
			return nil, false
		}
		if field == 1 && wt == 2 {
			s := bytes.ReplaceAll(val, []byte(" "), []byte("▁"))
			out = append(out, 0x0a)
			out = binary.AppendUvarint(out, uint64(len(s)))
			out = append(out, s...)
		} else {
			out = append(out, msg[i:end]...)
		}
		i = end
	}
	return out, true
}

// spField parses the protobuf field starting at i, returning its number,
// wire type, value bytes (length-delimited and varint fields), and the index
// past the field.
func spField(b []byte, i int) (field uint64, wt uint64, val []byte, end int, ok bool) {
	tag, n := binary.Uvarint(b[i:])
	if n <= 0 {
		return 0, 0, nil, 0, false
	}
	i += n
	field, wt = tag>>3, tag&7
	switch wt {
	case 0: // varint
		v, m := binary.Uvarint(b[i:])
		if m <= 0 {
			return 0, 0, nil, 0, false
		}
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], v)
		return field, wt, buf[:], i + m, true
	case 1: // 64-bit
		if i+8 > len(b) {
			return 0, 0, nil, 0, false
		}
		return field, wt, b[i : i+8], i + 8, true
	case 2: // length-delimited
		l, m := binary.Uvarint(b[i:])
		if m <= 0 || i+m+int(l) > len(b) {
			return 0, 0, nil, 0, false
		}
		return field, wt, b[i+m : i+m+int(l)], i + m + int(l), true
	case 5: // 32-bit
		if i+4 > len(b) {
			return 0, 0, nil, 0, false
		}
		return field, wt, b[i : i+4], i + 4, true
	}
	return 0, 0, nil, 0, false
}
