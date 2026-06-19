package sptok

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func makeModelBytes(pieces []SentencePiece, modelType ModelType, escape bool, fallback bool, dummyPrefix bool) []byte {
	var buf []byte

	// 1. Pieces
	for _, p := range pieces {
		var sub []byte
		// Field 1: Piece
		sub = append(sub, 0x0a)
		sub = binary.AppendUvarint(sub, uint64(len(p.Piece)))
		sub = append(sub, p.Piece...)
		// Field 2: Score
		sub = append(sub, 0x15)
		var fbits [4]byte
		binary.LittleEndian.PutUint32(fbits[:], math.Float32bits(p.Score))
		sub = append(sub, fbits[:]...)
		// Field 3: Type
		pType := p.Type
		if pType == 0 {
			pType = PieceTypeNormal
		}
		if pType != PieceTypeNormal {
			sub = append(sub, 0x18)
			sub = binary.AppendUvarint(sub, uint64(pType))
		}

		buf = append(buf, 0x0a)
		buf = binary.AppendUvarint(buf, uint64(len(sub)))
		buf = append(buf, sub...)
	}

	// 2. TrainerSpec
	var trainer []byte
	trainer = append(trainer, 0x18)
	trainer = binary.AppendUvarint(trainer, uint64(modelType))
	if fallback {
		trainer = append(trainer, 0x98, 0x02) // field 35 varint
		trainer = append(trainer, 1)
	}

	buf = append(buf, 0x12)
	buf = binary.AppendUvarint(buf, uint64(len(trainer)))
	buf = append(buf, trainer...)

	// 3. NormalizerSpec
	var normalizer []byte
	normalizer = append(normalizer, 0x18) // field 3 varint (add_dummy_prefix)
	if dummyPrefix {
		normalizer = append(normalizer, 1)
	} else {
		normalizer = append(normalizer, 0)
	}
	normalizer = append(normalizer, 0x28) // field 5 varint (escape_whitespaces)
	if escape {
		normalizer = append(normalizer, 1)
	} else {
		normalizer = append(normalizer, 0)
	}

	buf = append(buf, 0x1a)
	buf = binary.AppendUvarint(buf, uint64(len(normalizer)))
	buf = append(buf, normalizer...)

	return buf
}

func TestProtoParser(t *testing.T) {
	pieces := []SentencePiece{
		{Piece: "<unk>", Score: 0.0, Type: PieceTypeUnknown},
		{Piece: "▁", Score: -1.0, Type: PieceTypeNormal},
		{Piece: "hello", Score: -2.0, Type: PieceTypeNormal},
	}
	pb := makeModelBytes(pieces, ModelTypeBPE, true, true, true)

	spec, err := parseModelSpec(pb)
	if err != nil {
		t.Fatal(err)
	}

	if spec.ModelType != ModelTypeBPE {
		t.Errorf("model type = %v, want %v", spec.ModelType, ModelTypeBPE)
	}
	if !spec.ByteFallback {
		t.Errorf("byte fallback not set")
	}
	if !spec.EscapeWhitespaces {
		t.Errorf("escape whitespaces not set")
	}
	if len(spec.Pieces) != len(pieces) {
		t.Fatalf("len(pieces) = %d, want %d", len(spec.Pieces), len(pieces))
	}
	for i, p := range pieces {
		if spec.Pieces[i].Piece != p.Piece || spec.Pieces[i].Type != p.Type || spec.Pieces[i].Score != p.Score {
			t.Errorf("piece %d = %+v, want %+v", i, spec.Pieces[i], p)
		}
	}
}

func TestTokenizerSplitSpecial(t *testing.T) {
	pieces := []SentencePiece{
		{Piece: "<unk>", Score: 0.0, Type: PieceTypeUnknown},
		{Piece: "▁", Score: -1.0, Type: PieceTypeNormal},
		{Piece: "<bos>", Score: 0.0, Type: PieceTypeControl},
		{Piece: "<eos>", Score: 0.0, Type: PieceTypeControl},
	}
	pb := makeModelBytes(pieces, ModelTypeBPE, true, false, true)
	tok, err := New(pb)
	if err != nil {
		t.Fatal(err)
	}

	chunks := tok.splitSpecial("<bos>hello<eos>world<bos>")
	want := []chunk{
		{text: "<bos>", id: 2},
		{text: "hello", id: -1},
		{text: "<eos>", id: 3},
		{text: "world", id: -1},
		{text: "<bos>", id: 2},
	}
	if !reflect.DeepEqual(chunks, want) {
		t.Errorf("splitSpecial = %+v, want %+v", chunks, want)
	}
}

func TestTokenizerBPE(t *testing.T) {
	// Simple vocab:
	// 0: <unk> (Type=2)
	// 1: ▁ (score -1)
	// 2: h (score -2)
	// 3: e (score -2)
	// 4: l (score -2)
	// 5: o (score -2)
	// 6: he (score -1.5)
	// 7: ll (score -1.4)
	// 8: lo (score -1.6)
	// 9: ▁he (score -1.0)
	pieces := []SentencePiece{
		{Piece: "<unk>", Score: 0, Type: PieceTypeUnknown},
		{Piece: "▁", Score: -1.0},
		{Piece: "h", Score: -2.0},
		{Piece: "e", Score: -2.0},
		{Piece: "l", Score: -2.0},
		{Piece: "o", Score: -2.0},
		{Piece: "he", Score: -1.5},
		{Piece: "ll", Score: -1.4},
		{Piece: "lo", Score: -1.6},
		{Piece: "▁he", Score: -1.0},
	}

	pb := makeModelBytes(pieces, ModelTypeBPE, true, false, true)
	tok, err := New(pb)
	if err != nil {
		t.Fatal(err)
	}

	// Input: "hello"
	// Normalized: "▁hello"
	// Initial nodes: [▁, h, e, l, l, o]
	// Possible merges:
	// - ▁ + h = ▁h (no)
	// - h + e = he (score -1.5)
	// - e + l = el (no)
	// - l + l = ll (score -1.4)
	// - l + o = lo (score -1.6)
	// Best score is ll (-1.4) -> merge to: [▁, h, e, ll, o]
	// Next merges:
	// - h + e = he (-1.5)
	// Best score is he (-1.5) -> merge to: [▁, he, ll, o]
	// Next merges:
	// - ▁ + he = ▁he (-1.0)
	// - ll + o = llo (no)
	// Best score is ▁he (-1.0) -> merge to: [▁he, ll, o]
	// Next merges:
	// - ▁he + ll = no
	// - ll + o = llo (no)
	// Final nodes: [▁he, ll, o]
	// Expected IDs: [9, 7, 5]
	ids := tok.Encode("hello")
	want := []int32{9, 7, 5}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("Encode('hello') = %v, want %v", ids, want)
	}

	// Decode roundtrip
	decoded := tok.Decode([]int{9, 7, 5})
	if decoded != "hello" {
		t.Errorf("Decode = %q, want 'hello'", decoded)
	}
}

func TestTokenizerByteFallback(t *testing.T) {
	// Vocab:
	// 0: <unk>
	// 1: <0x41> (A)
	// 2: <0x42> (B)
	// 3: AB
	pieces := []SentencePiece{
		{Piece: "<unk>", Score: 0, Type: PieceTypeUnknown},
		{Piece: "<0x41>", Score: -3.0, Type: PieceTypeByte},
		{Piece: "<0x42>", Score: -3.0, Type: PieceTypeByte},
		{Piece: "AB", Score: -1.0},
	}
	pb := makeModelBytes(pieces, ModelTypeBPE, false, true, false)
	tok, err := New(pb)
	if err != nil {
		t.Fatal(err)
	}

	// Input "AB"
	// Since byte fallback is true and 'A' and 'B' are not directly in the vocab as character runes (only as byte pieces <0x41> and <0x42>), they will fall back to their bytes.
	// Initial nodes: bytes 0x41 ("\x41", id=1) and 0x42 ("\x42", id=2).
	// Merge: "\x41" + "\x42" = "\x41\x42" which is "AB", in vocab (id=3).
	// So they should merge to "AB" (id=3)!
	ids := tok.Encode("AB")
	want := []int32{3}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("Encode('AB') = %v, want %v", ids, want)
	}

	decoded := tok.Decode([]int{3})
	if decoded != "AB" {
		t.Errorf("Decode = %q, want 'AB'", decoded)
	}
}
