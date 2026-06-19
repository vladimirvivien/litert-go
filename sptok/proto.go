package sptok

import (
	"encoding/binary"
	"fmt"
	"math"
)

type ModelType int

const (
	ModelTypeUnigram ModelType = 1
	ModelTypeBPE     ModelType = 2
	ModelTypeChar    ModelType = 3
	ModelTypeWord    ModelType = 4
)

type PieceType int

const (
	PieceTypeNormal      PieceType = 1
	PieceTypeUnknown     PieceType = 2
	PieceTypeControl     PieceType = 3
	PieceTypeUserDefined PieceType = 4
	PieceTypeByte        PieceType = 5
	PieceTypeUnused      PieceType = 6
)

type SentencePiece struct {
	Piece string
	Score float32
	Type  PieceType
}

type ModelSpec struct {
	Pieces                 []SentencePiece
	ModelType              ModelType
	UnkID                  int
	BosID                  int
	EosID                  int
	PadID                  int
	ByteFallback           bool
	HasPrecompiledCharsmap bool
	AddDummyPrefix         bool
	RemoveExtraWhitespaces bool
	EscapeWhitespaces      bool
}

func parseModelSpec(sp []byte) (*ModelSpec, error) {
	spec := &ModelSpec{
		UnkID:                  0,
		BosID:                  1,
		EosID:                  2,
		PadID:                  -1,
		AddDummyPrefix:         true,
		RemoveExtraWhitespaces: true,
		EscapeWhitespaces:      true,
	}

	i := 0
	for i < len(sp) {
		field, wt, val, next, ok := parseField(sp, i)
		if !ok {
			return nil, fmt.Errorf("sptok: corrupt model protobuf at index %d", i)
		}
		switch field {
		case 1: // pieces
			if wt == 2 {
				p, err := parsePiece(val)
				if err != nil {
					return nil, err
				}
				spec.Pieces = append(spec.Pieces, p)
			}
		case 2: // trainer_spec
			if wt == 2 {
				if err := parseTrainerSpec(val, spec); err != nil {
					return nil, err
				}
			}
		case 3: // normalizer_spec
			if wt == 2 {
				if err := parseNormalizerSpec(val, spec); err != nil {
					return nil, err
				}
			}
		}
		i = next
	}
	return spec, nil
}

func parseField(b []byte, i int) (field uint64, wt uint64, val []byte, end int, ok bool) {
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
		return field, wt, buf[:m], i + m, true
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

func decodeFloat32(b []byte) float32 {
	if len(b) < 4 {
		return 0
	}
	bits := binary.LittleEndian.Uint32(b[:4])
	return math.Float32frombits(bits)
}

func parsePiece(b []byte) (SentencePiece, error) {
	var p SentencePiece
	p.Type = PieceTypeNormal // default
	i := 0
	for i < len(b) {
		field, wt, val, next, ok := parseField(b, i)
		if !ok {
			return p, fmt.Errorf("sptok: corrupt piece protobuf")
		}
		switch field {
		case 1: // piece string
			if wt == 2 {
				p.Piece = string(val)
			}
		case 2: // score
			if wt == 5 {
				p.Score = decodeFloat32(val)
			}
		case 3: // type
			if wt == 0 {
				v, _ := binary.Uvarint(val)
				p.Type = PieceType(v)
			}
		}
		i = next
	}
	return p, nil
}

func parseTrainerSpec(b []byte, spec *ModelSpec) error {
	i := 0
	for i < len(b) {
		field, wt, val, next, ok := parseField(b, i)
		if !ok {
			return fmt.Errorf("sptok: corrupt trainer_spec protobuf")
		}
		if wt == 0 {
			v, _ := binary.Uvarint(val)
			switch field {
			case 3: // model_type
				spec.ModelType = ModelType(v)
			case 15: // unk_id
				spec.UnkID = int(int32(v))
			case 16: // bos_id
				spec.BosID = int(int32(v))
			case 17: // eos_id
				spec.EosID = int(int32(v))
			case 18: // pad_id
				spec.PadID = int(int32(v))
			case 35: // byte_fallback
				spec.ByteFallback = v != 0
			}
		}
		i = next
	}
	return nil
}

func parseNormalizerSpec(b []byte, spec *ModelSpec) error {
	i := 0
	for i < len(b) {
		field, wt, val, next, ok := parseField(b, i)
		if !ok {
			return fmt.Errorf("sptok: corrupt normalizer_spec protobuf")
		}
		switch field {
		case 2: // precompiled_charsmap
			if wt == 2 && len(val) > 0 {
				spec.HasPrecompiledCharsmap = true
			}
		case 3: // add_dummy_prefix
			if wt == 0 {
				v, _ := binary.Uvarint(val)
				spec.AddDummyPrefix = v != 0
			}
		case 4: // remove_extra_whitespaces
			if wt == 0 {
				v, _ := binary.Uvarint(val)
				spec.RemoveExtraWhitespaces = v != 0
			}
		case 5: // escape_whitespaces
			if wt == 0 {
				v, _ := binary.Uvarint(val)
				spec.EscapeWhitespaces = v != 0
			}
		}
		i = next
	}
	return nil
}
