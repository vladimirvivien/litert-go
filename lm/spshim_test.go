package lm

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// buildSPProto assembles a minimal ModelProto: pieces (field 1) and a
// normalizer_spec (field 3) with escape_whitespaces (field 5).
func buildSPProto(pieces []string, escapeWS bool) []byte {
	var out []byte
	for _, p := range pieces {
		var msg []byte
		msg = append(msg, 0x0a)
		msg = binary.AppendUvarint(msg, uint64(len(p)))
		msg = append(msg, p...)
		msg = append(msg, 0x15, 0, 0, 0, 0) // field 2 (score, float32) = 0
		out = append(out, 0x0a)
		out = binary.AppendUvarint(out, uint64(len(msg)))
		out = append(out, msg...)
	}
	norm := []byte{0x28, 0} // field 5 varint
	if escapeWS {
		norm[1] = 1
	}
	out = append(out, 0x1a)
	out = binary.AppendUvarint(out, uint64(len(norm)))
	out = append(out, norm...)
	return out
}

func collectPieces(t *testing.T, proto []byte) []string {
	t.Helper()
	var got []string
	i := 0
	for i < len(proto) {
		field, wt, val, end, ok := spField(proto, i)
		if !ok {
			t.Fatalf("unparseable proto at %d", i)
		}
		if field == 1 && wt == 2 {
			f, w, v, _, ok := spField(val, 0)
			if !ok || f != 1 || w != 2 {
				t.Fatalf("piece submessage does not start with piece string")
			}
			got = append(got, string(v))
		}
		i = end
	}
	return got
}

func TestRewriteRawSpacePieces(t *testing.T) {
	tests := []struct {
		name     string
		pieces   []string
		escapeWS bool
		want     []string
	}{
		{
			name:     "raw-space vocab rewritten",
			pieces:   []string{"The", " quick", "<0x20>", " a b "},
			escapeWS: false,
			want:     []string{"The", "▁quick", "<0x20>", "▁a▁b▁"},
		},
		{
			name:     "escaping vocab untouched",
			pieces:   []string{"▁quick", " raw"},
			escapeWS: true,
			want:     []string{"▁quick", " raw"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := buildSPProto(tc.pieces, tc.escapeWS)
			out := rewriteRawSpacePieces(in)
			if tc.escapeWS && !bytes.Equal(in, out) {
				t.Fatal("escape_whitespaces=true proto must be returned unchanged")
			}
			got := collectPieces(t, out)
			if len(got) != len(tc.want) {
				t.Fatalf("got %d pieces, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("piece[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestRewriteRawSpacePiecesGarbage(t *testing.T) {
	junk := []byte{0xff, 0xff, 0xff}
	if out := rewriteRawSpacePieces(junk); !bytes.Equal(out, junk) {
		t.Fatal("unparseable input must be returned unchanged")
	}
}
