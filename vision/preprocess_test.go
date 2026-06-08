package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// patchify must order patches row-major, set positions to (col,row), and lay out
// each patch's pixels [ph][pw][c] with channels innermost, normalized to [0,1].
func TestPatchifyLayout(t *testing.T) {
	// 32x32 = 2x2 patches. Encode position into pixels: R=x, G=y, B=7.
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 7, A: 255})
		}
	}
	p := patchify(img)
	if p.Count != 4 {
		t.Fatalf("Count = %d, want 4", p.Count)
	}
	if p.Tokens != 1 { // ceil(4/9)
		t.Fatalf("Tokens = %d, want 1", p.Tokens)
	}
	wantPos := [][2]int32{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	for i, w := range wantPos {
		if got := [2]int32{p.Positions[i*2], p.Positions[i*2+1]}; got != w {
			t.Errorf("Positions[%d] = %v, want %v", i, got, w)
		}
	}
	// Check every pixel of every patch against R=x, G=y, B=7.
	for gy := 0; gy < 2; gy++ {
		for gx := 0; gx < 2; gx++ {
			idx := gy*2 + gx
			for py := 0; py < PatchSize; py++ {
				for px := 0; px < PatchSize; px++ {
					base := idx*PatchDim + (py*PatchSize+px)*Channels
					wantR := float32(gx*PatchSize+px) / 255
					wantG := float32(gy*PatchSize+py) / 255
					if p.Images[base] != wantR || p.Images[base+1] != wantG || p.Images[base+2] != 7.0/255 {
						t.Fatalf("patch %d px(%d,%d) = [%v %v %v], want [%v %v %v]",
							idx, px, py, p.Images[base], p.Images[base+1], p.Images[base+2], wantR, wantG, 7.0/255)
					}
				}
			}
		}
	}
}

// targetSize must snap each aspect-preserved side down to a multiple of 48 and
// stay within the patch budget.
func TestTargetSize(t *testing.T) {
	const maxPatches = 2520
	cases := []struct {
		w, h, th, tw int
	}{
		{1000, 1000, 768, 768},  // square: factor .803, ideal 803 -> 768 (16*48)
		{2000, 1000, 528, 1104}, // factor .568: ideal h 568->528 (11*48), w 1136->1104 (23*48)
	}
	for _, c := range cases {
		th, tw, err := targetSize(c.w, c.h, maxPatches)
		if err != nil {
			t.Fatalf("targetSize(%d,%d): %v", c.w, c.h, err)
		}
		if th%sideMult != 0 || tw%sideMult != 0 {
			t.Errorf("targetSize(%d,%d) = %dx%d not multiples of %d", c.w, c.h, tw, th, sideMult)
		}
		patches := (th / PatchSize) * (tw / PatchSize)
		if patches > maxPatches {
			t.Errorf("targetSize(%d,%d) -> %d patches > %d", c.w, c.h, patches, maxPatches)
		}
		if th != c.th || tw != c.tw {
			t.Errorf("targetSize(%d,%d) = %dx%d (hxw), want %dx%d", c.w, c.h, th, tw, c.th, c.tw)
		}
	}
}

// Preprocess end-to-end: decode a PNG, resize, patchify; dims and counts must be
// consistent (sides multiples of 48, tokens = ceil(patches/9)).
func TestPreprocessEndToEnd(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	for y := 0; y < 480; y++ {
		for x := 0; x < 640; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 128, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	p, err := Preprocess(buf.Bytes(), 280)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Images) != p.Count*PatchDim {
		t.Errorf("Images len %d != Count*PatchDim %d", len(p.Images), p.Count*PatchDim)
	}
	if len(p.Positions) != p.Count*2 {
		t.Errorf("Positions len %d != Count*2 %d", len(p.Positions), p.Count*2)
	}
	if p.Count > 2520 {
		t.Errorf("Count %d exceeds budget 2520", p.Count)
	}
	if want := (p.Count + 8) / 9; p.Tokens != want {
		t.Errorf("Tokens %d, want ceil(%d/9)=%d", p.Tokens, p.Count, want)
	}
}
