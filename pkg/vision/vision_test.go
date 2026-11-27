package vision_test

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"testing"

	"github.com/vladimirvivien/litert-go/pkg/tensor"
	"github.com/vladimirvivien/litert-go/pkg/vision"
)

func makeCheckerboard(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			if (x < w/2) != (y < h/2) {
				img.SetRGBA(x, y, color.RGBA{R: 255, G: 0, B: 0, A: 255})
			} else {
				img.SetRGBA(x, y, color.RGBA{R: 0, G: 0, B: 255, A: 255})
			}
		}
	}
	return img
}

type mockScaler struct {
	called bool
}

func (m *mockScaler) Scale(dst *image.RGBA, src image.Image) {
	m.called = true
	b := dst.Rect
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.SetRGBA(x, y, color.RGBA{R: 42, G: 84, B: 126, A: 255})
		}
	}
}

func TestDecodePNGAndJPEG(t *testing.T) {
	src := makeCheckerboard(16, 12)

	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, src); err != nil {
		t.Fatal(err)
	}
	imgPNG, err := vision.DecodeBytes(pngBuf.Bytes())
	if err != nil {
		t.Fatalf("DecodeBytes PNG: %v", err)
	}
	if imgPNG.Bounds().Dx() != 16 || imgPNG.Bounds().Dy() != 12 {
		t.Fatalf("unexpected PNG bounds: %v", imgPNG.Bounds())
	}

	var jpgBuf bytes.Buffer
	if encodeErr := jpeg.Encode(&jpgBuf, src, nil); encodeErr != nil {
		t.Fatal(encodeErr)
	}
	imgJPG, err := vision.Decode(&jpgBuf)
	if err != nil {
		t.Fatalf("Decode JPEG: %v", err)
	}
	if imgJPG.Bounds().Dx() != 16 || imgJPG.Bounds().Dy() != 12 {
		t.Fatalf("unexpected JPEG bounds: %v", imgJPG.Bounds())
	}
}

func TestRawVideoAdapters(t *testing.T) {
	rgb := []byte{
		255, 10, 20, 30, 40, 50,
		60, 70, 80, 90, 100, 110,
	}
	imgRGB, err := vision.FromRGB(rgb, 2, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := imgRGB.RGBAAt(1, 0); got != (color.RGBA{R: 30, G: 40, B: 50, A: 255}) {
		t.Fatalf("FromRGB(1,0) = %+v", got)
	}

	rgba := make([]byte, 2*12) // stride 12 for width 2
	rgba[0], rgba[1], rgba[2], rgba[3] = 11, 22, 33, 255
	rgba[12], rgba[13], rgba[14], rgba[15] = 44, 55, 66, 255
	imgRGBA, err := vision.FromRGBA(rgba, 2, 2, 12)
	if err != nil {
		t.Fatal(err)
	}
	if got := imgRGBA.RGBAAt(0, 1); got != (color.RGBA{R: 44, G: 55, B: 66, A: 255}) {
		t.Fatalf("FromRGBA(0,1) = %+v", got)
	}

	y := []byte{16, 16, 16, 16, 200, 200, 200, 200}
	cb := []byte{128, 128}
	cr := []byte{128, 128}
	imgYUV, err := vision.FromYCbCr420(y, cb, cr, 4, 2, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if imgYUV.Bounds().Dx() != 4 || imgYUV.Bounds().Dy() != 2 {
		t.Fatalf("unexpected YUV bounds: %v", imgYUV.Bounds())
	}

	if _, badErr := vision.FromRGB(rgb[:2], 2, 2, 0); badErr == nil {
		t.Fatal("expected short buffer error")
	}
}

func TestPreprocessModesOrdersAndDtypes(t *testing.T) {
	// 4x2 horizontal red bar on top, blue bar on bottom? Make uniform orange for normalization check.
	uniform := image.NewRGBA(image.Rect(0, 0, 8, 4))
	for y := range 4 {
		for x := range 8 {
			uniform.SetRGBA(x, y, color.RGBA{R: 255, G: 128, B: 0, A: 255})
		}
	}

	// Default float32 [0, 1] RGB normalization.
	f32, _, err := vision.PreprocessFloat32(uniform, vision.Config{
		Width:  4,
		Height: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(f32) != 4*4*3 {
		t.Fatalf("len = %d, want 48", len(f32))
	}
	if math.Abs(float64(f32[0]-1.0)) > 1e-5 || math.Abs(float64(f32[1]-128.0/255.0)) > 1e-3 || f32[2] != 0 {
		t.Fatalf("unexpected default normalized pixel: %v", f32[:3])
	}

	// BGR uint8 + Nearest + CenterCrop.
	u8, _, err := vision.PreprocessUint8(uniform, vision.Config{
		Width:         2,
		Height:        2,
		ResizeMode:    vision.ResizeCenterCrop,
		Interpolation: vision.InterpNearest,
		ChannelOrder:  vision.OrderBGR,
	})
	if err != nil {
		t.Fatal(err)
	}
	if u8[0] != 0 || u8[1] != 128 || u8[2] != 255 {
		t.Fatalf("unexpected BGR pixel: %v", u8[:3])
	}

	// Signed int8 centering (-128..127).
	i8, _, err := vision.PreprocessInt8(uniform, vision.Config{Width: 2, Height: 2})
	if err != nil {
		t.Fatal(err)
	}
	if i8[0] != 127 || i8[1] != 0 || i8[2] != -128 {
		t.Fatalf("unexpected int8 pixel: %v", i8[:3])
	}

	// Custom Scaler hook.
	ms := &mockScaler{}
	u8Scaled, _, err := vision.PreprocessUint8(uniform, vision.Config{
		Width:  4,
		Height: 4,
		Scaler: ms,
	})
	if err != nil || !ms.called {
		t.Fatalf("custom Scaler failed: called=%v err=%v", ms.called, err)
	}
	if u8Scaled[0] != 42 || u8Scaled[1] != 84 || u8Scaled[2] != 126 {
		t.Fatalf("unexpected Scaler output: %v", u8Scaled[:3])
	}

	// Letterbox padding and UnletterboxBox round-trip.
	wide := image.NewRGBA(image.Rect(0, 0, 400, 200))
	for y := range 200 {
		for x := range 400 {
			wide.SetRGBA(x, y, color.RGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}
	lbU8, info, err := vision.PreprocessUint8(wide, vision.Config{
		Width:      200,
		Height:     200,
		ResizeMode: vision.ResizeLetterbox,
		PadColor:   color.RGBA{R: 114, G: 114, B: 114, A: 255},
	})
	if err != nil {
		t.Fatal(err)
	}
	if info.PadX != 0 || info.PadY != 50 || math.Abs(float64(info.Scale-0.5)) > 1e-6 {
		t.Fatalf("unexpected LetterboxInfo: %+v", info)
	}
	// Row 0 is top letterbox padding (114, 114, 114).
	if lbU8[0] != 114 || lbU8[1] != 114 || lbU8[2] != 114 {
		t.Fatalf("expected pad color at top border, got %v", lbU8[:3])
	}

	unboxed := vision.UnletterboxBox(tensor.Box{YMin: 0.25, XMin: 0, YMax: 0.75, XMax: 1}, info)
	if math.Abs(float64(unboxed.YMin)) > 1e-4 || math.Abs(float64(unboxed.YMax-1.0)) > 1e-4 {
		t.Fatalf("UnletterboxBox = %+v, want [0,0,1,1]", unboxed)
	}
}

func TestPostprocessHelpers(t *testing.T) {
	cls := vision.ClassifyTopK([]float32{0.1, 0.7, 0.2}, []string{"bg", "cat"}, 3)
	if len(cls) != 3 || cls[0].Label != "cat" || cls[1].Label != "class_2" {
		t.Fatalf("ClassifyTopK = %+v", cls)
	}

	dets, err := vision.DecodeDetections(
		[]float32{0.1, 0.1, 0.5, 0.5, 0.2, 0.2, 0.8, 0.8},
		[]float32{1, 3},
		[]float32{0.9, 0.3},
		2,
		0.5,
	)
	if err != nil || len(dets) != 1 || dets[0].ClassID != 1 {
		t.Fatalf("DecodeDetections = %+v, %v", dets, err)
	}

	mask, err := vision.ArgmaxMask([]float32{
		0.1, 0.9, 0.0,
		0.8, 0.2, 0.0,
		0.0, 0.1, 0.9,
		0.2, 0.5, 0.3,
	}, 2, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	wantMask := []uint8{1, 0, 2, 1}
	for i := range wantMask {
		if mask[i] != wantMask[i] {
			t.Fatalf("mask = %v, want %v", mask, wantMask)
		}
	}
}
