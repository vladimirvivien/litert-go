// Package vision provides standard-library image and raw video-frame
// preprocessing and postprocessing helpers for .tflite computer vision models.
package vision

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"slices"
)

// ResizeMode controls how source frames map onto target model input dimensions.
type ResizeMode int

const (
	// ResizeStretch stretches source images directly to Config.Width x Config.Height.
	ResizeStretch ResizeMode = iota
	// ResizeLetterbox preserves aspect ratio and pads remaining borders with PadColor.
	ResizeLetterbox
	// ResizeCenterCrop center-crops the source to match target aspect ratio before scaling.
	ResizeCenterCrop
)

// Interpolation selects the pixel resampling filter used by built-in resizing.
type Interpolation int

const (
	InterpBilinear Interpolation = iota
	InterpNearest
)

// ChannelOrder selects output RGB channel arrangement.
type ChannelOrder int

const (
	OrderRGB ChannelOrder = iota
	OrderBGR
)

// Scaler allows callers to plug in a custom high-quality image resizer without
// introducing extra dependencies on litert-go.
type Scaler interface {
	Scale(dst *image.RGBA, src image.Image)
}

// Config configures image and video-frame preprocessing.
type Config struct {
	Width         int
	Height        int
	ResizeMode    ResizeMode
	Interpolation Interpolation
	ChannelOrder  ChannelOrder
	Mean          [3]float32
	Std           [3]float32
	PadColor      color.RGBA
	Scaler        Scaler
}

// LetterboxInfo records source/destination geometry for mapping detection boxes
// back into normalized source-frame coordinates.
type LetterboxInfo struct {
	SrcWidth  int
	SrcHeight int
	DstWidth  int
	DstHeight int
	Scale     float32
	PadX      int
	PadY      int
}

// Decode decodes a PNG or JPEG image stream into an image.Image.
func Decode(r io.Reader) (image.Image, error) {
	if r == nil {
		return nil, fmt.Errorf("vision: nil reader")
	}
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("vision: decode image: %w", err)
	}
	return img, nil
}

// DecodeBytes decodes PNG or JPEG image bytes into an image.Image.
func DecodeBytes(data []byte) (image.Image, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("vision: empty image bytes")
	}
	return Decode(bytes.NewReader(data))
}

// FromRGB copies raw packed RGB24 video frame bytes into an *image.RGBA.
// If stride is zero, it defaults to width * 3.
func FromRGB(data []byte, width, height, stride int) (*image.RGBA, error) {
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("vision: invalid dimensions %dx%d", width, height)
	}
	minStride := width * 3
	if stride == 0 {
		stride = minStride
	}
	if stride < minStride {
		return nil, fmt.Errorf("vision: stride %d smaller than row bytes %d", stride, minStride)
	}
	need := (height-1)*stride + minStride
	if len(data) < need {
		return nil, fmt.Errorf("vision: RGB buffer length %d < required %d", len(data), need)
	}

	out := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		srcRow := data[y*stride:]
		dstRow := out.Pix[y*out.Stride:]
		for x := range width {
			si := x * 3
			di := x * 4
			dstRow[di] = srcRow[si]
			dstRow[di+1] = srcRow[si+1]
			dstRow[di+2] = srcRow[si+2]
			dstRow[di+3] = 255
		}
	}
	return out, nil
}

// FromRGBA copies raw packed RGBA32 video frame bytes into an *image.RGBA.
// If stride is zero, it defaults to width * 4.
func FromRGBA(data []byte, width, height, stride int) (*image.RGBA, error) {
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("vision: invalid dimensions %dx%d", width, height)
	}
	minStride := width * 4
	if stride == 0 {
		stride = minStride
	}
	if stride < minStride {
		return nil, fmt.Errorf("vision: stride %d smaller than row bytes %d", stride, minStride)
	}
	need := (height-1)*stride + minStride
	if len(data) < need {
		return nil, fmt.Errorf("vision: RGBA buffer length %d < required %d", len(data), need)
	}

	out := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		copy(out.Pix[y*out.Stride:y*out.Stride+minStride], data[y*stride:y*stride+minStride])
	}
	return out, nil
}

// FromYCbCr420 wraps copied YUV420 planar camera frame planes in an *image.YCbCr.
// If yStride or cStride is zero, packed widths are assumed.
func FromYCbCr420(y, cb, cr []byte, width, height, yStride, cStride int) (*image.YCbCr, error) {
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("vision: invalid dimensions %dx%d", width, height)
	}
	if yStride == 0 {
		yStride = width
	}
	cWidth := (width + 1) / 2
	cHeight := (height + 1) / 2
	if cStride == 0 {
		cStride = cWidth
	}
	if yStride < width || cStride < cWidth {
		return nil, fmt.Errorf("vision: invalid plane strides y=%d c=%d for %dx%d", yStride, cStride, width, height)
	}
	needY := (height-1)*yStride + width
	needC := (cHeight-1)*cStride + cWidth
	if len(y) < needY || len(cb) < needC || len(cr) < needC {
		return nil, fmt.Errorf("vision: YCbCr420 plane sizes (%d,%d,%d) smaller than required (%d,%d,%d)",
			len(y), len(cb), len(cr), needY, needC, needC)
	}

	return &image.YCbCr{
		Y:              slices.Clone(y[:needY]),
		Cb:             slices.Clone(cb[:needC]),
		Cr:             slices.Clone(cr[:needC]),
		YStride:        yStride,
		CStride:        cStride,
		SubsampleRatio: image.YCbCrSubsampleRatio420,
		Rect:           image.Rect(0, 0, width, height),
	}, nil
}

// PreprocessFloat32 resizes img according to cfg and normalizes pixels into
// NHWC float32 order using (pixel - Mean[c]) / Std[c]. When Std is zero-valued,
// {255, 255, 255} is applied to scale bytes into [0, 1].
func PreprocessFloat32(img image.Image, cfg Config) ([]float32, LetterboxInfo, error) {
	std := cfg.Std
	if std == [3]float32{} {
		std = [3]float32{255, 255, 255}
	}
	if std[0] == 0 || std[1] == 0 || std[2] == 0 {
		return nil, LetterboxInfo{}, fmt.Errorf("vision: Std channels must be non-zero, got %v", cfg.Std)
	}

	rgba, info, err := prepareCanvas(img, cfg)
	if err != nil {
		return nil, LetterboxInfo{}, err
	}

	inv0, inv1, inv2 := 1.0/std[0], 1.0/std[1], 1.0/std[2]
	out := make([]float32, cfg.Width*cfg.Height*3)
	writePixels(rgba, cfg.Width, cfg.Height, cfg.ChannelOrder, func(idx int, c0, c1, c2 uint8) {
		out[idx] = (float32(c0) - cfg.Mean[0]) * inv0
		out[idx+1] = (float32(c1) - cfg.Mean[1]) * inv1
		out[idx+2] = (float32(c2) - cfg.Mean[2]) * inv2
	})
	return out, info, nil
}

// PreprocessUint8 resizes img according to cfg into NHWC uint8 [0, 255] bytes.
func PreprocessUint8(img image.Image, cfg Config) ([]uint8, LetterboxInfo, error) {
	rgba, info, err := prepareCanvas(img, cfg)
	if err != nil {
		return nil, LetterboxInfo{}, err
	}

	out := make([]uint8, cfg.Width*cfg.Height*3)
	writePixels(rgba, cfg.Width, cfg.Height, cfg.ChannelOrder, func(idx int, c0, c1, c2 uint8) {
		out[idx] = c0
		out[idx+1] = c1
		out[idx+2] = c2
	})
	return out, info, nil
}

// PreprocessInt8 resizes img according to cfg into NHWC signed int8 [-128, 127]
// bytes by centering each uint8 channel around 128.
func PreprocessInt8(img image.Image, cfg Config) ([]int8, LetterboxInfo, error) {
	rgba, info, err := prepareCanvas(img, cfg)
	if err != nil {
		return nil, LetterboxInfo{}, err
	}

	out := make([]int8, cfg.Width*cfg.Height*3)
	writePixels(rgba, cfg.Width, cfg.Height, cfg.ChannelOrder, func(idx int, c0, c1, c2 uint8) {
		out[idx] = int8(int(c0) - 128)
		out[idx+1] = int8(int(c1) - 128)
		out[idx+2] = int8(int(c2) - 128)
	})
	return out, info, nil
}

func writePixels(rgba *image.RGBA, width, height int, order ChannelOrder, emit func(idx int, c0, c1, c2 uint8)) {
	for y := range height {
		row := rgba.Pix[y*rgba.Stride:]
		rowIdx := y * width * 3
		for x := range width {
			p := row[x*4:]
			r, g, b := p[0], p[1], p[2]
			if order == OrderBGR {
				r, b = b, r
			}
			emit(rowIdx+x*3, r, g, b)
		}
	}
}

func prepareCanvas(img image.Image, cfg Config) (*image.RGBA, LetterboxInfo, error) {
	if img == nil {
		return nil, LetterboxInfo{}, fmt.Errorf("vision: nil image")
	}
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, LetterboxInfo{}, fmt.Errorf("vision: invalid target dimensions %dx%d", cfg.Width, cfg.Height)
	}
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, LetterboxInfo{}, fmt.Errorf("vision: empty source image bounds %v", bounds)
	}

	info := LetterboxInfo{
		SrcWidth:  srcW,
		SrcHeight: srcH,
		DstWidth:  cfg.Width,
		DstHeight: cfg.Height,
		Scale:     1,
	}

	dst := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	srcRGBA := ensureRGBA(img, bounds)

	switch cfg.ResizeMode {
	case ResizeStretch:
		scaleInto(dst, srcRGBA, cfg.Interpolation, cfg.Scaler)
		return dst, info, nil

	case ResizeLetterbox:
		fillRGBA(dst, cfg.PadColor)
		scale := min(float64(cfg.Width)/float64(srcW), float64(cfg.Height)/float64(srcH))
		scaledW := max(1, min(cfg.Width, int(math.Round(float64(srcW)*scale))))
		scaledH := max(1, min(cfg.Height, int(math.Round(float64(srcH)*scale))))
		padX := (cfg.Width - scaledW) / 2
		padY := (cfg.Height - scaledH) / 2

		sub := dst.SubImage(image.Rect(padX, padY, padX+scaledW, padY+scaledH)).(*image.RGBA)
		normalized := &image.RGBA{
			Pix:    sub.Pix,
			Stride: sub.Stride,
			Rect:   image.Rect(0, 0, scaledW, scaledH),
		}
		scaleInto(normalized, srcRGBA, cfg.Interpolation, cfg.Scaler)

		info.Scale = float32(scale)
		info.PadX = padX
		info.PadY = padY
		return dst, info, nil

	case ResizeCenterCrop:
		targetAspect := float64(cfg.Width) / float64(cfg.Height)
		srcAspect := float64(srcW) / float64(srcH)
		cropX, cropY, cropW, cropH := 0, 0, srcW, srcH
		if srcAspect > targetAspect {
			cropW = max(1, int(math.Round(float64(srcH)*targetAspect)))
			cropX = (srcW - cropW) / 2
		} else if srcAspect < targetAspect {
			cropH = max(1, int(math.Round(float64(srcW)/targetAspect)))
			cropY = (srcH - cropH) / 2
		}
		cropped := &image.RGBA{
			Pix:    srcRGBA.Pix[cropY*srcRGBA.Stride+cropX*4:],
			Stride: srcRGBA.Stride,
			Rect:   image.Rect(0, 0, cropW, cropH),
		}
		scaleInto(dst, cropped, cfg.Interpolation, cfg.Scaler)
		info.Scale = float32(cfg.Width) / float32(cropW)
		return dst, info, nil

	default:
		return nil, LetterboxInfo{}, fmt.Errorf("vision: unknown ResizeMode %d", cfg.ResizeMode)
	}
}

func ensureRGBA(img image.Image, bounds image.Rectangle) *image.RGBA {
	if r, ok := img.(*image.RGBA); ok && bounds.Min == (image.Point{}) {
		return r
	}
	rgba := image.NewRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	draw.Draw(rgba, rgba.Bounds(), img, bounds.Min, draw.Src)
	return rgba
}

func fillRGBA(dst *image.RGBA, c color.RGBA) {
	if c == (color.RGBA{}) {
		return
	}
	w, h := dst.Rect.Dx(), dst.Rect.Dy()
	for y := range h {
		row := dst.Pix[y*dst.Stride:]
		for x := range w {
			p := row[x*4:]
			p[0] = c.R
			p[1] = c.G
			p[2] = c.B
			p[3] = c.A
		}
	}
}

func scaleInto(dst, src *image.RGBA, interp Interpolation, scaler Scaler) {
	if scaler != nil {
		scaler.Scale(dst, src)
		return
	}
	dstW, dstH := dst.Rect.Dx(), dst.Rect.Dy()
	srcW, srcH := src.Rect.Dx(), src.Rect.Dy()

	if interp == InterpNearest {
		for y := range dstH {
			sy := (y * srcH) / dstH
			srcRow := src.Pix[sy*src.Stride:]
			dstRow := dst.Pix[y*dst.Stride:]
			for x := range dstW {
				sx := (x * srcW) / dstW
				copy(dstRow[x*4:x*4+4], srcRow[sx*4:sx*4+4])
			}
		}
		return
	}

	xRatio := float64(srcW) / float64(dstW)
	yRatio := float64(srcH) / float64(dstH)
	for y := range dstH {
		fy := (float64(y)+0.5)*yRatio - 0.5
		if fy < 0 {
			fy = 0
		}
		y0 := int(fy)
		wy := fy - float64(y0)
		y1 := min(y0+1, srcH-1)
		y0 = min(y0, srcH-1)

		row0 := src.Pix[y0*src.Stride:]
		row1 := src.Pix[y1*src.Stride:]
		dstRow := dst.Pix[y*dst.Stride:]
		iy0 := 1.0 - wy

		for x := range dstW {
			fx := (float64(x)+0.5)*xRatio - 0.5
			if fx < 0 {
				fx = 0
			}
			x0 := int(fx)
			wx := fx - float64(x0)
			x1 := min(x0+1, srcW-1)
			x0 = min(x0, srcW-1)
			ix0 := 1.0 - wx

			w00 := ix0 * iy0
			w10 := wx * iy0
			w01 := ix0 * wy
			w11 := wx * wy

			p00 := row0[x0*4:]
			p10 := row0[x1*4:]
			p01 := row1[x0*4:]
			p11 := row1[x1*4:]

			d := dstRow[x*4:]
			for c := range 4 {
				val := float64(p00[c])*w00 + float64(p10[c])*w10 + float64(p01[c])*w01 + float64(p11[c])*w11
				d[c] = uint8(math.Round(val))
			}
		}
	}
}
