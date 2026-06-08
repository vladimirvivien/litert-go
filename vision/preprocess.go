// Package vision preprocesses images for the gemma-4 vision encoder: decode,
// aspect-preserving resize, normalize, and patchify into the encoder's
// (images, positions_xy) inputs.
//
// The recipe mirrors LiteRT-LM's stb_image_preprocessor / image_preprocessor_utils
// and gemma4_data_processor_config: 16x16 patches over RGB, 3x3 pooling (so the
// encoder emits one visual token per 9 patches), and a resize that snaps each
// side down to a multiple of 48 (= 3*16) while preserving aspect ratio and
// staying within the patch budget.
package vision

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"golang.org/x/image/draw"
)

const (
	// PatchSize is the side of a square image patch in pixels.
	PatchSize = 16
	// PoolingKernel is the encoder's spatial pooling factor; one visual token
	// covers PoolingKernel*PoolingKernel patches.
	PoolingKernel = 3
	// Channels is the RGB channel count.
	Channels = 3
	// PatchDim is the flattened size of one patch (16*16*3).
	PatchDim = PatchSize * PatchSize * Channels
	// DefaultBudget is the default visual-token budget; max patches = budget*9.
	DefaultBudget = 280
)

// sideMult is the side-length granularity: a resized side is always a multiple
// of PoolingKernel*PatchSize so it divides evenly into pooled patches.
const sideMult = PoolingKernel * PatchSize // 48

// Patches is a patchified image ready for the vision encoder.
type Patches struct {
	Images    []float32 // [Count, PatchDim], row-major (patch, then [ph][pw][c])
	Positions []int32   // [Count, 2], each (x=col, y=row) patch-grid index
	Count     int       // number of patches P
	Tokens    int       // visual tokens = ceil(P / 9)
}

// Preprocess decodes data, resizes it preserving aspect ratio within
// budget*9 patches, normalizes to [0,1], and patchifies it. budget is the
// visual-token budget (e.g. 70/140/280); pass 0 for DefaultBudget.
func Preprocess(data []byte, budget int) (*Patches, error) {
	if budget <= 0 {
		budget = DefaultBudget
	}
	maxPatches := budget * PoolingKernel * PoolingKernel

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("vision: decode: %w", err)
	}
	b := img.Bounds()
	th, tw, err := targetSize(b.Dx(), b.Dy(), maxPatches)
	if err != nil {
		return nil, err
	}

	// Resize (and normalize the format) onto an RGBA canvas of the target size.
	dst := image.NewRGBA(image.Rect(0, 0, tw, th))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, b, draw.Src, nil)

	return patchify(dst), nil
}

// targetSize returns the resized (height, width) for an w x h image: snap each
// aspect-preserved side down to a multiple of sideMult, within maxPatches.
// Mirrors GetAspectRatioPreservingSize.
func targetSize(w, h, maxPatches int) (th, tw int, err error) {
	targetPx := float64(maxPatches) * float64(PatchSize*PatchSize)
	totalPx := float64(w) * float64(h)
	factor := math.Sqrt(targetPx / totalPx)
	idealH := factor * float64(h)
	idealW := factor * float64(w)

	th = int(math.Floor(idealH/float64(sideMult))) * sideMult
	tw = int(math.Floor(idealW/float64(sideMult))) * sideMult

	if th == 0 && tw == 0 {
		return 0, 0, fmt.Errorf("vision: image %dx%d resizes to 0x0", w, h)
	}
	maxSide := (maxPatches / (PoolingKernel * PoolingKernel)) * sideMult
	if th == 0 {
		th = sideMult
		tw = min(int(math.Floor(float64(w)/float64(h)))*sideMult, maxSide)
	} else if tw == 0 {
		tw = sideMult
		th = min(int(math.Floor(float64(h)/float64(w)))*sideMult, maxSide)
	}
	if float64(th)*float64(tw) > targetPx {
		return 0, 0, fmt.Errorf("vision: resized %dx%d exceeds patch budget", tw, th)
	}
	return th, tw, nil
}

// patchify splits an RGBA image (whose sides are multiples of PatchSize) into
// 16x16x3 patches, normalizing each channel to [0,1]. Patch order is row-major
// (row h outer, column w inner); positions are (col, row); patch pixels are
// laid out [ph][pw][c] with channels innermost.
func patchify(img *image.RGBA) *Patches {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	pw := w / PatchSize
	ph := h / PatchSize
	p := pw * ph

	out := &Patches{
		Images:    make([]float32, p*PatchDim),
		Positions: make([]int32, p*2),
		Count:     p,
		Tokens:    (p + PoolingKernel*PoolingKernel - 1) / (PoolingKernel * PoolingKernel),
	}
	for gy := 0; gy < ph; gy++ {
		for gx := 0; gx < pw; gx++ {
			idx := gy*pw + gx
			out.Positions[idx*2] = int32(gx)
			out.Positions[idx*2+1] = int32(gy)
			for py := 0; py < PatchSize; py++ {
				for px := 0; px < PatchSize; px++ {
					off := img.PixOffset(gx*PatchSize+px, gy*PatchSize+py)
					dst := idx*PatchDim + (py*PatchSize+px)*Channels
					out.Images[dst] = float32(img.Pix[off]) / 255     // R
					out.Images[dst+1] = float32(img.Pix[off+1]) / 255 // G
					out.Images[dst+2] = float32(img.Pix[off+2]) / 255 // B
				}
			}
		}
	}
	return out
}
