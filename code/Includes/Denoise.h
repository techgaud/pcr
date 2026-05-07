#pragma once

#include <vector>

// In-place cross-bilateral filter on an 8-bit RGB framebuffer.
//
// Not a Real Denoiser™ — no AI models, no cross-bilateral with normal/albedo
// guides. Just a 5x5 spatial-and-color-weighted blur that knocks down
// path-tracing speckle while preserving the edges that have a strong color
// gradient (walls, sphere outline). Quality is far below Intel OIDN, but
// it's ~30 lines of code, no dependencies, and visibly better than nothing.
//
// Inputs:
//   rgb     - tightly packed RGB8 buffer of length width*height*3
//   width   - image width in pixels
//   height  - image height in pixels
namespace Denoise
{
    void bilateralRGB(std::vector<unsigned char> &rgb, int width, int height);
}
