#include "Includes/Denoise.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace Denoise
{
    // 5x5 cross-bilateral. Tuned for path-tracing speckle: the spatial sigma
    // is 1.5 pixels (so neighborhood matters but we don't blur edges much) and
    // the color sigma is ~10 in 8-bit space (small color differences merge,
    // larger ones get preserved as edges). Two passes for slightly better
    // result without making the kernel huge.
    void bilateralRGB(std::vector<unsigned char> &rgb, int width, int height)
    {
        if (width <= 0 || height <= 0 || rgb.size() < (size_t)width * height * 3)
            return;

        const int radius = 2; // 5x5 window
        const float sigmaSpatial = 1.5f;
        const float sigmaColor = 10.0f;
        const float invSpatial2 = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
        const float invColor2   = 1.0f / (2.0f * sigmaColor * sigmaColor);

        std::vector<unsigned char> out(rgb.size());

        for (int pass = 0; pass < 2; pass++)
        {
            const std::vector<unsigned char> &src = (pass == 0) ? rgb : out;
            std::vector<unsigned char> &dst = (pass == 0) ? out : rgb;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    const size_t cIdx = ((size_t)y * width + x) * 3;
                    float cR = (float)src[cIdx + 0];
                    float cG = (float)src[cIdx + 1];
                    float cB = (float)src[cIdx + 2];

                    float sumW = 0.0f;
                    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

                    for (int dy = -radius; dy <= radius; dy++)
                    {
                        int ny = y + dy;
                        if (ny < 0 || ny >= height) continue;
                        for (int dx = -radius; dx <= radius; dx++)
                        {
                            int nx = x + dx;
                            if (nx < 0 || nx >= width) continue;

                            const size_t nIdx = ((size_t)ny * width + nx) * 3;
                            float nR = (float)src[nIdx + 0];
                            float nG = (float)src[nIdx + 1];
                            float nB = (float)src[nIdx + 2];

                            float spatial = (float)(dx * dx + dy * dy) * invSpatial2;
                            float dR = nR - cR, dG = nG - cG, dB = nB - cB;
                            float color = (dR * dR + dG * dG + dB * dB) * invColor2;
                            float w = std::exp(-(spatial + color));

                            sumW += w;
                            sumR += w * nR;
                            sumG += w * nG;
                            sumB += w * nB;
                        }
                    }

                    if (sumW > 0.0f)
                    {
                        dst[cIdx + 0] = (unsigned char)std::min(255.0f, sumR / sumW + 0.5f);
                        dst[cIdx + 1] = (unsigned char)std::min(255.0f, sumG / sumW + 0.5f);
                        dst[cIdx + 2] = (unsigned char)std::min(255.0f, sumB / sumW + 0.5f);
                    }
                    else
                    {
                        dst[cIdx + 0] = src[cIdx + 0];
                        dst[cIdx + 1] = src[cIdx + 1];
                        dst[cIdx + 2] = src[cIdx + 2];
                    }
                }
            }
        }
    }
}
