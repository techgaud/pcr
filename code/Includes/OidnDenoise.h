#pragma once

#include <vector>

#include "Vec3f.h"

// Thin wrapper around Intel Open Image Denoise. Compiles to no-ops when
// the binary is built with -DPCR_USE_OIDN=OFF; OidnDenoise::isAvailable()
// reports the build-time setting so the caller can fall back to the
// existing 5x5 bilateral or print a "not available" error.
//
// Operates on HDR float radiance values (pre-tone-mapping). Aux buffers
// (albedo, normal) are optional — pass empty vectors and OIDN runs in
// plain-RGB mode (lower quality, but still better than no denoise).
namespace OidnDenoise
{
    bool isAvailable();

    // Denoise `color` in place. `albedo` and `normal` are optional
    // (pass empty to skip). All inputs must have width*height pixels in
    // row-major order, Vec3f packed (3 floats per pixel).
    //
    // Returns true if OIDN ran (always when isAvailable()); false if the
    // build doesn't have OIDN linked.
    bool denoise(std::vector<Vec3f> &color,
                 const std::vector<Vec3f> &albedo,
                 const std::vector<Vec3f> &normal,
                 int width, int height);
}
