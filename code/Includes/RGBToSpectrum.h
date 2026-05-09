#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "CIE.h"
#include "Spectrum.h"
#include "Vec3f.h"

// Jakob & Hanika 2019 RGB-to-spectrum upsampling, in-process flavor.
//
// Given a target linear-sRGB color, find sigmoid-polynomial
// coefficients (c0, c1, c2) such that the spectrum
//   S(lambda) = sigmoid(c0 + c1*lambda' + c2*lambda'^2)
// integrated against the CIE 1931 observer reproduces the target
// color exactly, and use Newton-Raphson to converge there starting
// from a flat-grey initial guess. Sigmoid is the smooth saturator
//   sigmoid(x) = 0.5 + x / (2 sqrt(1 + x^2))
// which has the very useful property that the output is always in
// [0, 1], automatically enforcing the physical reflectance bound
// regardless of which coefficients we settle on.
//
// We do this at scene-load time, once per material. The expected
// per-call cost is well under a millisecond; on every Cornell-class
// scene we own (4-5 materials), the total upsampling cost is in the
// noise. The Jakob paper precomputes a 64x64x64 LUT of these
// coefficients to amortize across many materials, but that's an
// optimization for production scenes with thousands of textures.
// We don't have those.
//
// Reference: Jakob & Hanika, "A Low-Dimensional Function Space for
// Efficient Spectral Upsampling" (2019).

namespace RGBToSpectrum
{
    // The smooth saturator and its derivative, used by the solver.
    // The derivative form 1 / (2 (1 + x^2)^(3/2)) follows from
    // differentiating the sigmoid expression; both are inlined so
    // the compiler can fold the math into the per-sample loops.
    inline float sigmoid(float x)
    {
        return 0.5f + x / (2.f * std::sqrt(1.f + x * x));
    }

    inline float sigmoidDerivative(float x)
    {
        float t = 1.f + x * x;
        return 0.5f / (t * std::sqrt(t));
    }

    // Inverse of sigmoid above: given a target output y in (0, 1),
    // return x with sigmoid(x) = y. Solving 2(y - 0.5) = x/sqrt(1+x^2)
    // gives x = u/sqrt(1-u^2) where u = 2y - 1. Used by fitCoefficients
    // to seed Newton-Raphson at a flat spectrum near the target's
    // luminance, which keeps the solver in the smooth-solution basin.
    inline float sigmoidInverse(float y)
    {
        y = std::clamp(y, 1e-4f, 1.f - 1e-4f);
        float u = 2.f * y - 1.f;
        return u / std::sqrt(1.f - u * u);
    }

    // Solve the 3x3 system Jx = b in-place (Cramer's rule). Returns
    // false if J is numerically singular. Defined in RGBToSpectrum.cpp.
    bool solve3x3(const float J[3][3], const float b[3], float x[3]);

    // Fit sigmoid-polynomial coefficients to a target XYZ color via
    // Newton-Raphson. Defined in RGBToSpectrum.cpp.
    void fitCoefficients(const Vec3f &targetXYZ,
                         float &outC0, float &outC1, float &outC2);

    // Lower-level Newton-Raphson core: takes a warm-start (cIn) and
    // produces the converged coefficients (cOut) for the given XYZ.
    // No homotopy; the warm-start has to already be in (or near) the
    // smooth basin. Used by the LUT builder, which warm-starts each
    // cell from the previous brightness step. Defined in
    // RGBToSpectrum.cpp.
    void newtonFit(const Vec3f &targetXYZ, const float cIn[3], float cOut[3]);

    // Compact form for runtime spectrum evaluation: the four floats the
    // GPU needs to reconstruct a Spectrum sample at any wavelength
    // without storing the 61-sample table. evalSigmoidFit below is the
    // canonical eval, used by Material::albedoAt / emissiveAt and
    // mirrored byte-for-byte in the GLSL spectral path.
    //
    //   sample(lambda) = sigmoid(c0 + L*c1 + L*L*c2) * scale
    //   L = (lambda - 550) / 150          (normalized to [-1, 1] across visible)
    //
    // For albedo, scale embeds the yBarIntegral conversion from
    // "fit-to-XYZ" to physical reflectance convention; the [0, 1] clamp
    // is applied in Material::albedoAt because reflectance > 1 is
    // unphysical. For emissive, scale additionally embeds the per-
    // material maxE re-scaling that populateSpectra does after fitting
    // normalized RGB; emissive radiance can be HDR so the clamp doesn't
    // apply on the read side either.
    struct SigmoidFit
    {
        float c0    = 0.f;
        float c1    = 0.f;
        float c2    = 0.f;
        float scale = 0.f;
    };

    // Linear-sRGB albedo or normalized-emissive RGB to a SigmoidFit. The
    // emissive caller multiplies the returned scale by maxE before
    // storage. Handles the gamut-corner cases (all-zero, all-one) with
    // synthesized coefficients that produce exact 0 or 1 samples
    // matching fitSpectrum's early-exit results. Defined in
    // RGBToSpectrum.cpp.
    SigmoidFit fitSigmoidCoefficients(const Vec3f &rgbLinear);

    // Evaluate a SigmoidFit at a single wavelength. Header-only because
    // it's on the per-bounce hot path; the compiler folds the polynomial
    // into the calling material accessors. Returns the raw scaled sample
    // without clamping. Material::albedoAt clamps the result to [0, 1]
    // because physical reflectance is bounded; Material::emissiveAt does
    // not clamp because emissive radiance is HDR. Doing the clamp here
    // would saturate emissive samples back to 1.0 once populateSpectra
    // had baked the per-material maxE multiplier into fit.scale.
    inline float evalSigmoidFit(const SigmoidFit &fit, float lambda)
    {
        constexpr float kLambdaMid  = 0.5f * (Spectrum::kLambdaMin + Spectrum::kLambdaMax);
        constexpr float kLambdaHalf = 0.5f * (Spectrum::kLambdaMax - Spectrum::kLambdaMin);
        float L = (lambda - kLambdaMid) / kLambdaHalf;
        float p = fit.c0 + L * (fit.c1 + L * fit.c2);
        return sigmoid(p) * fit.scale;
    }

    // Linear-sRGB albedo to a 61-sample Spectrum, suitable for
    // direct storage on a Material. Defined in RGBToSpectrum.cpp.
    Spectrum fitSpectrum(const Vec3f &rgbLinear);

    // Optional precomputed lookup table (Jakob & Hanika 2019 LUT, in the
    // shape of mitsuba-renderer/rgb2spec). Built in memory at process
    // startup when the user passes --lut, replacing the runtime homotopy
    // continuation in fitSigmoidCoefficients with trilinear interpolation
    // out of a 3-axis grid covering the full sRGB gamut.
    //
    // The build runs fitCoefficients (homotopy) per cell. Mitsuba does
    // something cleverer: warm-start a Gauss-Newton solver through a
    // brightness sweep so most cells take 2-3 iterations instead of the
    // full homotopy. Our analytical Newton-Raphson goes rank-deficient
    // mid-sweep on saturated chromaticities (the sigmoid floors and the
    // Jacobian collapses), so per-cell homotopy is the reliable path
    // without porting Gauss-Newton + finite-difference Jacobian.
    //
    // Lookup is nanoseconds. Build is a few seconds for kRes=16. Same
    // fit quality as the runtime homotopy - this is a precomputed cache
    // of the same algorithm, not a gamut-edge fix. The case where the
    // LUT wins is many-material scenes where amortizing the per-call
    // homotopy across all materials beats running it once per material;
    // for cornell-class scenes the build cost dominates and you should
    // leave --lut off.
    struct LUT
    {
        // Resolution per axis. 16 keeps build time tractable when each
        // cell goes through the full homotopy (mitsuba uses warm-start
        // through brightness with a stabler Gauss-Newton + finite-
        // difference Jacobian; our analytical Newton-Raphson goes rank-
        // deficient mid-chain when the sigmoid saturates, so per-cell
        // homotopy is the reliable path for now). Trilinear interpolation
        // between cells smooths the boundaries.
        static constexpr int kRes = 16;

        // 3 max-channel axes (R-, G-, B-dominant) x kRes brightness x
        // kRes "x" (other-channel-1 / max) x kRes "y" (other-channel-2 /
        // max) x 3 sigmoid coefficients (c0, c1, c2). ~12 MB at kRes=64,
        // ~1.5 MB at kRes=32.
        std::vector<float> data;

        LUT() : data(3 * kRes * kRes * kRes * 3, 0.f) {}

        // Linear address into the flat float array. (l, b, x, y, k) with
        // k in [0, 2] selecting c0/c1/c2.
        int linearIdx(int l, int b, int x, int y) const
        {
            return ((l * kRes + b) * kRes + x) * kRes * 3 + y * 3;
        }
    };

    // Build the LUT in place. Heavy: scene-load-time investment that pays
    // off across every fit thereafter.
    void buildLUT(LUT &lut);

    // Trilinear lookup. Replaces fitSigmoidCoefficients's runtime solver
    // when the LUT is enabled.
    SigmoidFit lookupSigmoidFit(const LUT &lut, const Vec3f &rgbLinear);

    // Process-wide LUT pointer. Set by main() before scene load when
    // --lut is on; left null otherwise. Material::populateSpectra (via
    // fitSigmoidCoefficients) checks it and dispatches accordingly.
    // Single thread of access because scene-load is serial.
    void setActiveLUT(const LUT *lut);
    const LUT *activeLUT();

    // Binary serialization for shippable LUTs. File format:
    //
    //   char[4]   magic   = "PLUT"
    //   uint32    version = 1
    //   uint32    res     (must equal LUT::kRes for the running build)
    //   uint32    nFloats (= 3 * res^3 * 3, redundant but a sanity pin)
    //   float[]   data    (nFloats values, native endianness)
    //
    // Native endianness is fine because pcr is currently x86_64-only on
    // CI and cross-arch LUT exchange isn't a goal. If that ever changes,
    // bump version and add a byte-order field.
    //
    // saveLUT returns true on success. loadLUT returns true and populates
    // `out` on success; on any error (missing file, bad magic, version
    // mismatch, res mismatch, truncated data) it returns false and writes
    // a one-line description to `outError` (if non-null).
    bool saveLUT(const LUT &lut, const std::string &path);
    bool loadLUT(const std::string &path, LUT &out, std::string *outError = nullptr);
}
