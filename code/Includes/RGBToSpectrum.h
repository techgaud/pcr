#pragma once

#include <algorithm>
#include <cmath>

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

    // Compact form for runtime spectrum evaluation: the four floats the
    // GPU needs to reconstruct a Spectrum sample at any wavelength
    // without storing the 61-sample table. evalSigmoidFit below is the
    // canonical eval, used by Material::albedoAt / emissiveAt and
    // mirrored byte-for-byte in the GLSL spectral path.
    //
    //   sample(lambda) = clamp(sigmoid(c0 + L*c1 + L*L*c2) * scale, 0, 1)
    //   L = (lambda - 550) / 150          (normalized to [-1, 1] across visible)
    //
    // For albedo, scale embeds the yBarIntegral conversion from
    // "fit-to-XYZ" to physical reflectance convention. For emissive,
    // scale additionally embeds the per-material maxE re-scaling that
    // populateSpectra does after fitting normalized RGB.
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
    // into the calling material accessors. The 1.f cap matches the
    // physical-reflectance clamp inside fitSpectrum.
    inline float evalSigmoidFit(const SigmoidFit &fit, float lambda)
    {
        constexpr float kLambdaMid  = 0.5f * (Spectrum::kLambdaMin + Spectrum::kLambdaMax);
        constexpr float kLambdaHalf = 0.5f * (Spectrum::kLambdaMax - Spectrum::kLambdaMin);
        float L = (lambda - kLambdaMid) / kLambdaHalf;
        float p = fit.c0 + L * (fit.c1 + L * fit.c2);
        return std::min(sigmoid(p) * fit.scale, 1.f);
    }

    // Linear-sRGB albedo to a 61-sample Spectrum, suitable for
    // direct storage on a Material. Defined in RGBToSpectrum.cpp.
    Spectrum fitSpectrum(const Vec3f &rgbLinear);
}
