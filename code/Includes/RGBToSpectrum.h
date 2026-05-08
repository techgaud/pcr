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

    // Solve the 3x3 system Jx = b in-place (Cramer's rule). Returns
    // false if J is numerically singular. Defined in RGBToSpectrum.cpp.
    bool solve3x3(const float J[3][3], const float b[3], float x[3]);

    // Fit sigmoid-polynomial coefficients to a target XYZ color via
    // Newton-Raphson. Defined in RGBToSpectrum.cpp.
    void fitCoefficients(const Vec3f &targetXYZ,
                         float &outC0, float &outC1, float &outC2);

    // Linear-sRGB albedo to a 61-sample Spectrum, suitable for
    // direct storage on a Material. Defined in RGBToSpectrum.cpp.
    Spectrum fitSpectrum(const Vec3f &rgbLinear);
}
