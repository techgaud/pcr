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

    // Solve the 3x3 system Jx = b in-place (Cramer's rule, the
    // straightforward choice at this size). Returns false if J is
    // numerically singular, in which case the caller should bail.
    // Performance is irrelevant; this runs maybe 60 times per
    // scene load and the cost is dwarfed by the spectrum integrals.
    inline bool solve3x3(const float J[3][3], const float b[3], float x[3])
    {
        float det =
              J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
        if (std::abs(det) < 1e-12f) return false;
        float invDet = 1.f / det;

        float d0 =
              b[0]    * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (b[1]    * J[2][2] - J[1][2] * b[2])
            + J[0][2] * (b[1]    * J[2][1] - J[1][1] * b[2]);
        float d1 =
              J[0][0] * (b[1]    * J[2][2] - J[1][2] * b[2])
            - b[0]    * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * b[2]    - b[1]    * J[2][0]);
        float d2 =
              J[0][0] * (J[1][1] * b[2]    - b[1]    * J[2][1])
            - J[0][1] * (J[1][0] * b[2]    - b[1]    * J[2][0])
            + b[0]    * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

        x[0] = d0 * invDet;
        x[1] = d1 * invDet;
        x[2] = d2 * invDet;
        return true;
    }

    // Fit sigmoid-polynomial coefficients to a target XYZ color.
    // Newton-Raphson with a max-iteration bound; converges in 5-10
    // iterations for the unsaturated colors typical of physically-
    // plausible reflectance, slower for highly saturated targets.
    //
    // residual(c) = integrate(spectrum(c)) - targetXYZ
    // jacobian(c) = d(residual)/d(c), computed analytically by
    //   chain-ruling through the sigmoid.
    //
    // Returns the fitted coefficients via outC0/outC1/outC2.
    inline void fitCoefficients(const Vec3f &targetXYZ,
                                float &outC0, float &outC1, float &outC2)
    {
        constexpr float kLambdaMid  = 0.5f * (Spectrum::kLambdaMin + Spectrum::kLambdaMax);
        constexpr float kLambdaHalf = 0.5f * (Spectrum::kLambdaMax - Spectrum::kLambdaMin);
        constexpr int   kMaxIter    = 15;
        constexpr float kTolerance  = 1e-6f;

        // Flat-grey initial guess: c = (0, 0, 0) gives sigmoid(0) =
        // 0.5 at every wavelength, an unbiased starting point that
        // leaves the solver free to push toward whatever shape the
        // target requires.
        float c[3] = {0.f, 0.f, 0.f};

        for (int iter = 0; iter < kMaxIter; iter++)
        {
            // Integrate spectrum(c) against the CIE observer to get
            // current XYZ, and compute the Jacobian columns at the
            // same time so we only walk the wavelength range once.
            float xyz[3] = {0.f, 0.f, 0.f};
            float J[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

            for (int i = 0; i < Spectrum::kSamples; i++)
            {
                float lambda     = Spectrum::lambdaAt(i);
                float lambdaNorm = (lambda - kLambdaMid) / kLambdaHalf;
                float p          = c[0] + lambdaNorm * (c[1] + lambdaNorm * c[2]);
                float s          = sigmoid(p);
                float ds         = sigmoidDerivative(p);

                float xb = CIE::xBar(lambda);
                float yb = CIE::yBar(lambda);
                float zb = CIE::zBar(lambda);

                xyz[0] += s * xb;
                xyz[1] += s * yb;
                xyz[2] += s * zb;

                // dS/dc0 = ds, dS/dc1 = ds * lambdaNorm,
                // dS/dc2 = ds * lambdaNorm^2.
                float ln2 = lambdaNorm * lambdaNorm;
                J[0][0] += ds * xb;
                J[0][1] += ds * lambdaNorm * xb;
                J[0][2] += ds * ln2 * xb;
                J[1][0] += ds * yb;
                J[1][1] += ds * lambdaNorm * yb;
                J[1][2] += ds * ln2 * yb;
                J[2][0] += ds * zb;
                J[2][1] += ds * lambdaNorm * zb;
                J[2][2] += ds * ln2 * zb;
            }

            // dlambda factor on every sum (Riemann integral).
            float dl = Spectrum::kStep;
            for (int k = 0; k < 3; k++)
            {
                xyz[k] *= dl;
                for (int j = 0; j < 3; j++) J[k][j] *= dl;
            }

            float residual[3] = {
                xyz[0] - targetXYZ[0],
                xyz[1] - targetXYZ[1],
                xyz[2] - targetXYZ[2]
            };
            float rNorm = std::sqrt(residual[0]*residual[0]
                                  + residual[1]*residual[1]
                                  + residual[2]*residual[2]);
            if (rNorm < kTolerance) break;

            float delta[3];
            if (!solve3x3(J, residual, delta)) break;

            // Damped step. Pure Newton can overshoot at the gamut
            // boundaries where the Jacobian gets ill-conditioned;
            // halving on residual increase is a cheap safeguard.
            float prevC[3] = {c[0], c[1], c[2]};
            float step = 1.f;
            for (int trial = 0; trial < 4; trial++)
            {
                c[0] = prevC[0] - step * delta[0];
                c[1] = prevC[1] - step * delta[1];
                c[2] = prevC[2] - step * delta[2];

                // Quick residual check at the trial point.
                float testXYZ[3] = {0.f, 0.f, 0.f};
                for (int i = 0; i < Spectrum::kSamples; i++)
                {
                    float lambda     = Spectrum::lambdaAt(i);
                    float lambdaNorm = (lambda - kLambdaMid) / kLambdaHalf;
                    float p          = c[0] + lambdaNorm * (c[1] + lambdaNorm * c[2]);
                    float s          = sigmoid(p);
                    testXYZ[0] += s * CIE::xBar(lambda);
                    testXYZ[1] += s * CIE::yBar(lambda);
                    testXYZ[2] += s * CIE::zBar(lambda);
                }
                testXYZ[0] *= dl; testXYZ[1] *= dl; testXYZ[2] *= dl;

                float trialR[3] = {
                    testXYZ[0] - targetXYZ[0],
                    testXYZ[1] - targetXYZ[1],
                    testXYZ[2] - targetXYZ[2]
                };
                float trialNorm = std::sqrt(trialR[0]*trialR[0]
                                          + trialR[1]*trialR[1]
                                          + trialR[2]*trialR[2]);
                if (trialNorm < rNorm) break;
                step *= 0.5f;
            }
        }

        outC0 = c[0]; outC1 = c[1]; outC2 = c[2];
    }

    // Convenience: linear-sRGB albedo to a 61-sample Spectrum,
    // suitable for direct storage on a Material. Black and white
    // are special-cased because sigmoid asymptotes to 0 and 1
    // respectively without ever reaching them; Newton-Raphson
    // would still converge close, but the shortcut is exact and
    // cheap.
    inline Spectrum fitSpectrum(const Vec3f &rgbLinear)
    {
        float r = rgbLinear[0], g = rgbLinear[1], b = rgbLinear[2];
        if (r <= 0.f && g <= 0.f && b <= 0.f) return Spectrum(0.f);
        if (r >= 1.f && g >= 1.f && b >= 1.f) return Spectrum(1.f);

        Vec3f xyz = CIE::linearSRGBToXYZ(rgbLinear);
        float c0, c1, c2;
        fitCoefficients(xyz, c0, c1, c2);
        return Spectrum::fromSigmoidCoefficients(c0, c1, c2);
    }
}
