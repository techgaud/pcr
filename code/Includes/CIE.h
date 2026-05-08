#pragma once

#include "Spectrum.h"
#include "Vec3f.h"

// CIE 1931 2-degree standard observer color-matching functions and
// the conversion from CIE XYZ tristimulus to linear sRGB.
//
// We use Wyman, Sloan & Shirley (2013), "Simple Analytic
// Approximations to the CIE XYZ Color Matching Functions". Their
// piecewise-Gaussian fits match the 1931 tabulated CMFs to within
// about 1% across the visible range, while collapsing what would
// otherwise be 3 x 81-entry tables into ~30 lines of analytic code.
// For a spectral path tracer at hobby precision, this is the right
// trade. Production renderers (Mitsuba, PBRT) ship the full
// tabulated CMFs because a 1% mismatch becomes visible after a
// chain of color transforms; we don't have that pipeline depth.
//
// Reference: J. Comput. Graph. Tech. Vol. 2, No. 2, 2013.

namespace CIE
{
    // Wyman-piecewise-Gaussian helper. g(x, mu, sigma1, sigma2)
    // returns a Gaussian whose left half (x < mu) uses sigma1 and
    // right half uses sigma2. Used because the visible-light CMFs
    // are asymmetric about their peaks.
    inline float wymanGauss(float lambda, float mu, float sigma1, float sigma2)
    {
        float t = (lambda < mu) ? (lambda - mu) * sigma1 : (lambda - mu) * sigma2;
        return std::exp(-0.5f * t * t);
    }

    // Sum of three Gaussian lobes. Matches the actual x-bar curve's
    // double-bump (small lobe near 442 nm, large lobe near 600 nm,
    // small negative-corrected dip near 501 nm).
    inline float xBar(float lambda)
    {
        return  0.362f * wymanGauss(lambda, 442.0f, 0.0624f, 0.0374f)
              + 1.056f * wymanGauss(lambda, 599.8f, 0.0264f, 0.0323f)
              - 0.065f * wymanGauss(lambda, 501.1f, 0.0490f, 0.0382f);
    }

    inline float yBar(float lambda)
    {
        return  0.821f * wymanGauss(lambda, 568.8f, 0.0213f, 0.0247f)
              + 0.286f * wymanGauss(lambda, 530.9f, 0.0613f, 0.0322f);
    }

    inline float zBar(float lambda)
    {
        return  1.217f * wymanGauss(lambda, 437.0f, 0.0845f, 0.0278f)
              + 0.681f * wymanGauss(lambda, 459.0f, 0.0385f, 0.0725f);
    }

    // Integrate a sampled spectrum against the CIE 1931 observer to
    // produce CIE XYZ tristimulus values. Riemann-sum on the
    // Spectrum's discretization (5 nm steps from 400-700 nm).
    //
    // The kSampleStep multiplier is the dlambda factor that turns
    // a sum of samples into an integral approximation. It's a
    // global scale that ultimately gets folded into the final
    // exposure constant; we keep it here for dimensional honesty.
    inline Vec3f spectrumToXYZ(const Spectrum &s)
    {
        float X = 0.f, Y = 0.f, Z = 0.f;
        for (int i = 0; i < Spectrum::kSamples; i++)
        {
            float lambda = Spectrum::lambdaAt(i);
            float v = s[i];
            X += v * xBar(lambda);
            Y += v * yBar(lambda);
            Z += v * zBar(lambda);
        }
        return Vec3f(X * Spectrum::kStep, Y * Spectrum::kStep, Z * Spectrum::kStep);
    }

    // Linear sRGB to CIE XYZ (D65 white point). Inverse of the
    // matrix below; used by RGBToSpectrum to convert a target RGB
    // albedo into the XYZ space the Newton-Raphson solver fits.
    inline Vec3f linearSRGBToXYZ(const Vec3f &rgb)
    {
        float R = rgb[0], G = rgb[1], B = rgb[2];
        return Vec3f(
            0.4124564f * R + 0.3575761f * G + 0.1804375f * B,
            0.2126729f * R + 0.7151522f * G + 0.0721750f * B,
            0.0193339f * R + 0.1191920f * G + 0.9503041f * B
        );
    }

    // CIE XYZ to linear sRGB (D65 white point). Standard 3x3 matrix
    // from IEC 61966-2-1; reproduced in every color-management spec.
    // Output is linear-light sRGB, so the existing tone-map +
    // gamma-correction stage downstream still applies unchanged.
    //
    // Negative components are possible for highly saturated XYZ
    // values that fall outside the sRGB gamut. Path-tracer code
    // should clamp to zero before writing to the LDR framebuffer.
    inline Vec3f xyzToLinearSRGB(const Vec3f &xyz)
    {
        float X = xyz[0], Y = xyz[1], Z = xyz[2];
        return Vec3f(
             3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z,
            -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z,
             0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z
        );
    }

    // Convenience: end-to-end Spectrum -> linear sRGB. The path
    // tracer's accumulator goes here before tone-map.
    inline Vec3f spectrumToLinearSRGB(const Spectrum &s)
    {
        return xyzToLinearSRGB(spectrumToXYZ(s));
    }

    // Single-wavelength variant: when a path tracer is in single-
    // lambda mode, each ray contributes scalar radiance(lambda) to a
    // single color channel weighted by the CIE observer at that
    // wavelength. The pixel accumulator stores running XYZ; this
    // helper produces the per-sample XYZ contribution for one ray.
    //
    // The 1/numLambdaSamples scaling that turns the Monte Carlo
    // estimator from sum-over-samples into an unbiased integral
    // belongs at the pixel accumulator, not here. This function is
    // pure: scalar in, vector out, no implicit averaging.
    inline Vec3f singleLambdaXYZ(float lambda, float radiance)
    {
        // Lambda is sampled uniformly on [kLambdaMin, kLambdaMax].
        // For a uniform PDF, the sampling weight (kLambdaMax -
        // kLambdaMin) cancels with the integration step in
        // spectrumToXYZ above. We carry it here so the absolute
        // brightness matches the full-spectrum case.
        constexpr float kLambdaRange = Spectrum::kLambdaMax - Spectrum::kLambdaMin;
        return Vec3f(
            radiance * xBar(lambda) * kLambdaRange,
            radiance * yBar(lambda) * kLambdaRange,
            radiance * zBar(lambda) * kLambdaRange
        );
    }
}
