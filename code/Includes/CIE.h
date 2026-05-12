#pragma once

#include "Spectrum.h"
#include "Vec3f.h"

// CIE 1931 2-degree standard observer color-matching functions and
// the conversion from CIE XYZ tristimulus to linear sRGB.
//
// Two CMF implementations ship side by side and the renderer
// chooses at runtime:
//
//   1. Wyman, Sloan & Shirley (2013), piecewise-Gaussian fits. The
//      original pcr CMF. Matches the 1931 tabulated values to ~1%
//      across the visible range, compounded through sRGB conversion
//      that becomes ~25% drift in integrated RGB equivalents. Cheap
//      to evaluate (a handful of exp() calls), no table required.
//      Reference: J. Comput. Graph. Tech. Vol. 2, No. 2, 2013.
//
//   2. CIE 1931 tabulated, 61 samples at 5 nm steps from 400 to 700
//      nm. The actual standard CMF used by Mitsuba, PBRT, and every
//      reference renderer. Linear interpolation between samples for
//      off-grid lambdas. ~1 KB constant table, lookup-and-lerp cost
//      similar to the Wyman analytic eval, no measurable perf
//      difference at typical sample counts.
//
// Mode is selected by passing useCieCmf to singleLambdaXYZ /
// spectrumToXYZ / spectrumToLinearSRGB. The default (no arg) is
// Wyman for backward compatibility with the rest of the pipeline
// (Jakob fitter, LUT builder), which haven't been migrated yet.
// Output-side conversion in the renderer is the high-leverage
// place to flip CMFs because the 25% integrated-RGB drift shows
// up directly in the rendered image.

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

    // CIE 1931 2-degree standard observer, tabulated at 5 nm steps
    // from 400 to 700 nm. 61 samples each for x_bar, y_bar, z_bar.
    // Values from the canonical CIE publication (matches PBRT-v4,
    // Mitsuba 3, every reference renderer). y_bar(555) = 1.0 exactly
    // at the y-peak. Out-of-range lambdas return 0, same convention
    // as Spectrum::sampleAt.
    //
    // Single static table, 3 * 61 * 4 = 732 bytes. Negligible.
    struct CieTableEntry { float x, y, z; };
    inline const CieTableEntry *cieTable()
    {
        static const CieTableEntry kCie[Spectrum::kSamples] = {
            {0.0143f,    0.000396f, 0.0679f},   // 400 nm
            {0.0232f,    0.000640f, 0.1102f},   // 405
            {0.0435f,    0.001210f, 0.2074f},   // 410
            {0.0776f,    0.002180f, 0.3713f},   // 415
            {0.13438f,   0.004000f, 0.6456f},   // 420
            {0.21477f,   0.0073f,   1.03905f},  // 425
            {0.2839f,    0.0116f,   1.3856f},   // 430
            {0.3285f,    0.01684f,  1.62296f},  // 435
            {0.34828f,   0.023f,    1.74706f},  // 440
            {0.34806f,   0.0298f,   1.7826f},   // 445
            {0.3362f,    0.038f,    1.77211f},  // 450
            {0.3187f,    0.048f,    1.7441f},   // 455
            {0.2908f,    0.060f,    1.6692f},   // 460
            {0.2511f,    0.0739f,   1.5281f},   // 465
            {0.19536f,   0.09098f,  1.28764f},  // 470
            {0.1421f,    0.1126f,   1.0419f},   // 475
            {0.09564f,   0.13902f,  0.81295f},  // 480
            {0.05795f,   0.1693f,   0.6162f},   // 485
            {0.03201f,   0.20802f,  0.46518f},  // 490
            {0.0147f,    0.2586f,   0.3533f},   // 495
            {0.0049f,    0.323f,    0.272f},    // 500
            {0.0024f,    0.4073f,   0.2123f},   // 505
            {0.0093f,    0.503f,    0.1582f},   // 510
            {0.0291f,    0.6082f,   0.1117f},   // 515
            {0.06327f,   0.710f,    0.07825f},  // 520
            {0.1096f,    0.7932f,   0.05725f},  // 525
            {0.1655f,    0.862f,    0.04216f},  // 530
            {0.22575f,   0.91485f,  0.02984f},  // 535
            {0.2904f,    0.954f,    0.0203f},   // 540
            {0.3597f,    0.9803f,   0.0134f},   // 545
            {0.43345f,   0.99495f,  0.00875f},  // 550
            {0.51205f,   1.000f,    0.00575f},  // 555  (y peak)
            {0.5945f,    0.995f,    0.0039f},   // 560
            {0.6784f,    0.9786f,   0.00275f},  // 565
            {0.7621f,    0.952f,    0.0021f},   // 570
            {0.8425f,    0.9154f,   0.0018f},   // 575
            {0.9163f,    0.870f,    0.00165f},  // 580
            {0.9786f,    0.8163f,   0.0014f},   // 585
            {1.0263f,    0.757f,    0.0011f},   // 590
            {1.0567f,    0.6949f,   0.0010f},   // 595
            {1.0622f,    0.631f,    0.0008f},   // 600
            {1.0456f,    0.5668f,   0.0006f},   // 605
            {1.0026f,    0.503f,    0.00034f},  // 610
            {0.93832f,   0.4412f,   0.00024f},  // 615
            {0.85445f,   0.381f,    0.00019f},  // 620
            {0.7514f,    0.321f,    0.0001f},   // 625
            {0.6424f,    0.265f,    0.00005f},  // 630
            {0.5419f,    0.217f,    0.00003f},  // 635
            {0.4479f,    0.175f,    0.00002f},  // 640
            {0.3608f,    0.1382f,   0.00001f},  // 645
            {0.2835f,    0.107f,    0.0f},      // 650
            {0.2187f,    0.0816f,   0.0f},      // 655
            {0.1649f,    0.061f,    0.0f},      // 660
            {0.1212f,    0.04458f,  0.0f},      // 665
            {0.0874f,    0.032f,    0.0f},      // 670
            {0.0636f,    0.0232f,   0.0f},      // 675
            {0.04677f,   0.017f,    0.0f},      // 680
            {0.0329f,    0.01192f,  0.0f},      // 685
            {0.0227f,    0.00821f,  0.0f},      // 690
            {0.01584f,   0.005723f, 0.0f},      // 695
            {0.01136f,   0.004102f, 0.0f},      // 700
        };
        return kCie;
    }

    // Linear interpolation against the tabulated CMFs. Lambda outside
    // [400, 700] returns 0 to match Spectrum::sampleAt out-of-range
    // semantics.
    inline Vec3f cieXYZ(float lambda)
    {
        if (lambda < Spectrum::kLambdaMin || lambda > Spectrum::kLambdaMax)
            return Vec3f(0.f, 0.f, 0.f);
        float t = (lambda - Spectrum::kLambdaMin) / Spectrum::kStep;
        int i = (int)t;
        if (i >= Spectrum::kSamples - 1)
        {
            const CieTableEntry &e = cieTable()[Spectrum::kSamples - 1];
            return Vec3f(e.x, e.y, e.z);
        }
        float frac = t - (float)i;
        const CieTableEntry &a = cieTable()[i];
        const CieTableEntry &b = cieTable()[i + 1];
        return Vec3f(
            a.x + frac * (b.x - a.x),
            a.y + frac * (b.y - a.y),
            a.z + frac * (b.z - a.z)
        );
    }

    inline float cieXBar(float lambda) { return cieXYZ(lambda)[0]; }
    inline float cieYBar(float lambda) { return cieXYZ(lambda)[1]; }
    inline float cieZBar(float lambda) { return cieXYZ(lambda)[2]; }

    // Integral of yBar over the visible range with our discretization
    // (61 samples at 5 nm steps, Wyman 2013 piecewise-Gaussian
    // approximation). The normalization constant that bridges the two
    // unit conventions for spectra used in this codebase:
    //
    //   "fit-to-XYZ"    s_fit(lambda) integrated against cmf gives RGB
    //                   in [0, 1] linear-sRGB units. For a perfect
    //                   white reflector, s_fit ~= 1/yBarIntegral ~=
    //                   0.0095. Convenient for one-bounce sanity, but
    //                   multi-bounce attenuates by an extra factor of
    //                   yBarIntegral per bounce vs the RGB pipeline,
    //                   leaving everything black.
    //
    //   "physical"      s(lambda) is a reflectance fraction in [0, 1].
    //                   For a perfect white reflector, s = 1 everywhere.
    //                   Multi-bounce throughput attenuates exactly like
    //                   per-channel albedos do in the RGB pipeline.
    //                   Output XYZ accumulator picks up an extra
    //                   yBarIntegral factor that singleLambdaXYZ divides
    //                   out below.
    //
    // The codebase uses the physical convention. RGBToSpectrum::fitSpectrum
    // scales its output by yBarIntegral after Newton-Raphson, and
    // singleLambdaXYZ divides by yBarIntegral on the way out.
    inline float yBarIntegral()
    {
        static const float val = []() {
            float sum = 0.f;
            for (int i = 0; i < Spectrum::kSamples; i++)
                sum += yBar(Spectrum::lambdaAt(i));
            return sum * Spectrum::kStep;
        }();
        return val;
    }

    // CIE-table equivalent of yBarIntegral. Computed once at first
    // access and cached. Sums the tabulated y_bar across all 61
    // pcr-Spectrum samples (400..700 in 5 nm steps). Differs from
    // Wyman's by ~1%, which is exactly the bias we're correcting
    // when the renderer flips to the tabulated CMF.
    inline float cieYBarIntegral()
    {
        static const float val = []() {
            float sum = 0.f;
            for (int i = 0; i < Spectrum::kSamples; i++)
                sum += cieTable()[i].y;
            return sum * Spectrum::kStep;
        }();
        return val;
    }

    // Pick the right integral for the active CMF. Used by spectrum->
    // XYZ normalization sites to keep the unit convention correct
    // regardless of which CMF is integrating.
    inline float yBarIntegralFor(bool useCieCmf)
    {
        return useCieCmf ? cieYBarIntegral() : yBarIntegral();
    }

    // Integrate a sampled spectrum against the CIE 1931 observer to
    // produce CIE XYZ tristimulus values. Defined in CIE.cpp.
    // useCieCmf=true uses the tabulated CMFs, false uses Wyman.
    Vec3f spectrumToXYZ(const Spectrum &s, bool useCieCmf = false);

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
    // tracer's accumulator goes here before tone-map. Defined in
    // CIE.cpp.
    Vec3f spectrumToLinearSRGB(const Spectrum &s, bool useCieCmf = false);

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
    inline Vec3f singleLambdaXYZ(float lambda, float radiance, bool useCieCmf = false)
    {
        // Lambda is sampled uniformly on [kLambdaMin, kLambdaMax].
        // For a uniform PDF, the sampling weight (kLambdaMax -
        // kLambdaMin) cancels with the integration step in
        // spectrumToXYZ above. We carry it here so the absolute
        // brightness matches the full-spectrum case.
        //
        // The 1/yBarIntegral term converts back from the physical
        // reflectance convention (s = 1 for a perfect white reflector)
        // into linear-sRGB-comparable XYZ where Y(white) ~= 1. See
        // yBarIntegral() above.
        constexpr float kLambdaRange = Spectrum::kLambdaMax - Spectrum::kLambdaMin;
        float scale = kLambdaRange / yBarIntegralFor(useCieCmf);
        if (useCieCmf)
        {
            Vec3f cmf = cieXYZ(lambda);
            return Vec3f(
                radiance * cmf[0] * scale,
                radiance * cmf[1] * scale,
                radiance * cmf[2] * scale
            );
        }
        return Vec3f(
            radiance * xBar(lambda) * scale,
            radiance * yBar(lambda) * scale,
            radiance * zBar(lambda) * scale
        );
    }
}
