// Colorspace unit tests.
//
// CIE matrices and the yBarIntegral normalization constant are the
// glue between the path tracer's per-bounce per-wavelength radiance
// and the linear-sRGB framebuffer. The "black walls" bug from the
// spectral branch (project_pcr_spectral.md, bug 2) traced back to a
// unit-convention mismatch around yBarIntegral; this file pins the
// constant and the matrices in place so any future regression is
// caught by a fast unit test rather than a render diff.

#include <cmath>
#include <cstdio>

#include "CIE.h"
#include "Spectrum.h"
#include "Vec3f.h"

namespace {

int g_failed = 0;

void check(bool ok, const char *label)
{
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) g_failed++;
}

bool nearly(float a, float b, float eps)
{
    return std::fabs(a - b) <= eps;
}

void test_matrix_roundtrip()
{
    // sRGB -> XYZ -> sRGB should recover the original within FP tolerance.
    // Exercise primaries and white plus a chromatic mix.
    const Vec3f cases[] = {
        Vec3f(1.f, 0.f, 0.f),
        Vec3f(0.f, 1.f, 0.f),
        Vec3f(0.f, 0.f, 1.f),
        Vec3f(1.f, 1.f, 1.f),
        Vec3f(0.4f, 0.7f, 0.2f),
    };
    for (const auto &rgb : cases) {
        Vec3f xyz = CIE::linearSRGBToXYZ(rgb);
        Vec3f back = CIE::xyzToLinearSRGB(xyz);
        bool ok = nearly(back[0], rgb[0], 1e-4f)
               && nearly(back[1], rgb[1], 1e-4f)
               && nearly(back[2], rgb[2], 1e-4f);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "sRGB->XYZ->sRGB roundtrip (%.2f, %.2f, %.2f)",
                      rgb[0], rgb[1], rgb[2]);
        check(ok, buf);
    }
}

void test_white_xyz_d65()
{
    // sRGB white maps to the D65 white point: X ~= 0.9505, Y == 1, Z ~= 1.0891.
    Vec3f xyz = CIE::linearSRGBToXYZ(Vec3f(1.f, 1.f, 1.f));
    check(nearly(xyz[0], 0.9505f, 1e-3f), "white XYZ X ~= 0.9505 (D65)");
    check(nearly(xyz[1], 1.0f,    1e-4f), "white XYZ Y == 1.0");
    check(nearly(xyz[2], 1.0891f, 1e-3f), "white XYZ Z ~= 1.0891 (D65)");
}

void test_y_bar_integral_pin()
{
    // yBarIntegral is the integral of the Wyman-piecewise-Gaussian yBar
    // CMF over the discrete 61-sample, 5 nm grid. The numerical value
    // is what bridges the "fit-to-XYZ" and physical-reflectance unit
    // conventions; pin it so any change to either the CMF approximation
    // or the discretization is caught by this test.
    float y = CIE::yBarIntegral();
    check(nearly(y, 106.895f, 0.5f), "yBarIntegral ~= 106.895");
}

void test_perfect_white_reflector_xyz()
{
    // s(lambda) = 1 everywhere -> spectrumToXYZ should produce Y == 1
    // exactly (that's the load-bearing convention pin). X and Z drift
    // from the canonical D65 ratios (X/Y = 0.9505, Z/Y = 1.0891)
    // because the Wyman 2013 piecewise-Gaussian approximation differs
    // from the tabulated CIE 1931 CMFs by ~1%, so the integrated
    // X/Y and Z/Y for a constant-1 spectrum land at ~0.998 and ~0.996.
    // Pin those measured values so any change to the CMF approximation
    // or the 5 nm discretization fails this test loudly.
    Spectrum white(1.f);
    Vec3f xyz = CIE::spectrumToXYZ(white);
    check(nearly(xyz[0], 0.9976f, 5e-3f), "white spectrum -> XYZ X ~= 0.998 (Wyman)");
    check(nearly(xyz[1], 1.0f,    1e-4f), "white spectrum -> XYZ Y == 1.0");
    check(nearly(xyz[2], 0.9959f, 5e-3f), "white spectrum -> XYZ Z ~= 0.996 (Wyman)");
}

void test_black_spectrum()
{
    Spectrum black;
    Vec3f xyz = CIE::spectrumToXYZ(black);
    check(nearly(xyz[0], 0.f, 1e-6f) && nearly(xyz[1], 0.f, 1e-6f)
          && nearly(xyz[2], 0.f, 1e-6f),
          "black spectrum -> XYZ == 0");
}

void test_single_lambda_agrees_with_spectrum()
{
    // singleLambdaXYZ produces a per-sample XYZ contribution that, when
    // averaged over uniform-lambda samples, agrees with the full
    // spectrumToXYZ integral for a constant-radiance spectrum. The
    // path tracer relies on this for spectral-mode brightness parity.
    //
    // Sum of singleLambdaXYZ at every kSamples wavelength, divided by
    // kSamples, should match spectrumToXYZ(constant=1).
    Spectrum one(1.f);
    Vec3f fullIntegral = CIE::spectrumToXYZ(one);

    Vec3f mc(0.f, 0.f, 0.f);
    for (int i = 0; i < Spectrum::kSamples; i++) {
        Vec3f c = CIE::singleLambdaXYZ(Spectrum::lambdaAt(i), 1.f);
        mc[0] += c[0]; mc[1] += c[1]; mc[2] += c[2];
    }
    mc[0] /= float(Spectrum::kSamples);
    mc[1] /= float(Spectrum::kSamples);
    mc[2] /= float(Spectrum::kSamples);

    // Tolerance: the full integral uses a midpoint-style discrete sum
    // and singleLambdaXYZ uses the (kLambdaMax - kLambdaMin) range
    // factor. They agree to better than 1% on a constant-1 spectrum.
    check(nearly(mc[0], fullIntegral[0], 0.02f), "singleLambdaXYZ mean ~= spectrumToXYZ X");
    check(nearly(mc[1], fullIntegral[1], 0.02f), "singleLambdaXYZ mean ~= spectrumToXYZ Y");
    check(nearly(mc[2], fullIntegral[2], 0.02f), "singleLambdaXYZ mean ~= spectrumToXYZ Z");
}

} // namespace

int main()
{
    test_matrix_roundtrip();
    test_white_xyz_d65();
    test_y_bar_integral_pin();
    test_perfect_white_reflector_xyz();
    test_black_spectrum();
    test_single_lambda_agrees_with_spectrum();

    if (g_failed) {
        std::printf("\n%d FAIL(s)\n", g_failed);
        return 1;
    }
    std::printf("\nAll colorspace tests passed.\n");
    return 0;
}
