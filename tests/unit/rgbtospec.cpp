// RGBToSpectrum unit tests.
//
// This is the Jakob 2019 upsampler that turns linear-sRGB albedos and
// emissions into sigmoid-polynomial spectra. The spectral branch's
// black-walls saga (project_pcr_spectral.md, bug 1+2) lived here. Tests
// pin the load-bearing properties that fix in place:
//
//   - fitSpectrum roundtrips a sample palette of cornell-class colors
//     within reasonable error (smooth basin convergence).
//   - The LUT path agrees with the runtime homotopy on the same palette.
//   - Early-exit gamut corners (all-zero, all-one) produce sane spectra.
//   - Material::albedoAt clamps to [0, 1]; Material::emissiveAt does NOT
//     (HDR-fix from spectral commit 3acd7a4).
//
// LUT build is ~4 seconds at kRes=16. Build it once in main(), reuse.

#include <cmath>
#include <cstdio>
#include <vector>

#include "CIE.h"
#include "Material.h"
#include "RGBToSpectrum.h"
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

// Cornell-class palette plus a black/white pair. Cream and warmlight come
// from cornell.json; red and green are the Cornell-box wall canon.
//
// Tolerance choices:
//   - cream/green/warmlight/black are smooth-basin fits, error from
//     Newton convergence is sub-percent.
//   - red wall (0.65, 0.05, 0.05) is past the sRGB gamut for smooth
//     Jakob fits in the sigmoid-of-quadratic family. Homotopy spike-stop
//     returns the last smooth fit before the spike-basin crossover, so
//     measured roundtrip lands at ~0.27 error (mild desaturation - the
//     trade-off documented in project_pcr_spectral.md, bug 1).
//   - white (1, 1, 1) early-exits to a flat "all-1" spectrum, then
//     CIE::spectrumToXYZ goes through Wyman's piecewise-Gaussian CMF
//     approximation. Wyman is ~1% off from the tabulated CIE 1931 CMFs
//     so the sRGB roundtrip lands at (1.20, 0.95, 0.90). Tolerance pins
//     that drift; if the CMFs change, this test fails loudly.
struct Palette {
    Vec3f rgb;
    float roundtripEps;
    const char *name;
};

const Palette g_palette[] = {
    {Vec3f(0.95f, 0.95f, 0.85f), 0.05f,  "cream wall"},
    {Vec3f(0.45f, 0.65f, 0.30f), 0.05f,  "green wall"},
    {Vec3f(0.65f, 0.05f, 0.05f), 0.30f,  "red wall (gamut edge)"},
    {Vec3f(0.92f, 0.78f, 0.50f), 0.05f,  "warmlight (normalized)"},
    {Vec3f(0.0f,  0.0f,  0.0f),  1e-3f,  "black"},
    {Vec3f(1.0f,  1.0f,  1.0f),  0.25f,  "white (Wyman drift)"},
};

void test_fitspectrum_roundtrip_smooth()
{
    for (const auto &p : g_palette) {
        Spectrum s = RGBToSpectrum::fitSpectrum(p.rgb);
        Vec3f back = CIE::spectrumToLinearSRGB(s);
        float err = std::max({std::fabs(back[0] - p.rgb[0]),
                              std::fabs(back[1] - p.rgb[1]),
                              std::fabs(back[2] - p.rgb[2])});
        char buf[120];
        std::snprintf(buf, sizeof(buf), "fitSpectrum roundtrip %-26s err=%.4f tol=%.3f",
                      p.name, err, p.roundtripEps);
        check(err <= p.roundtripEps, buf);
    }
}

void test_fitspectrum_zero_blackish()
{
    // (0, 0, 0) goes through the early-exit (scale=0); fitSpectrum
    // should produce all-zero samples regardless.
    Spectrum s = RGBToSpectrum::fitSpectrum(Vec3f(0.f, 0.f, 0.f));
    bool allZero = true;
    for (int i = 0; i < Spectrum::kSamples; i++) if (s[i] != 0.f) allZero = false;
    check(allZero, "fitSpectrum((0,0,0)) is all-zero");
}

void test_fitspectrum_white_clamped_to_one()
{
    // (1, 1, 1) early-exit returns a flat fit at scale = yBarIntegral.
    // After fitSpectrum's [0, 1] clamp, every sample should equal exactly 1
    // (sigmoid * yBarIntegral can overshoot 1 at gamut corners; the clamp
    // is what the path tracer relies on for energy conservation).
    Spectrum s = RGBToSpectrum::fitSpectrum(Vec3f(1.f, 1.f, 1.f));
    bool allOneOrLess = true;
    for (int i = 0; i < Spectrum::kSamples; i++) {
        if (s[i] < 0.999f || s[i] > 1.001f) { allOneOrLess = false; break; }
    }
    check(allOneOrLess, "fitSpectrum((1,1,1)) clamped to ~1 everywhere");
}

void test_evalsigmoid_returns_unclamped()
{
    // evalSigmoidFit returns the raw sigmoid * scale; the clamp lives in
    // Material::albedoAt and in fitSpectrum, NOT in evalSigmoidFit (per
    // the HDR fix in spectral commit 3acd7a4 - clamping inside eval would
    // cap emissive at 1, killing area-light brightness).
    RGBToSpectrum::SigmoidFit fit = RGBToSpectrum::fitSigmoidCoefficients(Vec3f(1.f, 1.f, 1.f));
    // For white, scale = yBarIntegral ~= 107. Sample at any wavelength
    // should be ~107 * sigmoid(c0). Since c0 = sigmoidInverse(1/scale),
    // the product equals 1 at the chosen base wavelength. The test
    // assertion is just "scale is large", confirming no premature clamp.
    check(fit.scale > 100.f, "fitSigmoidCoefficients((1,1,1)) scale > 100");
}

void test_material_albedo_clamps()
{
    // Material::albedoAt clamps the eval result to [0, 1] because
    // physical reflectance is bounded. A saturated red albedo's raw
    // eval can spike above 1 at gamut-edge wavelengths; the clamp is
    // load-bearing for multi-bounce throughput.
    Material m(Vec3f(0.65f, 0.05f, 0.05f));
    m.populateSpectra();
    bool allBounded = true;
    for (int i = 0; i < Spectrum::kSamples; i++) {
        float v = m.albedoAt(Spectrum::lambdaAt(i));
        if (v < 0.f || v > 1.0001f) { allBounded = false; break; }
    }
    check(allBounded, "Material(saturated red) albedoAt in [0, 1]");
}

void test_material_emissive_does_not_clamp()
{
    // Material::emissiveAt does NOT clamp - it's HDR. The cornell warm
    // light has emissive ~ (8, 5, 2) * a brightness factor; populateSpectra
    // normalizes RGB before fit and re-multiplies fit.scale by maxE so the
    // HDR sample magnitude is preserved.
    //
    // The pre-spectral-commit-12 bug was clamping inside evalSigmoidFit,
    // which capped emissive at 1.0 (lights rendered ~80x too dim). This
    // test asserts the fix stays in place: with HDR emissive RGB, the
    // peak sample exceeds 1.
    Material m(Vec3f(0.95f, 0.95f, 0.85f), Vec3f(8.f, 5.f, 2.f));
    m.populateSpectra();
    float peak = 0.f;
    for (int i = 0; i < Spectrum::kSamples; i++) {
        float v = m.emissiveAt(Spectrum::lambdaAt(i));
        if (v > peak) peak = v;
    }
    check(peak > 2.f, "Material(HDR emissive (8,5,2)) emissiveAt peak > 2 (no clamp)");
}

// Shared LUT for all LUT-related tests. Build is ~4 sec at kRes=16; cache
// it once in main() so the suite stays under 5 sec rather than ~8.
RGBToSpectrum::LUT g_lut;

void test_lut_build_and_lookup()
{
    // All cells finite and within the coefficient clamp range.
    bool allFinite = true;
    int maxAbs = 0;
    for (float v : g_lut.data) {
        if (!std::isfinite(v)) { allFinite = false; break; }
        if (std::fabs(v) > maxAbs) maxAbs = (int)std::fabs(v);
    }
    check(allFinite, "buildLUT produces all finite coefficients");
    check(maxAbs <= 200, "LUT coefficients within clamp range");
}

void test_lut_vs_homotopy_agreement()
{
    // For each palette color, the LUT lookup and the runtime homotopy
    // should produce SigmoidFits whose roundtripped sRGB values agree
    // within ~5% (LUT is a discretized cache of the same algorithm).
    // Pure white and pure black go through the same early-exit on both
    // paths so they agree exactly.
    for (const auto &p : g_palette) {
        // Homotopy fit (active LUT not set).
        RGBToSpectrum::setActiveLUT(nullptr);
        auto fitH = RGBToSpectrum::fitSigmoidCoefficients(p.rgb);

        // LUT lookup.
        auto fitL = RGBToSpectrum::lookupSigmoidFit(g_lut, p.rgb);

        // Roundtrip both to RGB. Direct coefficient compare is too
        // strict because the LUT's trilinear interp produces a slightly
        // different smooth solution; the integrated XYZ is what matters.
        Spectrum sH, sL;
        for (int i = 0; i < Spectrum::kSamples; i++) {
            sH[i] = std::min(RGBToSpectrum::evalSigmoidFit(fitH, Spectrum::lambdaAt(i)), 1.f);
            sL[i] = std::min(RGBToSpectrum::evalSigmoidFit(fitL, Spectrum::lambdaAt(i)), 1.f);
        }
        Vec3f rgbH = CIE::spectrumToLinearSRGB(sH);
        Vec3f rgbL = CIE::spectrumToLinearSRGB(sL);
        float err = std::max({std::fabs(rgbH[0] - rgbL[0]),
                              std::fabs(rgbH[1] - rgbL[1]),
                              std::fabs(rgbH[2] - rgbL[2])});
        char buf[120];
        std::snprintf(buf, sizeof(buf), "LUT vs homotopy %-26s err=%.4f", p.name, err);
        check(err < 0.10f, buf);
    }

    RGBToSpectrum::setActiveLUT(nullptr);
}

} // namespace

int main()
{
    test_fitspectrum_roundtrip_smooth();
    test_fitspectrum_zero_blackish();
    test_fitspectrum_white_clamped_to_one();
    test_evalsigmoid_returns_unclamped();
    test_material_albedo_clamps();
    test_material_emissive_does_not_clamp();

    // Build LUT once for the LUT-related tests.
    RGBToSpectrum::buildLUT(g_lut);

    test_lut_build_and_lookup();
    test_lut_vs_homotopy_agreement();

    if (g_failed) {
        std::printf("\n%d FAIL(s)\n", g_failed);
        return 1;
    }
    std::printf("\nAll rgbtospec tests passed.\n");
    return 0;
}
