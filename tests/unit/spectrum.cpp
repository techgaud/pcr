// Spectrum unit tests.
//
// Spectrum is the 61-sample wavelength-radiance container behind every
// spectral path-tracer operation. Bugs here turn into "every render is
// wrong by a constant factor" or "spectral mode reads the wrong sample
// at the wrong wavelength." Both have happened at least once on this
// branch (see project_pcr_spectral.md, bugs 2 and 12).
//
// Each test case prints "PASS:" or "FAIL:" with a short message; main()
// returns 1 if any case failed. Keeps the format aligned with the other
// unit-test files so CI logs are scannable.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Spectrum.h"

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

void test_constants()
{
    // Plan-level sanity: the discretization the rest of the codebase pins to.
    check(Spectrum::kSamples == 61, "kSamples == 61");
    check(nearly(Spectrum::kLambdaMin, 400.f, 1e-6f), "kLambdaMin == 400");
    check(nearly(Spectrum::kLambdaMax, 700.f, 1e-6f), "kLambdaMax == 700");
    check(nearly(Spectrum::kStep, 5.f, 1e-6f), "kStep == 5");
    check(nearly(Spectrum::lambdaAt(0), 400.f, 1e-6f), "lambdaAt(0) == 400");
    check(nearly(Spectrum::lambdaAt(60), 700.f, 1e-6f), "lambdaAt(60) == 700");
    check(nearly(Spectrum::lambdaAt(30), 550.f, 1e-6f), "lambdaAt(30) == 550");
}

void test_default_zero()
{
    Spectrum s;
    bool allZero = true;
    for (int i = 0; i < Spectrum::kSamples; i++) if (s[i] != 0.f) allZero = false;
    check(allZero, "default constructor zero-fills");
    check(s.isBlack(), "isBlack() true on default");
}

void test_constant_fill()
{
    Spectrum s(0.5f);
    bool allHalf = true;
    for (int i = 0; i < Spectrum::kSamples; i++) if (!nearly(s[i], 0.5f, 1e-7f)) allHalf = false;
    check(allHalf, "Spectrum(0.5) fills uniformly");
    check(!s.isBlack(), "isBlack() false on non-zero");
}

void test_interpolation()
{
    // Build a ramp where sample i = i. operator()(lambda) at sample
    // boundary should be exact; midway should be linearly interpolated.
    Spectrum s;
    for (int i = 0; i < Spectrum::kSamples; i++) s[i] = float(i);

    check(nearly(s(400.f), 0.f, 1e-5f), "operator()(400) == sample 0");
    check(nearly(s(405.f), 1.f, 1e-5f), "operator()(405) == sample 1");
    check(nearly(s(700.f), 60.f, 1e-5f), "operator()(700) == sample 60");
    // Halfway between sample 5 (=5) and sample 6 (=6) is wavelength 427.5.
    check(nearly(s(427.5f), 5.5f, 1e-4f), "operator()(427.5) midpoint interp");
}

void test_out_of_range()
{
    Spectrum s(1.f);
    check(s(399.f) == 0.f, "operator()(below min) returns 0");
    check(s(701.f) == 0.f, "operator()(above max) returns 0");
}

void test_integrate()
{
    // Manually computed: sum of i*kStep over 61 samples.
    Spectrum s;
    for (int i = 0; i < Spectrum::kSamples; i++) s[i] = float(i);
    float expected = 0.f;
    for (int i = 0; i < Spectrum::kSamples; i++) expected += float(i);
    expected *= Spectrum::kStep;
    check(nearly(s.integrate(), expected, 1e-3f), "integrate() == manual sum");

    // Constant-1 spectrum integrates to (kSamples * kStep), not the
    // visible-range width directly: it's the discrete approximation,
    // 61 * 5 = 305 nm.
    Spectrum one(1.f);
    check(nearly(one.integrate(), 305.f, 1e-3f), "Spectrum(1).integrate() == 305");
}

void test_arithmetic()
{
    Spectrum a(2.f), b(3.f);
    Spectrum c = a + b;
    check(nearly(c[0], 5.f, 1e-6f) && nearly(c[60], 5.f, 1e-6f), "a+b componentwise");

    Spectrum d = a * b;
    check(nearly(d[0], 6.f, 1e-6f), "a*b componentwise");

    Spectrum e = 0.5f * a;
    check(nearly(e[0], 1.f, 1e-6f), "scalar*spectrum");

    Spectrum f = a - b;
    check(nearly(f[0], -1.f, 1e-6f), "a-b componentwise");
}

void test_clamp_non_negative()
{
    Spectrum s;
    for (int i = 0; i < Spectrum::kSamples; i++) s[i] = (i % 2 == 0) ? -1.f : 1.f;
    Spectrum c = s.clampNonNegative();
    bool ok = true;
    for (int i = 0; i < Spectrum::kSamples; i++) if (c[i] < 0.f) ok = false;
    check(ok, "clampNonNegative() removes negatives");
}

void test_from_sigmoid()
{
    // c0=large-positive, c1=c2=0: sigmoid saturates to 1 everywhere.
    Spectrum hi = Spectrum::fromSigmoidCoefficients(50.f, 0.f, 0.f);
    bool allOne = true;
    for (int i = 0; i < Spectrum::kSamples; i++) if (!nearly(hi[i], 1.f, 1e-4f)) allOne = false;
    check(allOne, "fromSigmoidCoefficients(50) saturates near 1");

    // c0=large-negative: sigmoid floors near 0.
    Spectrum lo = Spectrum::fromSigmoidCoefficients(-50.f, 0.f, 0.f);
    bool allZero = true;
    for (int i = 0; i < Spectrum::kSamples; i++) if (!nearly(lo[i], 0.f, 1e-4f)) allZero = false;
    check(allZero, "fromSigmoidCoefficients(-50) floors near 0");

    // c0=0: sigmoid(0) == 0.5 everywhere.
    Spectrum mid = Spectrum::fromSigmoidCoefficients(0.f, 0.f, 0.f);
    check(nearly(mid[0], 0.5f, 1e-6f) && nearly(mid[30], 0.5f, 1e-6f),
          "fromSigmoidCoefficients(0) == 0.5 everywhere");
}

} // namespace

int main()
{
    test_constants();
    test_default_zero();
    test_constant_fill();
    test_interpolation();
    test_out_of_range();
    test_integrate();
    test_arithmetic();
    test_clamp_non_negative();
    test_from_sigmoid();

    if (g_failed) {
        std::printf("\n%d FAIL(s)\n", g_failed);
        return 1;
    }
    std::printf("\nAll spectrum tests passed.\n");
    return 0;
}
