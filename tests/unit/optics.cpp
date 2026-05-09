// Optics unit tests.
//
// Covers the dielectric helpers extracted into Optics.h on commit 8616460:
// schlickFresnel, cauchyIor, dielectricBounce. These were duplicated 7
// times across CPU + GLSL before the refactor; the canonical CPU forms
// live here.
//
// The cauchyIor cases double as a dispersion-presence safety net: if a
// future change accidentally zeroes cauchyB or hard-codes the IOR, the
// "different IOR at 400 vs 700 nm" assertion catches it loudly. The
// render-test golden for cornell-glass spectral catches the visual side.

#include <cmath>
#include <cstdio>

#include "Optics.h"
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

void test_schlick_normal_glass()
{
    // Air-to-glass at normal incidence: F0 = ((1 - 1.5)/(1 + 1.5))^2 = 0.04.
    float f = Optics::schlickFresnel(1.f, 1.f, 1.5f);
    check(nearly(f, 0.04f, 1e-4f), "schlickFresnel(cos=1, 1.0, 1.5) ~= 0.04");
}

void test_schlick_normal_same_index()
{
    // Same medium on both sides at normal incidence: no reflection.
    // Note Schlick does NOT recover the 0 at grazing when F0 == 0
    // (the (1 - cosTheta)^5 term dominates), which is a known limit
    // of the approximation; we don't exercise that case.
    float fNormal = Optics::schlickFresnel(1.f, 1.5f, 1.5f);
    check(nearly(fNormal, 0.f, 1e-6f), "schlickFresnel(cos=1, n, n) == 0");
}

void test_schlick_grazing_glass()
{
    // Air-to-glass at grazing: Schlick approaches 1.
    float f = Optics::schlickFresnel(0.f, 1.f, 1.5f);
    check(nearly(f, 1.f, 1e-3f), "schlickFresnel(cos=0, 1.0, 1.5) ~= 1");
}

void test_cauchy_d_line_baseline()
{
    // d-line is 587.6 nm; for crown glass with B=0.013 and base 1.5,
    // ior(587.6) ~= 1.5 + 0.013 * 1e4 / 587.6^2 ~= 1.5 + 0.0376 ~= 1.5376.
    // The "1.50" in the plan refers to the BASE; the function adds
    // dispersion on top. Test the formula explicitly.
    float n = Optics::cauchyIor(1.5f, 0.013f, 587.6f);
    float expected = 1.5f + 0.013f * 1e4f / (587.6f * 587.6f);
    check(nearly(n, expected, 1e-6f), "cauchyIor(1.5, 0.013, 587.6) matches formula");
}

void test_cauchy_no_dispersion()
{
    // B=0 means no wavelength dependence. Every lambda returns base IOR.
    float n400 = Optics::cauchyIor(1.5f, 0.f, 400.f);
    float n700 = Optics::cauchyIor(1.5f, 0.f, 700.f);
    check(nearly(n400, 1.5f, 1e-6f), "cauchyIor(1.5, 0, 400) == 1.5");
    check(nearly(n700, 1.5f, 1e-6f), "cauchyIor(1.5, 0, 700) == 1.5");
}

void test_cauchy_dispersion_present()
{
    // Crown-glass cauchyB=0.013 should produce a meaningfully different
    // IOR at 400 nm vs 700 nm. If a future change accidentally zeroes
    // cauchyB or hard-codes IOR, the rainbow caustic in cornell-glass
    // disappears - this assertion is the loud-and-fast warning. The
    // render-test golden for cornell-glass spectral is the slow-and-
    // visual safety net behind it.
    float n400 = Optics::cauchyIor(1.5f, 0.013f, 400.f);
    float n700 = Optics::cauchyIor(1.5f, 0.013f, 700.f);
    float spread = n400 - n700;
    // Expected ~= 130 * (1/160000 - 1/490000) ~= 130 * 4.21e-6 ~= 5.47e-4.
    // Small in absolute terms but produces visible chromatic separation
    // through angular deflection in glass.
    check(spread > 3e-4f && spread < 1e-3f,
          "cauchyIor: 400 vs 700 nm spread in [3e-4, 1e-3] for B=0.013");
    // 400 nm IOR > 700 nm IOR (blue refracts more than red).
    check(n400 > n700, "blue IOR > red IOR for normal dispersion");
}

void test_dielectric_tir()
{
    // Total internal reflection: ray inside denser medium hitting the
    // boundary at a shallow angle should reflect, not refract.
    // Setup: glass-to-air at angle past critical (~41.8 deg for n=1.5),
    // so cosI < cos(41.8) ~= 0.745. Use cosI = 0.5 (60 deg from normal).
    // entering=false means n1=ior, n2=1.0.
    Vec3f normal(0.f, 1.f, 0.f);
    Vec3f rayDir(std::sqrt(0.75f), -0.5f, 0.f); // angle 60 from -N
    rayDir = rayDir.normalize();
    Vec3f hit(0.f, 0.f, 0.f);
    auto out = Optics::dielectricBounce(rayDir, normal, hit,
                                        /*entering=*/false, 1.5f,
                                        /*fresnelRand=*/0.99f);
    // Reflected ray: y component flips sign (rayDir.y was -0.5; now ~+0.5
    // since reflection adds 2*cosI*N where cosI = 0.5).
    check(out.dir[1] > 0.f, "TIR reflects (y component flips up)");
    // Origin is shifted along +N to escape self-intersection.
    check(out.origin[1] > 0.f, "TIR origin shifts along +N");
}

void test_dielectric_eta_one_passes_through()
{
    // n1 = n2 = 1: no bend, ray goes straight. Use entering=true with ior=1
    // so eta = 1/1 = 1.
    Vec3f normal(0.f, 1.f, 0.f);
    Vec3f rayDir(0.f, -1.f, 0.f); // straight down
    Vec3f hit(0.f, 0.f, 0.f);
    auto out = Optics::dielectricBounce(rayDir, normal, hit,
                                        /*entering=*/true, 1.0f,
                                        /*fresnelRand=*/0.99f); // skip reflect
    check(nearly(out.dir[0], 0.f, 1e-5f) && nearly(out.dir[1], -1.f, 1e-5f),
          "eta=1 refraction passes straight through");
}

void test_dielectric_reflect_branch()
{
    // Below critical angle, fresnelRand=0 forces reflection. Verify the
    // outgoing direction matches the mirror reflection.
    Vec3f normal(0.f, 1.f, 0.f);
    Vec3f rayDir(0.6f, -0.8f, 0.f); // hits at ~37 deg from normal
    rayDir = rayDir.normalize();
    Vec3f hit(0.f, 0.f, 0.f);
    auto out = Optics::dielectricBounce(rayDir, normal, hit,
                                        /*entering=*/true, 1.5f,
                                        /*fresnelRand=*/0.f); // always reflect
    // Mirror reflection: dir = rayDir + 2*cosI*N. cosI = -dot(rayDir, N) = 0.8.
    // Expected: rayDir + 2*0.8*N = (0.6, -0.8, 0) + (0, 1.6, 0) = (0.6, 0.8, 0).
    check(nearly(out.dir[0], 0.6f, 1e-4f) && nearly(out.dir[1], 0.8f, 1e-4f),
          "fresnelRand=0 produces mirror reflection");
}

} // namespace

int main()
{
    test_schlick_normal_glass();
    test_schlick_normal_same_index();
    test_schlick_grazing_glass();
    test_cauchy_d_line_baseline();
    test_cauchy_no_dispersion();
    test_cauchy_dispersion_present();
    test_dielectric_tir();
    test_dielectric_eta_one_passes_through();
    test_dielectric_reflect_branch();

    if (g_failed) {
        std::printf("\n%d FAIL(s)\n", g_failed);
        return 1;
    }
    std::printf("\nAll optics tests passed.\n");
    return 0;
}
