#include "Includes/RGBToSpectrum.h"

#include <algorithm>
#include <cmath>

#include "Includes/CIE.h"
#include "Includes/Spectrum.h"
#include "Includes/Vec3f.h"

// Heavy functions moved out of RGBToSpectrum.h into this translation
// unit. Inline definitions in headers were causing MSVC to crash
// during link-time codegen ("Generating Code..." then CL.exe exit
// code -529706956, an internal compiler exception) because the
// Newton-Raphson solver gets re-instantiated in every TU that
// includes Material.h. One copy here, one .obj file per build.
//
// The light helpers (sigmoid, sigmoidDerivative) stay inline in
// the header since they're trivial.

namespace RGBToSpectrum
{
    bool solve3x3(const float J[3][3], const float b[3], float x[3])
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

    // Newton-Raphson core. Refines a starting (cIn[0], cIn[1], cIn[2]) toward
    // sigmoid-polynomial coefficients whose integrated XYZ matches targetXYZ.
    // Writes the converged coefficients to cOut. Used both by the runtime
    // homotopy in fitCoefficients and by the LUT builder in buildLUT.
    void newtonFit(const Vec3f &targetXYZ, const float cIn[3], float cOut[3])
    {
        constexpr float kLambdaMid  = 0.5f * (Spectrum::kLambdaMin + Spectrum::kLambdaMax);
        constexpr float kLambdaHalf = 0.5f * (Spectrum::kLambdaMax - Spectrum::kLambdaMin);
        constexpr int   kMaxIter    = 15;
        constexpr float kTolerance  = 1e-6f;

        float c[3] = {cIn[0], cIn[1], cIn[2]};

        for (int iter = 0; iter < kMaxIter; iter++)
        {
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

            float prevC[3] = {c[0], c[1], c[2]};
            float step = 1.f;
            bool reduced = false;
            for (int trial = 0; trial < 4; trial++)
            {
                c[0] = prevC[0] - step * delta[0];
                c[1] = prevC[1] - step * delta[1];
                c[2] = prevC[2] - step * delta[2];

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
                if (trialNorm < rNorm) { reduced = true; break; }
                step *= 0.5f;
            }
            if (!reduced)
            {
                // No step in [1, 1/2, 1/4, 1/8] reduced residual. Revert
                // to prevC and bail; further iterations from a worsened
                // c just diverge (this is the failure mode that kept the
                // LUT warm-start chain pinned at the coefficient clamp).
                c[0] = prevC[0]; c[1] = prevC[1]; c[2] = prevC[2];
                break;
            }
        }

        cOut[0] = c[0]; cOut[1] = c[1]; cOut[2] = c[2];
    }

    // Spike detector: max sample value over median sample value across the
    // 61-sample reflectance. A smooth physical reflectance has max/median in
    // the 1-10x range. The pathological spike-of-near-zero spectra Newton
    // can converge to for gamut-edge sRGB colors have max/median in the
    // hundreds or thousands, which is what kills Monte Carlo convergence in
    // the path tracer (most rays sample wavelengths off the spike, return
    // ~zero throughput, walls render black even after many samples).
    static float spikiness(const float c[3])
    {
        Spectrum s = Spectrum::fromSigmoidCoefficients(c[0], c[1], c[2]);
        float samples[Spectrum::kSamples];
        float maxV = 0.f;
        for (int i = 0; i < Spectrum::kSamples; i++)
        {
            samples[i] = s[i];
            if (samples[i] > maxV) maxV = samples[i];
        }
        std::nth_element(samples, samples + Spectrum::kSamples / 2,
                         samples + Spectrum::kSamples);
        float median = samples[Spectrum::kSamples / 2];
        if (median < 1e-8f) return (maxV < 1e-8f) ? 1.f : 1e30f;
        return maxV / median;
    }

    void fitCoefficients(const Vec3f &targetXYZ,
                         float &outC0, float &outC1, float &outC2)
    {
        // Cold-start Newton-Raphson on the Jakob 2019 sigmoid-polynomial
        // objective has multiple local minima for saturated chromatic
        // targets: a smooth physically-plausible reflectance and a tall
        // narrow spike-of-zero that integrates to the same XYZ but
        // renders as black under Monte Carlo path tracing. From any cold
        // initial guess, Newton's quickest descent path is into the
        // spike basin.
        //
        // The fix Jakob's reference implementation (mitsuba-renderer/
        // rgb2spec) uses is HOMOTOPY/CONTINUATION: solve an easy
        // desaturated problem first (where only the smooth basin
        // exists), then walk the target along a chromaticity ramp toward
        // the real RGB, warm-starting Newton from the previous solution
        // each step. The smooth basin tracks continuously; Newton can't
        // jump into the spike basin because each tiny step keeps it
        // close to where it already was.
        //
        // Easy starting target: gray with the same Y as the real target.
        // The sRGB->XYZ matrix row sums are exactly the D65 white point
        // (Xn = 0.95046, Yn = 1.0, Zn = 1.08906), so for RGB = (Y, Y, Y)
        // the corresponding XYZ is (Y*Xn, Y, Y*Zn).
        constexpr float kXn = 0.95046f;
        constexpr float kZn = 1.08906f;
        Vec3f xyzGray(targetXYZ[1] * kXn, targetXYZ[1], targetXYZ[1] * kZn);

        // Initial guess: flat spectrum integrating to target luminance.
        // For gray targets that's already the answer; the homotopy loop
        // below converges in one Newton step. For chromatic targets the
        // flat fit gets refined in many small steps as we walk the
        // target toward its real chromaticity.
        float c[3] = {sigmoidInverse(targetXYZ[1] / CIE::yBarIntegral()), 0.f, 0.f};

        // Walk the chromaticity ramp from gray to the real target. Save the
        // last "good" (smooth) coefficients along the way; if the homotopy
        // path eventually crosses into the spike basin (which it does for
        // sRGB gamut-edge colors that have no smooth representation in the
        // sigmoid-of-quadratic family), we return the last smooth fit
        // rather than the spiky exact-XYZ fit. The visual cost is mild
        // desaturation on highly chromatic surfaces; the alternative is
        // those surfaces rendering black under Monte Carlo path tracing.
        constexpr int kHomotopySteps = 32;
        constexpr float kSpikeThreshold = 200.f;
        float lastGoodC[3] = {c[0], c[1], c[2]};
        for (int step = 1; step <= kHomotopySteps; step++)
        {
            float t = (float)step / (float)kHomotopySteps;
            Vec3f xyzCurr(
                (1.f - t) * xyzGray[0] + t * targetXYZ[0],
                (1.f - t) * xyzGray[1] + t * targetXYZ[1],
                (1.f - t) * xyzGray[2] + t * targetXYZ[2]
            );
            float cNext[3];
            newtonFit(xyzCurr, c, cNext);
            if (spikiness(cNext) > kSpikeThreshold) break;
            c[0] = cNext[0]; c[1] = cNext[1]; c[2] = cNext[2];
            lastGoodC[0] = c[0]; lastGoodC[1] = c[1]; lastGoodC[2] = c[2];
        }

        outC0 = lastGoodC[0]; outC1 = lastGoodC[1]; outC2 = lastGoodC[2];
    }

    SigmoidFit fitSigmoidCoefficients(const Vec3f &rgbLinear)
    {
        // If --lut is in effect, dispatch to the precomputed table instead
        // of running the runtime homotopy. The LUT was built once at
        // startup with proper warm-starting through the gamut, which
        // handles saturated chromaticities the per-call solver can't.
        if (const LUT *lut = activeLUT())
            return lookupSigmoidFit(*lut, rgbLinear);

        // Clamp inputs to physical reflectance range. RGB > 1 is unphysical
        // for albedo (would gain energy at a wavelength). Emissive callers
        // (Material::populateSpectra) already normalize to <= 1 before
        // calling here and re-multiply by maxE afterward, so HDR emission
        // is unaffected.
        float r = std::clamp(rgbLinear[0], 0.f, 1.f);
        float g = std::clamp(rgbLinear[1], 0.f, 1.f);
        float b = std::clamp(rgbLinear[2], 0.f, 1.f);
        if (r <= 0.f && g <= 0.f && b <= 0.f)
        {
            // All-zero. Any (c0, c1, c2) work since scale=0 zeroes the
            // result; pick a flat polynomial so debug prints look sane.
            return {0.f, 0.f, 0.f, 0.f};
        }
        if (r >= 1.f && g >= 1.f && b >= 1.f)
        {
            // Perfect white. Choose c0 such that sigmoid(c0) * yBarIntegral
            // = 1 exactly, c1 = c2 = 0 (flat polynomial). evalSigmoidFit's
            // clamp would also catch a slight overshoot, but solving for c0
            // exactly is cheap and keeps the sample equal to 1.0.
            float scale = CIE::yBarIntegral();
            return {sigmoidInverse(1.f / scale), 0.f, 0.f, scale};
        }

        Vec3f xyz = CIE::linearSRGBToXYZ(Vec3f(r, g, b));
        float c0, c1, c2;
        fitCoefficients(xyz, c0, c1, c2);
        return {c0, c1, c2, CIE::yBarIntegral()};
    }

    Spectrum fitSpectrum(const Vec3f &rgbLinear)
    {
        // Convenience wrapper that builds a 61-sample Spectrum from the
        // SigmoidFit by evaluating at every stored wavelength. Caller
        // semantics are "spectrum samples = physical reflectance," so we
        // clamp to [0, 1] here even though evalSigmoidFit no longer does.
        // Used by the probe and by any external code path that wants a
        // dense spectrum; per-bounce lookups don't go through this -
        // they call evalSigmoidFit directly via Material::albedoAt /
        // emissiveAt.
        SigmoidFit fit = fitSigmoidCoefficients(rgbLinear);
        Spectrum s;
        for (int i = 0; i < Spectrum::kSamples; i++)
            s[i] = std::min(evalSigmoidFit(fit, Spectrum::lambdaAt(i)), 1.f);
        return s;
    }

    namespace { const LUT *g_activeLUT = nullptr; }
    void setActiveLUT(const LUT *lut) { g_activeLUT = lut; }
    const LUT *activeLUT() { return g_activeLUT; }

    // Reassemble (R, G, B) given which channel is the "max" axis and the
    // brightness-and-chromaticity coordinates the LUT indexes by:
    //   maxChannel = which channel holds the brightness on this slab
    //   b = brightness on max channel, in [0, 1]
    //   x = (other-channel-1) / b, in [0, 1] for valid sRGB
    //   y = (other-channel-2) / b
    static Vec3f composeRGB(int maxChannel, float x, float y, float b)
    {
        float v0 = b;
        float v1 = x * b;
        float v2 = y * b;
        if (maxChannel == 0) return Vec3f(v0, v1, v2);
        if (maxChannel == 1) return Vec3f(v1, v0, v2);
        return Vec3f(v1, v2, v0);
    }

    void buildLUT(LUT &lut)
    {
        // For each max-channel axis, build a kRes^3 grid in (b, x, y).
        // Walk b-sweep outward from a moderate starting brightness so each
        // cell warm-starts from a less-saturated neighbor. The starting
        // point is "kRes/5" steps in - far enough from the gamut corner
        // that the smooth basin is well-defined, close enough that two
        // sweeps cover the whole grid.
        constexpr int kRes = LUT::kRes;
        constexpr int kStart = kRes / 5;

        for (int l = 0; l < 3; l++)
        {
            for (int xi = 0; xi < kRes; xi++)
            {
                for (int yi = 0; yi < kRes; yi++)
                {
                    float x = float(xi) / float(kRes - 1);
                    float y = float(yi) / float(kRes - 1);

                    // Each cell runs the full homotopy to produce its fit.
                    // In principle a warm-start chain through brightness
                    // would be faster (mitsuba's approach with Gauss-
                    // Newton + finite-difference Jacobian); our analytical
                    // Newton-Raphson hits rank-deficient Jacobians when
                    // the spectrum's sigmoid is mostly saturated near 0
                    // and the chain falls into the coefficient clamp.
                    // Paying the homotopy cost per cell keeps the LUT
                    // honest at the cost of seconds at startup.
                    for (int bi = 0; bi < kRes; bi++)
                    {
                        int idx = lut.linearIdx(l, bi, xi, yi);
                        float b = float(bi) / float(kRes - 1);
                        if (b <= 0.f)
                        {
                            lut.data[idx + 0] = 0.f;
                            lut.data[idx + 1] = 0.f;
                            lut.data[idx + 2] = 0.f;
                            continue;
                        }
                        Vec3f rgb = composeRGB(l, x, y, b);
                        Vec3f xyz = CIE::linearSRGBToXYZ(rgb);
                        float c0, c1, c2;
                        fitCoefficients(xyz, c0, c1, c2);
                        constexpr float kCoeffMax = 200.f;
                        lut.data[idx + 0] = std::clamp(c0, -kCoeffMax, kCoeffMax);
                        lut.data[idx + 1] = std::clamp(c1, -kCoeffMax, kCoeffMax);
                        lut.data[idx + 2] = std::clamp(c2, -kCoeffMax, kCoeffMax);
                    }
                    (void)kStart; // unused now; kept for the comment context
                }
            }
        }
    }

    SigmoidFit lookupSigmoidFit(const LUT &lut, const Vec3f &rgbLinear)
    {
        float r = std::clamp(rgbLinear[0], 0.f, 1.f);
        float g = std::clamp(rgbLinear[1], 0.f, 1.f);
        float b = std::clamp(rgbLinear[2], 0.f, 1.f);
        if (r <= 0.f && g <= 0.f && b <= 0.f) return {0.f, 0.f, 0.f, 0.f};
        if (r >= 1.f && g >= 1.f && b >= 1.f)
        {
            float scale = CIE::yBarIntegral();
            return {sigmoidInverse(1.f / scale), 0.f, 0.f, scale};
        }

        // Pick the max channel, pull out the corresponding axis grid.
        int l = 0;
        float maxV = r;
        if (g >= maxV) { l = 1; maxV = g; }
        if (b >= maxV) { l = 2; maxV = b; }

        float other1, other2;
        if (l == 0) { other1 = g; other2 = b; }
        else if (l == 1) { other1 = r; other2 = b; }
        else            { other1 = r; other2 = g; }

        float xn = other1 / maxV;
        float yn = other2 / maxV;
        float bn = maxV;

        // Snap to grid indices and fractional parts. Clamp the integer
        // parts to [0, kRes - 2] so the +1 below stays in-bounds.
        constexpr int kRes = LUT::kRes;
        float xf = xn * float(kRes - 1);
        float yf = yn * float(kRes - 1);
        float bf = bn * float(kRes - 1);
        int xi = std::clamp(int(xf), 0, kRes - 2);
        int yi = std::clamp(int(yf), 0, kRes - 2);
        int bi = std::clamp(int(bf), 0, kRes - 2);
        float xt = std::clamp(xf - xi, 0.f, 1.f);
        float yt = std::clamp(yf - yi, 0.f, 1.f);
        float bt = std::clamp(bf - bi, 0.f, 1.f);

        // 8-corner trilinear interpolation in (b, x, y) space.
        SigmoidFit fit;
        fit.scale = CIE::yBarIntegral();
        for (int k = 0; k < 3; k++)
        {
            auto sample = [&](int db, int dx, int dy) {
                return lut.data[lut.linearIdx(l, bi + db, xi + dx, yi + dy) + k];
            };
            float v000 = sample(0, 0, 0);
            float v100 = sample(0, 1, 0);
            float v010 = sample(0, 0, 1);
            float v110 = sample(0, 1, 1);
            float v001 = sample(1, 0, 0);
            float v101 = sample(1, 1, 0);
            float v011 = sample(1, 0, 1);
            float v111 = sample(1, 1, 1);
            float v00 = v000 + xt * (v100 - v000);
            float v10 = v010 + xt * (v110 - v010);
            float v01 = v001 + xt * (v101 - v001);
            float v11 = v011 + xt * (v111 - v011);
            float v0  = v00  + yt * (v10  - v00);
            float v1  = v01  + yt * (v11  - v01);
            float v   = v0   + bt * (v1   - v0);
            if (k == 0) fit.c0 = v;
            else if (k == 1) fit.c1 = v;
            else             fit.c2 = v;
        }
        return fit;
    }
}
