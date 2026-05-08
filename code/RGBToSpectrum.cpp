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
    // Writes the converged coefficients to cOut.
    static void newtonFit(const Vec3f &targetXYZ, const float cIn[3], float cOut[3])
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
                if (trialNorm < rNorm) break;
                step *= 0.5f;
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

    Spectrum fitSpectrum(const Vec3f &rgbLinear)
    {
        float r = rgbLinear[0], g = rgbLinear[1], b = rgbLinear[2];
        if (r <= 0.f && g <= 0.f && b <= 0.f) return Spectrum(0.f);
        if (r >= 1.f && g >= 1.f && b >= 1.f) return Spectrum(1.f);

        Vec3f xyz = CIE::linearSRGBToXYZ(rgbLinear);
        float c0, c1, c2;
        fitCoefficients(xyz, c0, c1, c2);
        Spectrum s = Spectrum::fromSigmoidCoefficients(c0, c1, c2);
        // Convert from "fit-to-XYZ" to physical reflectance convention so
        // multi-bounce path-tracer throughput attenuates the same way RGB
        // albedos do. See CIE::yBarIntegral() for the full discussion.
        s *= CIE::yBarIntegral();
        return s;
    }
}
