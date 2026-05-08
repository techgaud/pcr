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

    void fitCoefficients(const Vec3f &targetXYZ,
                         float &outC0, float &outC1, float &outC2)
    {
        constexpr float kLambdaMid  = 0.5f * (Spectrum::kLambdaMin + Spectrum::kLambdaMax);
        constexpr float kLambdaHalf = 0.5f * (Spectrum::kLambdaMax - Spectrum::kLambdaMin);
        constexpr int   kMaxIter    = 15;
        constexpr float kTolerance  = 1e-6f;

        float c[3] = {0.f, 0.f, 0.f};

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

        outC0 = c[0]; outC1 = c[1]; outC2 = c[2];
    }

    Spectrum fitSpectrum(const Vec3f &rgbLinear)
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
