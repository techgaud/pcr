#include "Includes/CIE.h"

#include "Includes/Spectrum.h"
#include "Includes/Vec3f.h"

// spectrumToXYZ is a 61-iteration loop calling Wyman's
// piecewise-Gaussian CMFs (each does 3 exp() calls), so inlining
// it everywhere generated enough code per translation unit to
// crash MSVC's batched codegen. Move it here; the trivially-small
// CMF helpers and the XYZ -> sRGB matrix multiply stay inline.

namespace CIE
{
    Vec3f spectrumToXYZ(const Spectrum &s)
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

    Vec3f spectrumToLinearSRGB(const Spectrum &s)
    {
        return xyzToLinearSRGB(spectrumToXYZ(s));
    }
}
