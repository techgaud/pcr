#pragma once

#include <cmath>

#include "Vec3f.h"

// Dielectric (glass) optics. The math for "given an incoming ray hitting
// a smooth dielectric surface, what direction does it bounce / refract?"
// got reproduced four times in Renderer.cpp and three times in the GLSL
// path tracer string before this header existed. The CPU side now calls
// Optics::dielectricBounce; the GLSL side keeps a parallel definition in
// GpuRenderer.cpp's shader string that mirrors this API line-for-line.

namespace Optics
{
    // Schlick approximation to the Fresnel reflectance at the interface
    // between two media. cosTheta is the cosine of the angle between the
    // incident ray and the surface normal (i.e. -dot(rayDir, N) for a
    // ray hitting the surface). Returns the fraction of the incident
    // light that reflects rather than refracts; the renderer treats it
    // as a coin-flip probability between reflection and refraction.
    inline float schlickFresnel(float cosTheta, float n1, float n2)
    {
        float F0 = (n1 - n2) / (n1 + n2);
        F0 *= F0;
        return F0 + (1.f - F0) * std::pow(1.f - cosTheta, 5.f);
    }

    // Two-term Cauchy IOR. baseIor is the material's nominal IOR (the
    // value that would apply at infinite wavelength, conventionally
    // measured at the d-line, 587.6 nm); cauchyB is the per-material
    // dispersion coefficient. lambdaNm is wavelength in nanometers.
    // ior(lambda) = baseIor + cauchyB * 1e4 / lambda^2
    // The 1e4 factor compresses cauchyB into a number near 0.013 for
    // crown glass instead of 130 (a common point of confusion in
    // production codebases).
    inline float cauchyIor(float baseIor, float cauchyB, float lambdaNm)
    {
        return baseIor + cauchyB * 1e4f / (lambdaNm * lambdaNm);
    }

    struct DielectricOut
    {
        Vec3f dir;
        Vec3f origin;
    };

    // Fresnel-weighted glass bounce. Caller passes the incoming ray
    // direction, surface normal (oriented toward the incident side - so
    // dot(rayDir, N) < 0), the hit point, whether the ray is entering
    // or exiting the dielectric, the IOR to use (caller chooses base or
    // wavelength-dispersed), and one uniform random in [0, 1) for the
    // reflect/refract coin flip.
    //
    // Returns an outgoing direction and an offset origin (epsilon-shifted
    // along +/- N to escape self-intersection at the surface). Total
    // internal reflection branches go reflective unconditionally; below
    // the critical angle, the random is compared against the Schlick F.
    inline DielectricOut dielectricBounce(
        const Vec3f &rayDir, const Vec3f &N, const Vec3f &hit,
        bool entering, float ior, float fresnelRand)
    {
        float cosI = -rayDir.dot(N);
        float n1 = entering ? 1.f : ior;
        float n2 = entering ? ior : 1.f;
        float eta = n1 / n2;
        float sinT2 = eta * eta * (1.f - cosI * cosI);

        if (sinT2 >= 1.f)
        {
            // Total internal reflection.
            return { rayDir + N * (2.f * cosI), hit + N * 1e-3f };
        }

        float F = schlickFresnel(cosI, n1, n2);
        if (fresnelRand < F)
        {
            return { rayDir + N * (2.f * cosI), hit + N * 1e-3f };
        }
        float cosT = std::sqrt(1.f - sinT2);
        return { rayDir * eta + N * (eta * cosI - cosT), hit - N * 1e-3f };
    }
}
