#pragma once

#include "RGBToSpectrum.h"
#include "Spectrum.h"
#include "Vec3f.h"

struct Material
{
    Material() {}
    Material(Vec3f &&a) : albedo{a}, emissive{0, 0, 0} {}
    Material(Vec3f &&a, Vec3f &&e) : albedo{a}, emissive{e} {}

    Vec3f albedo{0, 0, 0};
    Vec3f emissive{0, 0, 0};

    // Perfect mirror (specular reflection only, no diffuse). albedo tints
    // the reflected radiance. set to (1,1,1) for a neutral mirror, or
    // (0.95, 0.65, 0.3) for a tinted gold-ish mirror.
    bool metallic = false;

    // Transparent dielectric (glass). Fresnel-weighted reflection +
    // refraction via Snell's law. ior = index of refraction (1.5 = glass,
    // 1.33 = water, 2.4 = diamond). albedo tints both reflected and
    // refracted contributions.
    bool transparent = false;
    float ior = 1.5f;

    // Spectral counterparts of albedo and emissive. Populated at scene-
    // load time by populateSpectra() (called via SceneData) once the
    // RGB values are finalized. The path tracer's spectral mode reads
    // these directly; the RGB path ignores them.
    //
    // Stored on the Material rather than computed on demand because a
    // Newton-Raphson fit is ~10 microseconds per material and gets run
    // across every primitive in every dispatch otherwise. Once at
    // scene load is far cheaper than every ray.
    Spectrum albedoSpectrum;
    Spectrum emissiveSpectrum;

    bool isEmissive() const
    {
        return emissive[0] > 0 || emissive[1] > 0 || emissive[2] > 0;
    }

    // Specular materials skip the diffuse direct-lighting and indirect-
    // hemisphere sampling paths; they reflect/refract along a
    // deterministic (or Fresnel-weighted random) direction instead.
    bool isSpecular() const { return metallic || transparent; }

    // Fit albedo and emissive into 61-sample spectra via Jakob 2019
    // sigmoid upsampling. Idempotent. Emissive RGB is normalized
    // before fit (the absolute brightness is restored after) because
    // the upsampler expects values in [0, 1] and area lights
    // routinely have emission much brighter than that.
    void populateSpectra()
    {
        albedoSpectrum = RGBToSpectrum::fitSpectrum(albedo);
        if (isEmissive())
        {
            float maxE = std::max({emissive[0], emissive[1], emissive[2]});
            Vec3f normalized(emissive[0] / maxE, emissive[1] / maxE, emissive[2] / maxE);
            emissiveSpectrum = RGBToSpectrum::fitSpectrum(normalized);
            emissiveSpectrum *= maxE;
        }
        else
        {
            emissiveSpectrum = Spectrum(0.f);
        }
    }
};