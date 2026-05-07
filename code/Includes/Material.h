#pragma once

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

    bool isEmissive() const
    {
        return emissive[0] > 0 || emissive[1] > 0 || emissive[2] > 0;
    }

    // Specular materials skip the diffuse direct-lighting and indirect-
    // hemisphere sampling paths; they reflect/refract along a
    // deterministic (or Fresnel-weighted random) direction instead.
    bool isSpecular() const { return metallic || transparent; }
};