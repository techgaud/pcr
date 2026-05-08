#pragma once

#include "RGBToSpectrum.h"
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

    // Cauchy dispersion coefficient. ior_at_lambda(nm) = ior +
    // cauchyB * 1e4 / lambda^2, so a typical crown glass (B ~ 0.013
    // for the standard sellmeier-to-cauchy reduction) shifts ior by
    // ~0.04 across the visible range and produces a visible
    // rainbow when light refracts through glass at oblique angles.
    // Default 0 = no dispersion = renders identical to before.
    //
    // Only meaningful in spectral mode, where the path tracer
    // splits into per-channel sub-paths at glass surfaces (the 4
    // hero wavelengths see different IORs and refract at different
    // angles, producing real visible spectral separation). Ignored
    // in RGB mode since RGB has no wavelength notion.
    float cauchyB = 0.0f;

    // Helper. Wavelength-dependent IOR via the Cauchy relation,
    // truncated to the leading two terms. lambda is in nm.
    float iorAtLambda(float lambda) const
    {
        return ior + cauchyB * 1e4f / (lambda * lambda);
    }

    // Spectral fit. Populated at scene-load by populateSpectra() once
    // the RGB values are finalized. The path tracer's spectral mode
    // samples per-wavelength via albedoAt / emissiveAt; the RGB path
    // ignores these. 16 bytes per spectrum (vs 244 if we cached the
    // 61-sample table); per-lookup evaluates one sqrt and a few
    // multiply-adds inline. Same form the GpuMaterial uploads.
    RGBToSpectrum::SigmoidFit albedoFit;
    RGBToSpectrum::SigmoidFit emissiveFit;

    // Per-bounce spectral accessors. Hot path; inlined.
    float albedoAt(float lambda) const
    {
        return RGBToSpectrum::evalSigmoidFit(albedoFit, lambda);
    }
    float emissiveAt(float lambda) const
    {
        return RGBToSpectrum::evalSigmoidFit(emissiveFit, lambda);
    }

    bool isEmissive() const
    {
        return emissive[0] > 0 || emissive[1] > 0 || emissive[2] > 0;
    }

    // Specular materials skip the diffuse direct-lighting and indirect-
    // hemisphere sampling paths; they reflect/refract along a
    // deterministic (or Fresnel-weighted random) direction instead.
    bool isSpecular() const { return metallic || transparent; }

    // Fit albedo and emissive into Jakob 2019 sigmoid coefficients.
    // Idempotent. Emissive RGB is normalized before fit (the absolute
    // brightness is restored via the SigmoidFit's scale field) because
    // the upsampler expects values in [0, 1] and area lights routinely
    // emit much brighter than that.
    void populateSpectra()
    {
        albedoFit = RGBToSpectrum::fitSigmoidCoefficients(albedo);
        if (isEmissive())
        {
            float maxE = std::max({emissive[0], emissive[1], emissive[2]});
            Vec3f normalized(emissive[0] / maxE, emissive[1] / maxE, emissive[2] / maxE);
            emissiveFit = RGBToSpectrum::fitSigmoidCoefficients(normalized);
            emissiveFit.scale *= maxE;
        }
        else
        {
            emissiveFit = {0.f, 0.f, 0.f, 0.f};
        }
    }
};