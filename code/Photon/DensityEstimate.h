#pragma once

#include <numbers>

#include "../Includes/Vec3f.h"
#include "PhotonMap.h"

// Caustic-map density estimate. Header-only so both the CPU renderer
// and the eventual GPU ports (which will inline-translate the body
// into MSL / GLSL) work from a single algorithmic source of truth.
//
// Mathematical form (Jensen 1996 / PBRT v4 ch. 16):
//
//   L_o(x, w_o) ~= (1 / (pi r^2)) * sum_p [ f_r(x, w_o, w_p) * Phi_p * k(d_p) ]
//
// where the sum runs over photons p within radius r of x, f_r is the
// surface BRDF evaluated at (incoming photon direction w_p, outgoing
// view direction w_o), Phi_p is the photon power, d_p is the
// photon-to-x distance, and k is the kernel weight (1 / (1 - 2/(3k))
// for the standard Jensen cone with kernel constant k; we use the
// simple box kernel k(d)=1 for now and bake the normalization into
// the (1 / pi r^2) prefactor).
//
// For a Lambertian diffuse BSDF f_r = albedo / pi, the per-photon
// contribution simplifies to:
//
//   delta_L = (albedo / pi) * Phi_p * (1 / (pi r^2))
//
// We factor (albedo / pi) out of the loop and apply it once at the
// end. The remaining sum is just sum_p Phi_p over in-radius photons,
// normalized by pi r^2. albedo is RGB-vector-multiplied with the sum
// to color-tint the caustic according to the material it landed on.

namespace Photon
{
    // Estimate the caustic-radiance contribution at a diffuse hit
    // point `x` with surface normal `N` and Lambertian reflectance
    // `albedo`. Returns the radiance to add directly to the eye-path
    // accumulator at this hit.
    //
    // Photons whose direction-of-travel `wi` does not strike the
    // surface from the front (wi.N >= 0) are dropped: they represent
    // either back-side hits or photons that walked tangent to the
    // surface, neither of which contributes to outgoing radiance on
    // this side.
    //
    // When the map is empty, returns zero with no cost beyond the
    // size check. Important because the call site adds this
    // unconditionally on every diffuse hit in the photon-map render
    // path; cheap exit keeps the non-photon-map code path's overhead
    // at "one branch, one load."
    inline Vec3f densityEstimate(const Map &map,
                                 const Vec3f &x,
                                 const Vec3f &N,
                                 const Vec3f &albedo)
    {
        if (map.size() == 0) return Vec3f(0.f, 0.f, 0.f);

        Vec3f sumPower(0.f, 0.f, 0.f);
        map.query(x, [&](const Record &p, float /*distSq*/) {
            // Reject photons hitting the back side of this surface.
            // wi points along travel direction; for a photon to
            // illuminate this side, wi.N must be negative.
            if (p.wi.dot(N) >= 0.f) return;
            sumPower[0] += p.power[0];
            sumPower[1] += p.power[1];
            sumPower[2] += p.power[2];
        });

        const float r = map.radius();
        const float invArea = 1.0f / ((float)std::numbers::pi * r * r);
        const float invPi   = 1.0f / (float)std::numbers::pi;

        // f_r (Lambert) = albedo / pi; density = sum_power / (pi r^2).
        return Vec3f(
            albedo[0] * invPi * sumPower[0] * invArea,
            albedo[1] * invPi * sumPower[1] * invArea,
            albedo[2] * invPi * sumPower[2] * invArea);
    }
}
