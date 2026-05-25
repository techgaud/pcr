#pragma once

#include <cmath>
#include <cstdint>

// Stochastic Progressive Photon Mapping per-pixel state (Hachisuka &
// Jensen 2009). Each pixel maintains a "visible point" -- conceptually
// the first diffuse hit along its primary ray -- along with three
// running statistics that converge asymptotically to the unbiased
// caustic radiance estimate.
//
//   R : current search radius. Shrinks each progressive pass based on
//       the per-pixel photon density. As N -> infinity, R -> 0 and
//       the kernel-density bias -> 0, which is the asymptotic-
//       unbiasedness guarantee that distinguishes SPPM from plain
//       progressive (ensemble averaging) and from classical Jensen
//       photon mapping.
//
//   tau : accumulated flux (RGB). After each progressive pass, the
//         per-pass delta_tau (sum of BSDF * photon power for photons
//         within R at the visible point) gets folded in with the
//         shrinkage factor (N_new / (N + M)).
//
//   N : accumulated effective photon count. Grows by alpha * M each
//       pass; alpha = 2/3 is the Hachisuka recommendation, balances
//       fast initial convergence against the asymptotic guarantee.
//
// Storage layout is 20 bytes / pixel; at 8K = 1.3 GB, well within
// budget on the Mac Studio's unified memory. Both Metal (MTLBuffer)
// and OpenGL (SSBO) bind this layout directly; std430 / packed_float
// rules give the same in-memory representation as the host POD.
//
// Per-pass mutation buffer (delta_tau + M) lives separately so the
// per-pass parallel writes don't race with the per-pass-end update
// kernel that consumes them. After the update kernel runs, the
// delta buffer is zeroed for the next pass.

namespace Photon
{
    // Per-pixel state. Persists across progressive passes for the
    // lifetime of the SPPM render.
    struct SppmPixel
    {
        float R;        // current search radius (scene units)
        float tauR;
        float tauG;
        float tauB;
        float N;        // accumulated effective photon count
    };
    static_assert(sizeof(SppmPixel) == 20,
                  "Photon::SppmPixel must be 20 bytes (5x float) to match the "
                  "MSL + GLSL std430 SppmPixel layouts");

    // Per-pass scratch. Written by the per-diffuse-hit density-
    // estimate during the eye path; consumed (and zeroed) by the
    // end-of-pass update step.
    //
    // M is stored as a float not an int to sidestep needing a
    // separate atomic-uint buffer on GPU; on the GPU side we use the
    // existing atomic_add_float helper that's already wired up for
    // the perPixelAccum fork-mode buffer. The end-of-pass update
    // round-trips it through int when applying Hachisuka's recipe.
    struct SppmDelta
    {
        float dtauR;
        float dtauG;
        float dtauB;
        float M;
    };
    static_assert(sizeof(SppmDelta) == 16,
                  "Photon::SppmDelta must be 16 bytes (4x float)");

    // Hachisuka 2009 shrinkage parameter. 2/3 (the paper's
    // recommendation) trades off initial convergence speed against
    // asymptotic-unbiasedness; smaller alpha shrinks slower and
    // converges to less bias at the cost of more samples needed.
    inline constexpr float kSppmAlpha = 2.0f / 3.0f;

    // Apply the Hachisuka 2009 per-pixel update for a single pass.
    // Called per pixel after a progressive pass's diffuse-hit
    // density estimates have populated delta. On host this runs in
    // a CPU loop; on GPU it runs in an update compute kernel.
    //
    // Math (M >= 0; M == 0 leaves state unchanged):
    //   N_new = N + alpha * M
    //   R_new = R * sqrt(N_new / (N + M))
    //   tau_new = (tau + delta_tau) * (N_new / (N + M))
    inline void sppmUpdatePixel(SppmPixel &pix, const SppmDelta &delta)
    {
        float M = delta.M;
        if (M <= 0.0f) return;
        float N      = pix.N;
        float N_new  = N + kSppmAlpha * M;
        float shrink = N_new / (N + M);
        pix.R    = pix.R * std::sqrt(shrink);
        pix.tauR = (pix.tauR + delta.dtauR) * shrink;
        pix.tauG = (pix.tauG + delta.dtauG) * shrink;
        pix.tauB = (pix.tauB + delta.dtauB) * shrink;
        pix.N    = N_new;
    }
}
