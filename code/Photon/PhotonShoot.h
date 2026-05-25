#pragma once

#include <cstdint>

#include "PhotonMap.h"
#include "../Scenes/Scene.h"

// Caustic photon shooting. Shoots `photonCount` photons from the
// scene's area lights, tracing each through specular (mirror / glass)
// interactions until it lands on a diffuse surface, then deposits a
// photon record at that point. Photons that never touch a specular
// surface are dropped (caustic-only: eye-path tracing already handles
// diffuse-only light transport via NEE and indirect bounces).
//
// Runs single-threaded by design. The photon-shoot phase is small
// relative to the eye-path render (~milliseconds for 1M photons on
// a Cornell-scale BVH), and single-threaded keeps the implementation
// trivial. A future Metal port will move the trace loop onto the GPU.

namespace Photon
{
    // Shoot photons + build map. `radius` is the kernel radius the
    // map will be queried with (also doubles as hash-grid cell size).
    // `maxBounces` caps the specular chain length to avoid unbounded
    // ping-pong inside a TIR-prone glass sphere; Russian roulette
    // also terminates paths probabilistically once the carried power
    // is small.
    //
    // `seed` controls determinism. 0 = random_device-seeded. Non-zero
    // = deterministic; useful for golden-image diffs across renders.
    // The seed is local to this function and does NOT advance the
    // global NumGen PRNG state, so the same --seed produces the same
    // eye-path render whether --photon-map is on or off.
    Map shootCaustic(const Scenes::SceneData &scene,
                     int photonCount,
                     float radius,
                     int maxBounces,
                     uint64_t seed);
}
