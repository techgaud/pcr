#include "Includes/NumGen.h"

#include <cstdint>
#include <random>

// PCG32 (O'Neill 2014) instead of mt19937. PCG is roughly 10x faster
// than std::mt19937 + std::uniform_real_distribution and has
// statistical quality good enough for Monte Carlo sampling. The GPU
// shader already uses PCG (see GpuRenderer.cpp); using it on CPU too
// keeps both backends drawing from the same family of generators,
// which makes any cross-backend rendering bug easier to chase.
//
// Each thread gets its own state seeded from std::random_device on
// first call; thread_local keeps the per-thread sequences
// independent without locking.
//
// State advances via a multiplicative-LCG step (Knuth's constant)
// then mixes via xor-shift to produce the output. Output range is a
// uint32, divided by the maximum to land in [0, 1).

namespace
{
    inline uint32_t pcgStep(uint64_t &state)
    {
        // LCG step. The increment 1442695040888963407 (a popular
        // odd-prime seed-stream constant) keeps successive values
        // well-distributed even when state starts at 0.
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
        // Output: xorshift then random rotation.
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
    }
}

float NumGen::Epsilon()
{
    thread_local uint64_t state = []() -> uint64_t {
        std::random_device rd;
        // Seed with two 32-bit draws; std::random_device gives 32 bits
        // per call. Mix into a 64-bit PCG state.
        return (uint64_t(rd()) << 32) | rd();
    }();
    // 1.0f / 4294967296.0f maps a full uint32 into [0, 1) exactly. The
    // hard-coded reciprocal is faster than (rand / max) and matches
    // what the GPU shader does in its rand() helper.
    return (float)pcgStep(state) * (1.0f / 4294967296.0f);
}
