#include "Includes/NumGen.h"

#include <atomic>
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

    // Optional global seed set via NumGen::setSeed. 0 means "use
    // random_device per thread" (the default), which is what every
    // historical render uses. Non-zero produces bit-deterministic
    // renders when combined with single-threaded execution (Renderer
    // forces numThreads=1 in that case to avoid thread-interleaving
    // non-determinism).
    std::atomic<uint64_t> g_globalSeed{0};

    // Per-thread index counter, used to fan deterministic per-thread
    // PRNG states out from the global seed. fetch_add is atomic so
    // two threads asking for an index at the same time never collide.
    std::atomic<uint64_t> g_threadCounter{0};
}

void NumGen::setSeed(uint64_t seed)     { g_globalSeed.store(seed, std::memory_order_relaxed); }
uint64_t NumGen::getSeed()              { return g_globalSeed.load(std::memory_order_relaxed); }

float NumGen::Epsilon()
{
    thread_local uint64_t state = []() -> uint64_t {
        uint64_t seed = g_globalSeed.load(std::memory_order_relaxed);
        if (seed != 0)
        {
            // Deterministic per-thread state: derive from globalSeed and
            // a unique per-thread index. The index must be stable for the
            // lifetime of this thread, so we capture it here in the
            // thread_local initializer (runs once per thread on first
            // Epsilon() call).
            uint64_t threadIdx = g_threadCounter.fetch_add(1, std::memory_order_relaxed);
            // Mix seed with threadIdx via PCG step. One LCG step is enough
            // to scramble correlated inputs into independent streams.
            uint64_t s = seed ^ (threadIdx * 6364136223846793005ULL + 1442695040888963407ULL);
            return s == 0 ? 1ULL : s; // never start at 0 (degenerates LCG)
        }
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
