#pragma once

#include <cstdint>
#include <random>

class NumGen
{
public:
    static float Epsilon();

    // Optional deterministic seed. When set to non-zero, thread-local
    // PRNG state initializes from this value rather than from
    // std::random_device, making renders bit-reproducible across runs
    // on the same machine.
    //
    // Cross-machine determinism additionally requires single-threaded
    // rendering: thread interleaving on different core counts changes
    // the PRNG draw order. Renderer.cpp checks getSeed() != 0 and
    // forces numThreads = 1 in that case. Cost is one render in
    // exchange for deterministic golden-image diffing in tests/render.
    //
    // 0 means "no fixed seed, use random_device" (the default). If a
    // user genuinely wants seed = 0, they can pass --seed 1 or any
    // other non-zero value.
    static void setSeed(uint64_t seed);
    static uint64_t getSeed();

private:
    NumGen() = delete;
};