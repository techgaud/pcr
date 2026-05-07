#pragma once

#include "Vec3f.h"

// Shared tone-mapping curves, used by both the CPU Renderer and the GPU
// renderer's CPU-side post-readback path. The GPU shader used to do this
// in-place via GLSL aces()/reinhard() functions, but with OIDN aux
// buffers the shader now stores HDR linear radiance and tone-mapping
// runs on CPU after denoise. Same math, single source of truth.
namespace ToneMap
{
    // Simple Reinhard: c / (c + 1). Smooth concave curve, low-contrast
    // midtones. Cheap and parameter-free.
    inline void reinhard(Vec3f &c)
    {
        c[0] = c[0] / (c[0] + 1.f);
        c[1] = c[1] / (c[1] + 1.f);
        c[2] = c[2] / (c[2] + 1.f);
    }

    // Narkowicz 2015 ACES filmic approximation. Per-channel S-curve with
    // gentle toe and shoulder; preserves midtone contrast better than
    // Reinhard at the cost of mild hue shifts in saturated highlights.
    inline void aces(Vec3f &c)
    {
        constexpr float A = 2.51f;
        constexpr float B = 0.03f;
        constexpr float C = 2.43f;
        constexpr float D = 0.59f;
        constexpr float E = 0.14f;
        for (int i = 0; i < 3; i++)
        {
            float x = c[i];
            float v = (x * (A * x + B)) / (x * (C * x + D) + E);
            if (v < 0.f) v = 0.f;
            if (v > 1.f) v = 1.f;
            c[i] = v;
        }
    }
}
