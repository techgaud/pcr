#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>

#include "Ray.h"
#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"
#include "../Bvh/Bvh.h"
#include "../Scenes/Scene.h"

class Renderer
{
public:
    Renderer(int width, int height, int depth, int samples, int shadowSamples);

    // Optional GUI hooks. All may be null (the CLI leaves them null and
    // gets identical behavior to before). When non-null:
    //   progressRows    -> incremented by 1 each time a row is finished
    //   cancelRequested -> checked between rows, render bails if true
    //   onPartialFrame  -> called periodically (~2 Hz) from a snapshot thread
    //                      while the render is running, with the current (still
    //                      mutating) framebuffer. The callback should copy out
    //                      what it needs and return quickly.
    std::atomic<int> *progressRows = nullptr;
    std::atomic<bool> *cancelRequested = nullptr;
    std::function<void(const std::vector<Vec3f> &, int width, int height)> onPartialFrame;

    // Quality knobs. All default off so behavior matches the historical
    // baseline; flip on per-render via CLI flag or GUI toggle.
    bool useDenoise   = false; // post-process bilateral filter on the final RGB
    bool useMIS       = false; // multiple importance sampling for direct lighting
    bool useRussian   = false; // Russian roulette path termination at depth >= 1
    bool useStratified = false; // jittered stratified samples instead of pure random
    bool useACES      = false; // ACES filmic tone-map (default: Reinhard)

    // Per-pixel primary-ray count for anti-aliasing. 1 = no AA (one ray
    // dead through pixel center, current behavior). >1 = jittered primary
    // rays sampled across the pixel area, results averaged. Linear cost
    // multiplier (aaSamples=4 -> 4x total render time).
    int aaSamples = 1;

    // Adaptive sampling: within the per-pixel AA loop, track running
    // variance via Welford's algorithm and early-exit once relative
    // variance drops below threshold. Speeds up well-converged regions
    // (smooth walls, shadow-free direct light) at the cost of more
    // samples in noisy regions (corners, deep shadows). Only meaningful
    // when aaSamples > 1.
    bool useAdaptive = false;

    // Intel Open Image Denoise. Replaces (not complements) the existing
    // 5x5 bilateral filter when on. Runs on the HDR pre-tone-map
    // framebuffer with optional albedo + normal aux buffers, which the
    // renderer also outputs at primary-ray first hit. Build-time gated
    // by PCR_USE_OIDN. when not built with it, OidnDenoise::denoise is
    // a no-op and the flag prints a warning.
    bool useOIDN = false;

    void render(const Scenes::SceneData &scene,
                std::chrono::steady_clock::time_point start,
                const std::string &outputDir);

    // Filename of the most recently written PNG (set inside render()).
    std::string lastOutputPath;

private:
    std::vector<Plane> _planes;
    const int _width;
    const int _height;
    const int _maxDepth;
    const int _samples;
    const int _shadowSamples;

    // Optional out-params capture albedo + shading normal at the FIRST
    // hit (depth==0). Used to populate aux buffers for OIDN; recursive
    // calls inside castRay leave them null so deeper bounces don't
    // overwrite. Background-hit (no intersect at depth 0) writes
    // sentinel zeros so the aux buffer doesn't get random stack values.
    Vec3f castRay(const Ray &ray, const std::vector<Sphere> &spheres,
                  const std::vector<Triangle> &triangles,
                  const std::vector<Bvh::Node> &bvh,
                  const std::vector<Scenes::AreaLight> &lights,
                  float totalLightArea, int depth,
                  Vec3f *outFirstAlbedo = nullptr,
                  Vec3f *outFirstNormal = nullptr);
    bool sceneIntersect(const Ray &ray, const std::vector<Sphere> &spheres,
                        const std::vector<Triangle> &triangles,
                        const std::vector<Bvh::Node> &bvh,
                        Vec3f &hit, Vec3f &N, Material &material);
};
