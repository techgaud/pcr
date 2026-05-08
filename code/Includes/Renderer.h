#pragma once

#include <array>
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

// Hero wavelength sampling (Wilkie et al. 2014). Each spectral ray
// carries N wavelengths through the same path geometry, so the
// expensive part (intersection + light sampling) is paid once per
// path while the cheap part (per-lambda spectrum lookups + scalar
// multiplies) is paid N times. Convergence on color noise improves
// roughly N-fold over single-wavelength for the same wall-clock
// cost. N=4 matches the paper's recommendation; the additional
// lambdas are stratified offsets of the hero, wrapped around the
// visible range so all 4 cover [400, 700] together.
static constexpr int kHeroLambdaCount = 4;
using SpectralSample = std::array<float, kHeroLambdaCount>;

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

    // Spectral rendering mode. When on, each primary ray samples a
    // single wavelength (uniform in [400, 700] nm), tracks scalar
    // radiance through bounces, and contributes a CIE XYZ value to
    // the per-pixel accumulator. After all samples done, XYZ is
    // converted to linear sRGB and the rest of the pipeline (OIDN,
    // tone-map, bilateral, PNG) is identical. Materials must have
    // populated spectra (set up at scene-load via populateSpectra).
    //
    // Single-wavelength-per-ray means convergence is slower than RGB
    // for the same sample count, since each sample only covers one
    // wavelength of the visible spectrum. aaSamples >= 16 recommended.
    // Hero-wavelength sampling (phase 5) will close most of that gap.
    bool useSpectral = false;

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

    // Hero wavelength variant for spectral mode. Tracks N=4 scalar
    // radiances at correlated wavelengths through the same path
    // geometry, sampling materials' albedoSpectrum and
    // emissiveSpectrum at each lambda. Russian roulette decisions
    // and direct-light sampling use the hero (lambdas[0]) channel
    // so the path itself is sampled from a single distribution; the
    // other 3 channels ride along. OIDN aux captures RGB albedo +
    // normal at first hit regardless of mode.
    SpectralSample castRaySpectral(const Ray &ray, const std::vector<Sphere> &spheres,
                                   const std::vector<Triangle> &triangles,
                                   const std::vector<Bvh::Node> &bvh,
                                   const std::vector<Scenes::AreaLight> &lights,
                                   float totalLightArea, int depth,
                                   const SpectralSample &lambdas,
                                   Vec3f *outFirstAlbedo = nullptr,
                                   Vec3f *outFirstNormal = nullptr);

    bool sceneIntersect(const Ray &ray, const std::vector<Sphere> &spheres,
                        const std::vector<Triangle> &triangles,
                        const std::vector<Bvh::Node> &bvh,
                        Vec3f &hit, Vec3f &N, Material &material);
};
