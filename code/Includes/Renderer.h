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

    void render(const Scenes::SceneData &scene,
                std::chrono::steady_clock::time_point start,
                const std::string &outputDir);

    // Filename of the most recently written PNG (set inside render()).
    std::string lastOutputPath;

private:
    std::vector<Plane> _planes;
    Plane _light;
    const int _width;
    const int _height;
    const int _maxDepth;
    const int _samples;
    const int _shadowSamples;

    Vec3f castRay(const Ray &ray, const std::vector<Sphere> &spheres,
                  const std::vector<Triangle> &triangles,
                  const std::vector<Bvh::Node> &bvh, int depth);
    bool sceneIntersect(const Ray &ray, const std::vector<Sphere> &spheres,
                        const std::vector<Triangle> &triangles,
                        const std::vector<Bvh::Node> &bvh,
                        Vec3f &hit, Vec3f &N, Material &material);
    void reinhardToneMap(Vec3f &color);
};
