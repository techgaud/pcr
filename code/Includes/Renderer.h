#pragma once

#include <atomic>
#include <chrono>
#include <string>

#include "Ray.h"
#include "Sphere.h"
#include "Plane.h"
#include "../Scenes/Scene.h"

class Renderer
{
public:
    Renderer(int width, int height, float fov, int depth, int samples, int shadowSamples);

    // Optional GUI hooks. Either may be null (the CLI leaves both null and
    // gets identical behavior to before). When non-null:
    //   progressRows   -> incremented by 1 each time a row is finished
    //   cancelRequested -> checked between rows, render bails if true
    std::atomic<int> *progressRows = nullptr;
    std::atomic<bool> *cancelRequested = nullptr;

    void render(const Scenes::SceneData &scene,
                std::chrono::steady_clock::time_point start,
                const std::string &outputDir);

    // Filename of the most recently written PNG (set inside render()).
    std::string lastOutputPath;

private:
    std::vector<Plane> _planes;
    Plane _light;
    const float _fov;
    const int _width;
    const int _height;
    const int _maxDepth;
    const int _samples;
    const int _shadowSamples;

    Vec3f castRay(const Ray &ray, const std::vector<Sphere> &spheres, int depth);
    bool sceneIntersect(const Ray &ray, const std::vector<Sphere> &spheres, Vec3f &hit, Vec3f &N, Material &material);
    void reinhardToneMap(Vec3f &color);
};
