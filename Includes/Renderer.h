#pragma once

#include "Ray.h"
#include "Sphere.h"
#include "Plane.h"

class Renderer
{
public:
    Renderer(int width, int height, float fov, int depth, int samples, int shadowSamples, Plane &light);

    void render(const std::vector<Sphere> &sphere);

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
    void createWalls();
    void reinhardToneMap(Vec3f &color);
};