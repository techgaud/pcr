#include <vector>
#include <numbers>
#include <iostream>
#include <filesystem>
#include <thread>

#include "Includes/Renderer.h"
#include "Includes/Vec3f.h"
#include "Includes/stb_image_write.h"

Renderer::Renderer(int width, int height, float fov, int depth, int samples, int shadowSamples, Plane &light)
    : _width{width}, _height{height}, _fov{fov}, _maxDepth{depth}, _samples{samples}, _shadowSamples{shadowSamples}
{
    _light = light;
    _planes.push_back(light);
    createWalls();
}

void Renderer::render(const std::vector<Sphere> &spheres)
{
    std::vector<Vec3f> frameBuffer(_width * _height);
    Vec3f origin{0, 0, 0};
    float aspect = _width / (float)_height;
    float scale = std::tan((float)std::numbers::pi / 180.f * 0.5f * _fov);
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);

    for (size_t t = 0; t < numThreads; t++)
    {
        threads[t] = std::thread([&, t]()
                                 {
                                    try{
            for (size_t i = t; i < _height; i += numThreads)
            {
                for (size_t j = 0; j < _width; j++)
                {
                    auto x = ((2 * (j + 0.5f) / (float)_width) - 1) * scale * aspect;
                    auto y = -((2 * (i + 0.5f) / (float)_height) - 1) * scale;
                    Ray ray(Vec3f(x, y, -1.f).normalize(), origin);
                    frameBuffer[i * _width + j] = castRay(ray, spheres, 0);
                }
            } } 
            catch(const std::exception& ex){
                std::cerr << "Thread " << t << " threw: " << ex.what() << std::endl;
            }
            catch(...){
                std::cerr << "Thread " << t << " threw an unknown exception" << std::endl;
            } });
    }

    for (auto &t : threads)
        t.join();

    std::filesystem::path outputPath =
        std::filesystem::current_path().parent_path() / "Image" / "out.png";

    std::filesystem::create_directories(outputPath.parent_path());

    std::vector<unsigned char> rgb(_width * _height * 3);
    for (size_t i = 0; i < (size_t)_width * _height; i++)
    {
        reinhardToneMap(frameBuffer[i]);

        // gamma correction:
        // const float gamma = 2.2f;
        // frameBuffer[i][0] = std::pow(frameBuffer[i][0], 1/gamma);
        // frameBuffer[i][1] = std::pow(frameBuffer[i][1], 1/gamma);
        // frameBuffer[i][2] = std::pow(frameBuffer[i][2], 1/gamma);

        rgb[i * 3 + 0] = (unsigned char)(255 * frameBuffer[i][0] + 0.5f);
        rgb[i * 3 + 1] = (unsigned char)(255 * frameBuffer[i][1] + 0.5f);
        rgb[i * 3 + 2] = (unsigned char)(255 * frameBuffer[i][2] + 0.5f);
    }

    // PNG output is always lossless. Compression level only affects file
    // size vs. write speed; default is 8 (range 0-9), tweak if needed.
    if (!stbi_write_png(outputPath.string().c_str(), _width, _height, 3, rgb.data(), _width * 3))
        std::cerr << "Failed to write PNG to " << outputPath << std::endl;
}

Vec3f Renderer::castRay(const Ray &ray, const std::vector<Sphere> &spheres, int depth)
{
    Material material;
    Vec3f hit, N;

    if (depth >= _maxDepth || !sceneIntersect(ray, spheres, hit, N, material))
        return Vec3f(0.f, 0.f, 0.f); // background color

    if (ray.dir.dot(N) > 0)
        N = N * -1;

    if (material.isEmissive())
        return material.emissive;

    Vec3f indirectLo;
    // handle reflections
    //  auto reflectDir = ray.dir.reflect(N).normalize();
    //  auto reflectOrig = reflectDir.dot(N) > 0 ? hit + N * 1e-3 : hit - N * 1e-3;
    //  auto reflected = Ray(reflectDir, reflectOrig);
    //  indirectLo += castRay(reflected, spheres, depth + 1) * material.albedo;

    for (size_t i = 0; i < _samples; i++)
    {
        auto randomRay = Ray::genRayFromIntersection(N, hit + N * 1e-3);
        auto cos = std::max(0.f, randomRay.dir.dot(N));
        indirectLo += castRay(randomRay, spheres, depth + 1) * material.albedo;
    }
    indirectLo /= _samples;

    Vec3f directLo;
    for (size_t i = 0; i < _shadowSamples; i++)
    {
        auto Li = _light.getVecToPlaneFromHit(hit);
        auto wi = Li.normalize();
        auto cosTheta = std::max(0.f, wi.dot(N));
        auto lightDist2 = Li.dot(Li);

        // handle shadows
        auto shadowOrigin = cosTheta <= 0 ? hit - N * 1e-3 : hit + N * 1e-3;
        Vec3f shadowHit, shadowN;
        Material tmpMat;
        bool inShadow = sceneIntersect(Ray(wi, shadowOrigin), spheres, shadowHit, shadowN, tmpMat) && lightDist2 - 1e-3 > (shadowHit - shadowOrigin).dot(shadowHit - shadowOrigin) && !tmpMat.isEmissive();

        if (!inShadow){
            float cosLight = std::max(0.f, _light.N.dot(Li * -1));
            float G = (cosTheta * cosLight) / lightDist2;
            directLo += (material.albedo / std::numbers::pi) * _light.material.emissive * G * _light.getArea();
        }
    }

    return directLo / _shadowSamples + indirectLo;
}

bool Renderer::sceneIntersect(const Ray &ray, const std::vector<Sphere> &spheres, Vec3f &hit, Vec3f &N, Material &material)
{
    float closest_t = std::numeric_limits<float>::max();

    float t0 = 0.f;
    for (const auto &sphere : spheres)
    {
        if (!sphere.intersect(ray, t0) || t0 >= closest_t)
            continue;

        closest_t = t0;
        hit = ray.origin + ray.dir * t0;
        N = (hit - sphere.center).normalize();
        material = sphere.material;
    }

    for (const auto &plane : _planes)
    {
        if (!plane.intersect(ray, hit, t0, closest_t) || t0 >= closest_t)
            continue;

        closest_t = t0;
        N = plane.N;

        material = plane.material;
    }

    return closest_t < std::numeric_limits<float>::max();
}

void Renderer::createWalls()
{
    auto cream = Material{Vec3f(0.74f, 0.74f, 0.64f), Vec3f(0, 0, 0)};
    auto red = Material{Vec3f(0.63f, 0.06f, 0.05f), Vec3f(0, 0, 0)};
    auto green = Material{Vec3f(0.13f, 0.45f, 0.1f), Vec3f(0, 0, 0)};

    _planes.emplace_back(Vec3f{-2.f, 2.f, -6.f}, Vec3f{4, 0, 0}, Vec3f{0, 0, 7}, cream);  // ceiling
    _planes.emplace_back(Vec3f{2.f, 2.f, -6.f}, Vec3f{-6, 0, 0}, Vec3f{0, -6, 0}, cream); // back wall
    _planes.emplace_back(Vec3f{2.f, -2.f, 0.f}, Vec3f{0, 0, -7}, Vec3f{-4, 0, 0}, cream); // floor
    _planes.emplace_back(Vec3f{2.f, -2.f, 0.f}, Vec3f{0, 4, 0}, Vec3f{0, 0, -6}, green);  // right wall
    _planes.emplace_back(Vec3f{-2.f, -2.f, -6.f}, Vec3f{0, 4, 0}, Vec3f{0, 0, 6}, red);   // left wall
}

void Renderer::reinhardToneMap(Vec3f &color)
{
    // normal reinhard
    color[0] = color[0] / (color[0] + 1);
    color[1] = color[1] / (color[1] + 1);
    color[2] = color[2] / (color[2] + 1);

    // reinhard-jodie version
    // const float exposure = 0.65f;
    // color *= exposure;
    
    // float luminance = 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
    // if(luminance <= 0.f)
    //     return;
    // float mappedLuminance = luminance / (1.f + luminance);
    // color = color * (mappedLuminance / luminance);
}