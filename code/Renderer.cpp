#include <vector>
#include <numbers>
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <ctime>
#include <string>

#include "Includes/Renderer.h"
#include "Includes/Vec3f.h"
#include "Includes/lodepng.h"

namespace
{
    // Format current time as "YYYYMMDD-HHMMSS-<ZONE>". When utc=false, uses
    // system local time and the active zone abbreviation (e.g. EDT, EST, PST).
    // When utc=true, uses UTC and a literal "UTC" suffix.
    std::string formatTimestamp(bool utc)
    {
        std::time_t now = std::time(nullptr);
        std::tm tm{};
#ifdef _MSC_VER
        if (utc) gmtime_s(&tm, &now);
        else localtime_s(&tm, &now);
#else
        if (utc) gmtime_r(&now, &tm);
        else localtime_r(&now, &tm);
#endif
        char buf[64];
        if (utc)
            std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S-UTC", &tm);
        else
            std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S-%Z", &tm);
        return std::string(buf);
    }
}

Renderer::Renderer(int width, int height, float fov, int depth, int samples, int shadowSamples)
    : _width{width}, _height{height}, _fov{fov}, _maxDepth{depth}, _samples{samples}, _shadowSamples{shadowSamples}
{
}

void Renderer::render(const Scenes::SceneData &scene,
                      std::chrono::steady_clock::time_point start,
                      const std::string &outputDir)
{
    // Scene geometry: light first, then walls. Spheres are passed through
    // separately to castRay/sceneIntersect since they use a different intersect.
    _light = scene.lightSource;
    _planes.clear();
    _planes.push_back(_light);
    for (const auto &w : scene.walls)
        _planes.push_back(w);

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
            for (size_t i = t; i < (size_t)_height; i += numThreads)
            {
                for (size_t j = 0; j < (size_t)_width; j++)
                {
                    auto x = ((2 * (j + 0.5f) / (float)_width) - 1) * scale * aspect;
                    auto y = -((2 * (i + 0.5f) / (float)_height) - 1) * scale;
                    Ray ray(Vec3f(x, y, -1.f).normalize(), origin);
                    frameBuffer[i * _width + j] = castRay(ray, scene.spheres, 0);
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

    std::vector<unsigned char> rgb((size_t)_width * _height * 3);
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

    auto end = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Render took " << elapsedMs << " ms" << std::endl;

    std::string timestamp = formatTimestamp(false); // flip to true for UTC

    // Filename: <scene>-<version>-<timestamp>-d#-s#-S#-w#[-h#]-t<ms>.png.
    // Height segment is omitted when height == width (the common square case).
    std::string filename = scene.name + "-" + scene.version + "-"
                         + timestamp
                         + "-d" + std::to_string(_maxDepth)
                         + "-s" + std::to_string(_samples)
                         + "-S" + std::to_string(_shadowSamples)
                         + "-w" + std::to_string(_width);
    if (_width != _height)
        filename += "-h" + std::to_string(_height);
    filename += "-t" + std::to_string(elapsedMs) + ".png";

    std::filesystem::path outputPath = std::filesystem::path(outputDir) / filename;
    std::filesystem::create_directories(outputPath.parent_path());

    // Encode PNG via lodepng so we can attach tEXt metadata chunks.
    lodepng::State state;
    state.info_raw.colortype = LCT_RGB;
    state.info_raw.bitdepth = 8;
    state.info_png.color.colortype = LCT_RGB;
    state.info_png.color.bitdepth = 8;
    // 0 = write uncompressed tEXt chunks (lodepng default writes zTXt). For
    // short metadata strings tEXt is simpler and more universally readable.
    state.encoder.text_compression = 0;

    auto addText = [&](const char *key, const std::string &val) {
        lodepng_add_text(&state.info_png, key, val.c_str());
    };
    addText("Software", "pcr-cornell");
    addText("Scene", scene.name);
    addText("SceneVersion", scene.version);
    addText("CreationTime", timestamp);
    addText("Depth", std::to_string(_maxDepth));
    addText("Samples", std::to_string(_samples));
    addText("ShadowSamples", std::to_string(_shadowSamples));
    addText("Width", std::to_string(_width));
    addText("Height", std::to_string(_height));
    addText("RenderTimeMs", std::to_string(elapsedMs));

    std::vector<unsigned char> pngBuffer;
    unsigned encErr = lodepng::encode(pngBuffer, rgb, _width, _height, state);
    if (encErr)
    {
        std::cerr << "lodepng encode error " << encErr << ": " << lodepng_error_text(encErr) << std::endl;
        return;
    }

    unsigned saveErr = lodepng::save_file(pngBuffer, outputPath.string());
    if (saveErr)
    {
        std::cerr << "lodepng save error " << saveErr << ": " << lodepng_error_text(saveErr) << " (path: " << outputPath << ")" << std::endl;
        return;
    }

    std::cout << "Wrote " << outputPath << std::endl;
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

    for (size_t i = 0; i < (size_t)_samples; i++)
    {
        auto randomRay = Ray::genRayFromIntersection(N, hit + N * 1e-3);
        auto cos = std::max(0.f, randomRay.dir.dot(N));
        (void)cos;
        indirectLo += castRay(randomRay, spheres, depth + 1) * material.albedo;
    }
    indirectLo /= _samples;

    Vec3f directLo;
    for (size_t i = 0; i < (size_t)_shadowSamples; i++)
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
