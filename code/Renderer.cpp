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
#include "Includes/Denoise.h"

// PCR_BINARY_NAME is set per-target in CMake. Fallback for safety.
#ifndef PCR_BINARY_NAME
#define PCR_BINARY_NAME "frank-based-rendering"
#endif

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

Renderer::Renderer(int width, int height, int depth, int samples, int shadowSamples)
    : _width{width}, _height{height}, _maxDepth{depth}, _samples{samples}, _shadowSamples{shadowSamples}
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
    Vec3f origin = scene.camera.position;
    float aspect = _width / (float)_height;
    float scale = std::tan((float)std::numbers::pi / 180.f * 0.5f * scene.camera.fov);
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);

    for (size_t t = 0; t < numThreads; t++)
    {
        threads[t] = std::thread([&, t]()
                                 {
                                    try{
            for (size_t i = t; i < (size_t)_height; i += numThreads)
            {
                if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
                    return;
                for (size_t j = 0; j < (size_t)_width; j++)
                {
                    // Check cancel periodically inside the row, not just
                    // between rows. At high quality settings a single row
                    // can take many seconds; without this, cancel feels broken.
                    // Once every 64 pixels is plenty responsive (sub-second on
                    // typical hardware) and the atomic load is cheap.
                    if ((j & 63) == 0 && cancelRequested &&
                        cancelRequested->load(std::memory_order_relaxed))
                        return;
                    auto x = ((2 * (j + 0.5f) / (float)_width) - 1) * scale * aspect;
                    auto y = -((2 * (i + 0.5f) / (float)_height) - 1) * scale;
                    Ray ray(Vec3f(x, y, -1.f).normalize(), origin);
                    frameBuffer[i * _width + j] = castRay(ray, scene.spheres, scene.triangles, 0);
                }
                if (progressRows)
                    progressRows->fetch_add(1, std::memory_order_relaxed);
            } }
            catch(const std::exception& ex){
                std::cerr << "Thread " << t << " threw: " << ex.what() << std::endl;
            }
            catch(...){
                std::cerr << "Thread " << t << " threw an unknown exception" << std::endl;
            } });
    }

    // Snapshot thread for live preview. Wakes ~2 Hz while workers run, calls
    // user callback with the current (still-mutating) framebuffer. Tearing on
    // a few pixels per frame is acceptable for preview. Inactive when no
    // callback is registered (the common CLI path).
    std::atomic<bool> rendering{true};
    std::thread snapshotThread;
    if (onPartialFrame)
    {
        snapshotThread = std::thread([&]() {
            while (rendering.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (!rendering.load(std::memory_order_relaxed)) break;
                onPartialFrame(frameBuffer, _width, _height);
            }
        });
    }

    for (auto &t : threads)
        t.join();

    rendering.store(false, std::memory_order_relaxed);
    if (snapshotThread.joinable())
        snapshotThread.join();

    if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
    {
        std::cout << "Render cancelled before write." << std::endl;
        lastOutputPath.clear();
        return;
    }

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

    if (useDenoise)
        Denoise::bilateralRGB(rgb, _width, _height);

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
    addText("Software", PCR_BINARY_NAME);
    addText("Scene", scene.name);
    addText("SceneVersion", scene.version);
    addText("CreationTime", timestamp);
    addText("Depth", std::to_string(_maxDepth));
    addText("Samples", std::to_string(_samples));
    addText("ShadowSamples", std::to_string(_shadowSamples));
    addText("Width", std::to_string(_width));
    addText("Height", std::to_string(_height));
    addText("RenderTimeMs", std::to_string(elapsedMs));
    addText("Denoise",    useDenoise    ? "1" : "0");
    addText("MIS",        useMIS        ? "1" : "0");
    addText("Russian",    useRussian    ? "1" : "0");
    addText("Stratified", useStratified ? "1" : "0");

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
    lastOutputPath = outputPath.string();
}

Vec3f Renderer::castRay(const Ray &ray, const std::vector<Sphere> &spheres,
                        const std::vector<Triangle> &triangles, int depth)
{
    Material material;
    Vec3f hit, N;

    if (depth >= _maxDepth || !sceneIntersect(ray, spheres, triangles, hit, N, material))
        return Vec3f(0.f, 0.f, 0.f); // background color

    if (ray.dir.dot(N) > 0)
        N = N * -1;

    if (material.isEmissive())
        return material.emissive;

    Vec3f indirectLo;

    // Stratified sampling: lay out the indirect samples on a sqrt(N) x sqrt(N)
    // jittered grid instead of fully random. Reduces variance for the same N.
    const int strata = useStratified
                       ? std::max(1, (int)std::round(std::sqrt((float)_samples)))
                       : 0;

    for (size_t i = 0; i < (size_t)_samples; i++)
    {
        float r1, r2;
        if (useStratified)
        {
            int sx = (int)i % strata;
            int sy = ((int)i / strata) % strata;
            r1 = (sx + NumGen::Epsilon()) / (float)strata;
            r2 = (sy + NumGen::Epsilon()) / (float)strata;
        }
        else
        {
            r1 = NumGen::Epsilon();
            r2 = NumGen::Epsilon();
        }
        auto randomRay = Ray::genRayFromIntersection(N, hit + N * 1e-3, r1, r2);

        // Russian roulette: at depth >= 1, terminate with probability (1 - p)
        // where p reflects the surface's reflectance. Survivors get scaled by
        // 1/p to keep the estimator unbiased. Skipped at depth 0 so direct
        // viewing rays always shoot at least one bounce.
        if (useRussian && depth >= 1)
        {
            float maxAlbedo = std::max({material.albedo[0], material.albedo[1], material.albedo[2]});
            float p = std::min(0.95f, std::max(0.05f, maxAlbedo));
            if (NumGen::Epsilon() > p) continue;
            indirectLo += castRay(randomRay, spheres, triangles, depth + 1) * material.albedo / p;
        }
        else
        {
            indirectLo += castRay(randomRay, spheres, triangles, depth + 1) * material.albedo;
        }
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
        bool inShadow = sceneIntersect(Ray(wi, shadowOrigin), spheres, triangles, shadowHit, shadowN, tmpMat) && lightDist2 - 1e-3 > (shadowHit - shadowOrigin).dot(shadowHit - shadowOrigin) && !tmpMat.isEmissive();

        if (!inShadow){
            float cosLight = std::max(0.f, _light.N.dot(Li * -1));
            float G = (cosTheta * cosLight) / lightDist2;
            Vec3f directContrib = (material.albedo / std::numbers::pi) * _light.material.emissive * G * _light.getArea();

            // Partial MIS: down-weight the direct contribution by the
            // balance heuristic between light and BRDF sampling pdfs. Note
            // this is the "light-side" half of MIS; the symmetric BRDF-side
            // weighting on emissive returns from indirect bounces would
            // require a refactor of the recursion, so we ship the simpler
            // half. For diffuse-only Cornell the visible difference is small.
            if (useMIS && cosLight > 1e-6f)
            {
                float lightArea = _light.getArea();
                float pdfLight = lightDist2 / (cosLight * lightArea);
                float pdfBrdf  = cosTheta / (float)std::numbers::pi;
                float w = (pdfLight * pdfLight) /
                          (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                directContrib *= w;
            }
            directLo += directContrib;
        }
    }

    return directLo / _shadowSamples + indirectLo;
}

bool Renderer::sceneIntersect(const Ray &ray, const std::vector<Sphere> &spheres,
                              const std::vector<Triangle> &triangles,
                              Vec3f &hit, Vec3f &N, Material &material)
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

    // Linear scan over triangles. Phase 2 replaces this with BVH traversal
    // once mesh imports start producing thousands of them.
    Vec3f triHit, triN;
    for (const auto &tri : triangles)
    {
        if (!tri.intersect(ray, triHit, triN, t0, closest_t) || t0 >= closest_t)
            continue;

        closest_t = t0;
        hit = triHit;
        N = triN;
        material = tri.material;
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
