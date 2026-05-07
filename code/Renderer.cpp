#include <cctype>
#include <vector>
#include <numbers>
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <ctime>
#include <string>
#include <unordered_map>

#include "Bvh/Bvh.h"
#include "Includes/Renderer.h"
#include "Includes/Vec3f.h"
#include "Includes/lodepng.h"
#include "Includes/Denoise.h"
#include "Includes/OidnDenoise.h"
#include "Includes/ToneMap.h"

// PCR_BINARY_NAME is set per-target in CMake. Fallback for safety.
#ifndef PCR_BINARY_NAME
#define PCR_BINARY_NAME "frank-based-rendering"
#endif

namespace
{
    // strftime("%Z") returns short abbreviations on POSIX ("EDT", "PST")
    // but on Windows it returns the long Windows-registry name ("Eastern
    // Daylight Time", "Pacific Standard Time"). Compress those long names
    // into the same short form POSIX uses, so output filenames look the
    // same across platforms.
    std::string compressZone(const std::string &z)
    {
        static const std::unordered_map<std::string, std::string> map = {
            {"Eastern Standard Time",       "EST"},
            {"Eastern Daylight Time",       "EDT"},
            {"Central Standard Time",       "CST"},
            {"Central Daylight Time",       "CDT"},
            {"Mountain Standard Time",      "MST"},
            {"Mountain Daylight Time",      "MDT"},
            {"Pacific Standard Time",       "PST"},
            {"Pacific Daylight Time",       "PDT"},
            {"Alaskan Standard Time",       "AKST"},
            {"Alaskan Daylight Time",       "AKDT"},
            {"Hawaiian Standard Time",      "HST"},
            {"GMT Standard Time",           "GMT"},
            {"GMT Daylight Time",           "BST"},
            {"Coordinated Universal Time",  "UTC"},
        };
        auto it = map.find(z);
        if (it != map.end()) return it->second;
        // If the input has no spaces it's already an abbreviation
        // (POSIX returns short forms like "EDT", "PST", "UTC"). Pass
        // through.
        if (z.find(' ') == std::string::npos) return z;
        // Fallback: compose from the first letter of each whitespace-
        // separated word. "Some Other Zone" -> "SOZ". Ad-hoc but better
        // than spaces in a filename.
        std::string abbrev;
        bool atWordStart = true;
        for (char c : z)
        {
            if (c == ' ') { atWordStart = true; continue; }
            if (atWordStart && std::isalpha((unsigned char)c))
                abbrev += (char)std::toupper((unsigned char)c);
            atWordStart = false;
        }
        return abbrev.empty() ? z : abbrev;
    }

    // Format current time as "YYYYMMDD-HHMMSS-<ZONE>". When utc=false, uses
    // system local time and a short zone abbreviation (EDT, EST, PST, ...).
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
        std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
        std::string out(buf);
        if (utc)
        {
            out += "-UTC";
        }
        else
        {
            char zone[64];
            std::strftime(zone, sizeof(zone), "%Z", &tm);
            out += "-";
            out += compressZone(zone);
        }
        return out;
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
    // Scene geometry: light planes first, then walls. The order matters —
    // when a light plane is coplanar with a wall (a ceiling-cutout light is
    // the obvious case), the iteration-order tie has to go to the light or
    // shadow rays bound for the light get falsely occluded by the wall.
    // Hardcoded and JSON paths both keep light planes ONLY in areaLights;
    // walls is light-free.
    _planes.clear();
    for (const auto &L : scene.areaLights)
    {
        if (L.kind == Scenes::AreaLightKind::Plane)
            _planes.push_back(L.plane);
    }
    for (const auto &w : scene.walls)
        _planes.push_back(w);

    // Total area across all area lights, cached so the shadow loop doesn't
    // re-sum on every sample.
    float totalLightArea = 0.f;
    for (const auto &L : scene.areaLights)
        totalLightArea += L.totalArea;

    std::vector<Vec3f> frameBuffer(_width * _height);
    // OIDN aux buffers: only allocated when --oidn is on. Albedo and
    // shading normal at primary-ray first hit, captured below.
    std::vector<Vec3f> albedoBuffer;
    std::vector<Vec3f> normalBuffer;
    if (useOIDN)
    {
        albedoBuffer.assign(_width * _height, Vec3f(0, 0, 0));
        normalBuffer.assign(_width * _height, Vec3f(0, 0, 1));
    }
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
                    // Per-pixel primary-ray loop. aaSamples=1 (default) keeps
                    // legacy behavior — one ray dead through pixel center,
                    // no jitter. aaSamples>1 fires jittered primary rays
                    // across the pixel area and averages, integrating edge
                    // coverage. Welford's algorithm tracks running mean and
                    // M2 (sum of squared deviations) so we can early-exit
                    // adaptive renders once relative variance drops below
                    // the convergence threshold.
                    int aa = std::max(1, aaSamples);
                    Vec3f mean{0, 0, 0};
                    Vec3f M2{0, 0, 0};
                    int taken = 0;
                    for (int s = 0; s < aa; s++)
                    {
                        float jx = (aa > 1) ? (NumGen::Epsilon() - 0.5f) : 0.f;
                        float jy = (aa > 1) ? (NumGen::Epsilon() - 0.5f) : 0.f;
                        auto x = ((2 * (j + 0.5f + jx) / (float)_width) - 1) * scale * aspect;
                        auto y = -((2 * (i + 0.5f + jy) / (float)_height) - 1) * scale;
                        Ray ray(Vec3f(x, y, -1.f).normalize(), origin);
                        // Capture aux only on the first AA sample. OIDN
                        // doesn't need anti-aliased aux; the deterministic
                        // per-pixel-center first hit is enough.
                        Vec3f firstAlbedo, firstNormal;
                        Vec3f *albOut = (useOIDN && s == 0) ? &firstAlbedo : nullptr;
                        Vec3f *nrmOut = (useOIDN && s == 0) ? &firstNormal : nullptr;
                        Vec3f c = castRay(ray, scene.spheres, scene.triangles, scene.triangleBvh, scene.areaLights, totalLightArea, 0, albOut, nrmOut);
                        if (albOut) albedoBuffer[i * _width + j] = firstAlbedo;
                        if (nrmOut) normalBuffer[i * _width + j] = firstNormal;

                        taken++;
                        Vec3f delta{c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]};
                        mean[0] += delta[0] / taken;
                        mean[1] += delta[1] / taken;
                        mean[2] += delta[2] / taken;
                        Vec3f delta2{c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]};
                        M2[0] += delta[0] * delta2[0];
                        M2[1] += delta[1] * delta2[1];
                        M2[2] += delta[2] * delta2[2];

                        // Adaptive convergence check. After at least 4
                        // samples, stop if every channel's relative
                        // variance is below threshold. Relative form
                        // (variance/(mean²+eps)) handles HDR scenes where
                        // a fixed absolute threshold would mis-fire on
                        // bright pixels.
                        if (useAdaptive && taken >= 4)
                        {
                            constexpr float absMin = 0.01f;
                            constexpr float relThreshold = 0.05f;
                            bool converged = true;
                            for (int ch = 0; ch < 3; ch++)
                            {
                                float v = M2[ch] / (float)(taken - 1);
                                float rel = v / (mean[ch] * mean[ch] + absMin);
                                if (rel >= relThreshold) { converged = false; break; }
                            }
                            if (converged) break;
                        }
                    }
                    frameBuffer[i * _width + j] = mean;
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

    // OIDN denoise BEFORE tone mapping. OIDN expects HDR linear radiance
    // values; running it post-Reinhard would clobber its training-data
    // assumptions. The aux buffers (albedo, shading normal) populated
    // during the per-pixel loop above feed the denoiser network.
    if (useOIDN)
    {
        if (!OidnDenoise::isAvailable())
        {
            std::cerr << "warning: --oidn requested but binary was not built "
                         "with PCR_USE_OIDN=ON; skipping denoise.\n";
        }
        else
        {
            OidnDenoise::denoise(frameBuffer, albedoBuffer, normalBuffer,
                                 _width, _height);
        }
    }

    std::vector<unsigned char> rgb((size_t)_width * _height * 3);
    for (size_t i = 0; i < (size_t)_width * _height; i++)
    {
        if (useACES)
            ToneMap::aces(frameBuffer[i]);
        else
            ToneMap::reinhard(frameBuffer[i]);

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
    if (aaSamples > 1)
    {
        filename += "-aa" + std::to_string(aaSamples);
        if (useAdaptive) filename += "adaptive";
    }
    if (useACES)
        filename += "-aces";
    if (useOIDN)
        filename += "-oidn";
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
    addText("Tonemap",    useACES       ? "ACES" : "Reinhard");
    addText("AASamples",  std::to_string(aaSamples));
    addText("Adaptive",   useAdaptive ? "1" : "0");
    addText("OIDN",       useOIDN     ? "1" : "0");

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
                        const std::vector<Triangle> &triangles,
                        const std::vector<Bvh::Node> &bvh,
                        const std::vector<Scenes::AreaLight> &lights,
                        float totalLightArea, int depth,
                        Vec3f *outFirstAlbedo,
                        Vec3f *outFirstNormal)
{
    Material material;
    Vec3f hit, N;

    if (depth >= _maxDepth || !sceneIntersect(ray, spheres, triangles, bvh, hit, N, material))
    {
        // Background hit: write sentinel aux so the OIDN buffer doesn't
        // contain stack garbage for sky-pixels.
        if (outFirstAlbedo) *outFirstAlbedo = Vec3f(0.f, 0.f, 0.f);
        if (outFirstNormal) *outFirstNormal = Vec3f(0.f, 0.f, 1.f);
        return Vec3f(0.f, 0.f, 0.f);
    }

    // Capture which side of the surface we hit BEFORE flipping N — glass
    // refraction needs to know whether we're entering (n1=air, n2=glass)
    // or exiting (n1=glass, n2=air). Diffuse + mirror only need N facing
    // the ray, so the flip below preserves their behavior.
    bool entering = ray.dir.dot(N) < 0.f;
    if (!entering)
        N = N * -1;

    // OIDN aux at first hit. For specular materials we'd ideally capture
    // the underlying surface's albedo (mirror's tint, glass's color);
    // material.albedo serves that role. Normal is the geometric one
    // facing the ray, which is what OIDN expects.
    if (outFirstAlbedo) *outFirstAlbedo = material.albedo;
    if (outFirstNormal) *outFirstNormal = N;

    if (material.isEmissive())
        return material.emissive;

    // Perfect mirror: deterministic reflection, no diffuse contribution
    // and no shadow rays. Albedo tints the recursive radiance.
    if (material.metallic)
    {
        float cosI = -ray.dir.dot(N);
        Vec3f reflectedDir = ray.dir + N * (2.f * cosI);
        Vec3f reflOrigin = hit + N * 1e-3f;
        Vec3f recurse = castRay(Ray(reflectedDir, reflOrigin),
                                spheres, triangles, bvh, lights, totalLightArea, depth + 1);
        return Vec3f(recurse[0] * material.albedo[0],
                     recurse[1] * material.albedo[1],
                     recurse[2] * material.albedo[2]);
    }

    // Glass: Fresnel-weighted reflection vs refraction. Schlick's
    // approximation for the Fresnel coefficient. Total internal
    // reflection when refraction is impossible.
    if (material.transparent)
    {
        float n1 = entering ? 1.0f : material.ior;
        float n2 = entering ? material.ior : 1.0f;
        float cosI = -ray.dir.dot(N); // positive since N faces ray
        float eta = n1 / n2;
        float sinT2 = eta * eta * (1.f - cosI * cosI);

        Vec3f outDir;
        Vec3f outOrigin;
        if (sinT2 >= 1.f)
        {
            // Total internal reflection — only reflection survives.
            outDir = ray.dir + N * (2.f * cosI);
            outOrigin = hit + N * 1e-3f;
        }
        else
        {
            float cosT = std::sqrt(1.f - sinT2);
            // Schlick's Fresnel approximation
            float F0 = (n1 - n2) / (n1 + n2); F0 *= F0;
            float F = F0 + (1.f - F0) * std::pow(1.f - cosI, 5.f);
            if (NumGen::Epsilon() < F)
            {
                outDir = ray.dir + N * (2.f * cosI);
                outOrigin = hit + N * 1e-3f;
            }
            else
            {
                outDir = ray.dir * eta + N * (eta * cosI - cosT);
                outOrigin = hit - N * 1e-3f;
            }
        }
        Vec3f recurse = castRay(Ray(outDir, outOrigin),
                                spheres, triangles, bvh, lights, totalLightArea, depth + 1);
        return Vec3f(recurse[0] * material.albedo[0],
                     recurse[1] * material.albedo[1],
                     recurse[2] * material.albedo[2]);
    }

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
            indirectLo += castRay(randomRay, spheres, triangles, bvh, lights, totalLightArea, depth + 1) * material.albedo / p;
        }
        else
        {
            indirectLo += castRay(randomRay, spheres, triangles, bvh, lights, totalLightArea, depth + 1) * material.albedo;
        }
    }
    indirectLo /= _samples;

    Vec3f directLo;
    if (totalLightArea > 0.f)
    {
        for (size_t i = 0; i < (size_t)_shadowSamples; i++)
        {
            // Pick one light proportional to its surface area.
            const Scenes::AreaLight *picked = &lights.front();
            {
                float pickTarget = NumGen::Epsilon() * totalLightArea;
                float cumul = 0.f;
                for (const auto &L : lights)
                {
                    cumul += L.totalArea;
                    if (pickTarget <= cumul) { picked = &L; break; }
                }
            }

            // Sample uniformly within picked. PDF over total light surface
            // area = 1/totalLightArea regardless of which light was picked
            // (pick_i * within_i = (area_i/totalArea) * (1/area_i) = 1/totalArea).
            Vec3f sampleP, sampleN, sampleEmissive;
            if (picked->kind == Scenes::AreaLightKind::Plane)
            {
                const Plane &p = picked->plane;
                float ru = NumGen::Epsilon();
                float rv = NumGen::Epsilon();
                sampleP = p.origin + p.getU() * ru + p.getV() * rv;
                sampleN = p.N;
                sampleEmissive = p.material.emissive;
            }
            else
            {
                float rtri = NumGen::Epsilon() * picked->totalArea;
                auto it = std::lower_bound(picked->cumulativeArea.begin(),
                                           picked->cumulativeArea.end(), rtri);
                int triIdx = std::min((int)(it - picked->cumulativeArea.begin()),
                                      (int)picked->triangles.size() - 1);
                const Triangle &tri = picked->triangles[triIdx];

                // Uniform sample within a triangle: the standard r1+r2 fold.
                float r1 = NumGen::Epsilon();
                float r2 = NumGen::Epsilon();
                if (r1 + r2 > 1.f) { r1 = 1.f - r1; r2 = 1.f - r2; }
                sampleP = tri.v0 + (tri.v1 - tri.v0) * r1 + (tri.v2 - tri.v0) * r2;
                sampleN = tri.flatN;
                sampleEmissive = tri.material.emissive;
            }

            Vec3f Li = sampleP - hit;
            auto wi = Li.normalize();
            auto cosTheta = std::max(0.f, wi.dot(N));
            auto lightDist2 = Li.dot(Li);

            // handle shadows
            auto shadowOrigin = cosTheta <= 0 ? hit - N * 1e-3 : hit + N * 1e-3;
            Vec3f shadowHit, shadowN;
            Material tmpMat;
            bool inShadow = sceneIntersect(Ray(wi, shadowOrigin), spheres, triangles, bvh, shadowHit, shadowN, tmpMat) && lightDist2 - 1e-3 > (shadowHit - shadowOrigin).dot(shadowHit - shadowOrigin) && !tmpMat.isEmissive();

            if (!inShadow)
            {
                float cosLight = std::max(0.f, sampleN.dot(Li * -1));
                float G = (cosTheta * cosLight) / lightDist2;
                // pdf = 1/totalLightArea, so divide-by-pdf = totalLightArea.
                Vec3f directContrib = (material.albedo / std::numbers::pi) * sampleEmissive * G * totalLightArea;

                // Partial MIS: down-weight the direct contribution by the
                // balance heuristic between light and BRDF sampling pdfs.
                // Light-side half only; the symmetric BRDF-side weighting on
                // emissive returns from indirect bounces would require a
                // recursion refactor. For diffuse-only scenes the visible
                // effect is small.
                if (useMIS && cosLight > 1e-6f)
                {
                    float pdfLight = lightDist2 / (cosLight * totalLightArea);
                    float pdfBrdf  = cosTheta / (float)std::numbers::pi;
                    float w = (pdfLight * pdfLight) /
                              (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                    directContrib *= w;
                }
                directLo += directContrib;
            }
        }
    }

    return directLo / _shadowSamples + indirectLo;
}

bool Renderer::sceneIntersect(const Ray &ray, const std::vector<Sphere> &spheres,
                              const std::vector<Triangle> &triangles,
                              const std::vector<Bvh::Node> &bvh,
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

    // Triangles go through the BVH if one was built, otherwise linear.
    // The BVH is empty for scenes with zero triangles (most cornell scenes
    // today); for scenes with even a handful of triangles, BVH traversal is
    // already a wash with linear, and it's a clear win once a mesh lands.
    if (!bvh.empty())
    {
        Vec3f triHit, triN;
        Material triMat;
        float triT;
        if (Bvh::intersect(bvh, triangles, ray, triHit, triN, triMat, triT, closest_t))
        {
            closest_t = triT;
            hit = triHit;
            N = triN;
            material = triMat;
        }
    }
    else
    {
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
    }

    return closest_t < std::numeric_limits<float>::max();
}

