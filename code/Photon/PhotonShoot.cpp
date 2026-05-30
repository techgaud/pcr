#include "PhotonShoot.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>

#include "../Includes/Log.h"
#include "../Includes/Ray.h"
#include "../Includes/Optics.h"
#include "../Includes/Material.h"
#include "../Includes/Sphere.h"
#include "../Includes/Plane.h"
#include "../Includes/Triangle.h"
#include "../Bvh/Bvh.h"

namespace
{
    // Local PCG32 used only inside photon shooting. NumGen's
    // thread_local state would otherwise advance during the shoot,
    // which would change the eye-path render's PRNG sequence and
    // make --seed-deterministic A/B's between photon-map-on and
    // photon-map-off renders differ in the eye-path noise pattern
    // for reasons unrelated to the photon map. Keeping the photon-
    // shoot RNG local sidesteps that.
    struct PCG
    {
        uint64_t state;
        explicit PCG(uint64_t seed) : state{seed ? seed : 0xC0FFEEULL} {}
        float next()
        {
            uint64_t oldstate = state;
            state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
            uint32_t xs  = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = (uint32_t)(oldstate >> 59u);
            uint32_t out = (xs >> rot) | (xs << ((32u - rot) & 31u));
            return (float)out * (1.0f / 4294967296.0f);
        }
    };

    // Cosine-weighted hemisphere sample around a unit normal N.
    // Same Malley's method as Ray::genRayFromIntersection but inline
    // so we can use an explicit RNG instance.
    Vec3f cosineHemisphere(const Vec3f &N, float r1, float r2)
    {
        Vec3f helper = std::abs(N[0]) <= std::abs(N[1]) ? Vec3f(1, 0, 0)
                                                        : Vec3f(0, 1, 0);
        Vec3f T = N.cross(helper).normalize();
        Vec3f B = N.cross(T);

        float r = std::sqrt(r1);
        float phi = 2.0f * (float)std::numbers::pi * r2;
        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        float z = std::sqrt(std::max(0.f, 1.f - x * x - y * y));
        return (T * x + B * y + N * z).normalize();
    }

    // Locally re-implements the scene intersection that Renderer
    // does inside its private sceneIntersect. The photon trace
    // needs the same sphere+plane+triangle walk; rather than pry
    // open the Renderer's API surface, we duplicate the small loop
    // here. `combinedPlanes` is the union of area-light planes and
    // wall planes (Renderer does the same merge at the top of
    // render()).
    bool sceneIntersectPhoton(const Ray &ray,
                              const std::vector<Sphere> &spheres,
                              const std::vector<Plane> &combinedPlanes,
                              const std::vector<Triangle> &triangles,
                              const std::vector<Bvh::Node> &bvh,
                              Vec3f &hit, Vec3f &N, int &matIdx)
    {
        float closest_t = std::numeric_limits<float>::max();
        float t0 = 0.f;

        for (const auto &sphere : spheres)
        {
            if (!sphere.intersect(ray, t0) || t0 >= closest_t) continue;
            closest_t = t0;
            hit = ray.origin + ray.dir * t0;
            N = (hit - sphere.center).normalize();
            matIdx = sphere.matIdx;
        }

        for (const auto &plane : combinedPlanes)
        {
            Vec3f tempHit;
            if (!plane.intersect(ray, tempHit, t0, closest_t) || t0 >= closest_t) continue;
            closest_t = t0;
            hit = tempHit;
            N = plane.N;
            matIdx = plane.matIdx;
        }

        if (!bvh.empty())
        {
            Vec3f triHit, triN;
            int   triMatIdx;
            float triT;
            if (Bvh::intersect(bvh, triangles, ray, triHit, triN, triMatIdx, triT, closest_t))
            {
                closest_t = triT;
                hit = triHit;
                N = triN;
                matIdx = triMatIdx;
            }
        }
        else
        {
            Vec3f triHit, triN;
            for (const auto &tri : triangles)
            {
                if (!tri.intersect(ray, triHit, triN, t0, closest_t) || t0 >= closest_t) continue;
                closest_t = t0;
                hit = triHit;
                N = triN;
                matIdx = tri.matIdx;
            }
        }

        return closest_t < std::numeric_limits<float>::max();
    }
}

namespace Photon
{
    Map shootCaustic(const Scenes::SceneData &scene,
                     int photonCount,
                     float radius,
                     int maxBounces,
                     uint64_t seed)
    {
        Map map(radius);

        if (photonCount <= 0 || scene.areaLights.empty())
            return map;

        // Build the combined plane list, same as Renderer::render's
        // _planes setup. Light planes first so that when a light is
        // coplanar with a wall the light wins on the t-tie.
        std::vector<Plane> planes;
        for (const auto &L : scene.areaLights)
            if (L.kind == Scenes::AreaLightKind::Plane) planes.push_back(L.plane);
        for (const auto &w : scene.walls) planes.push_back(w);

        float totalLightArea = 0.f;
        for (const auto &L : scene.areaLights) totalLightArea += L.totalArea;
        if (totalLightArea <= 0.f) return map;

        // Photons per-light proportional to area. Per-photon power
        // works out to (pi * totalLightArea / photonCount) * emissive,
        // regardless of which light got picked: pick prob = area_i /
        // totalArea, light flux = pi * area_i * Le_i, photon power =
        // flux / (pickProb * photonCount) = pi * totalArea * Le_i /
        // photonCount.
        const float fluxScale = (float)std::numbers::pi * totalLightArea
                                / (float)photonCount;

        PCG rng(seed);

        PCR_LOG << "Photon::shootCaustic: " << photonCount
                  << " photons, radius=" << radius
                  << ", maxBounces=" << maxBounces << std::endl;

        // Statistics for the start-of-render log: how many photons
        // actually deposited vs. dropped. Useful when a scene has
        // zero specular surfaces (every photon path terminates with
        // hasSpecular=false and the map ends up empty), or when the
        // radius / bounce settings are starving the map.
        int deposited = 0;
        int droppedNonCaustic = 0;
        int droppedHitLight = 0;
        int droppedRR = 0;
        int droppedMissed = 0;

        for (int i = 0; i < photonCount; i++)
        {
            // Pick a light by area.
            float pickTarget = rng.next() * totalLightArea;
            const Scenes::AreaLight *picked = &scene.areaLights.front();
            float cumul = 0.f;
            for (const auto &L : scene.areaLights)
            {
                cumul += L.totalArea;
                if (pickTarget <= cumul) { picked = &L; break; }
            }

            // Sample position + outward normal on the light surface.
            Vec3f startPos, startN, lightEmissive;
            if (picked->kind == Scenes::AreaLightKind::Plane)
            {
                const Plane &p = picked->plane;
                float ru = rng.next();
                float rv = rng.next();
                startPos = p.origin + p.getU() * ru + p.getV() * rv;
                startN = p.N;
                lightEmissive = scene.materials[p.matIdx].emissive;
            }
            else
            {
                // Triangle-set light. Pick a triangle proportional
                // to area (the cumulative-area table was prebuilt
                // by makeTriangleSetLight).
                float rtri = rng.next() * picked->totalArea;
                auto it = std::lower_bound(picked->cumulativeArea.begin(),
                                           picked->cumulativeArea.end(), rtri);
                int triIdx = std::min((int)(it - picked->cumulativeArea.begin()),
                                      (int)picked->triangles.size() - 1);
                const Triangle &tri = picked->triangles[triIdx];

                float r1 = rng.next();
                float r2 = rng.next();
                if (r1 + r2 > 1.f) { r1 = 1.f - r1; r2 = 1.f - r2; }
                startPos = tri.v0 + (tri.v1 - tri.v0) * r1 + (tri.v2 - tri.v0) * r2;
                startN = tri.flatN;
                lightEmissive = scene.materials[tri.matIdx].emissive;
            }

            // Cosine-weighted direction off the light surface.
            Vec3f startDir = cosineHemisphere(startN, rng.next(), rng.next());

            // Push the start point above the light surface so the
            // very first intersect doesn't land back on the light.
            Vec3f photonOrigin = startPos + startN * 1e-3f;
            Ray   photonRay(startDir, photonOrigin);
            Vec3f photonPower = lightEmissive * fluxScale;

            bool  hasSpecular = false;
            bool  pathAlive = true;
            for (int bounce = 0; bounce < maxBounces && pathAlive; bounce++)
            {
                Vec3f hit, N;
                int matIdx = -1;
                if (!sceneIntersectPhoton(photonRay, scene.spheres, planes,
                                          scene.triangles, scene.triangleBvh,
                                          hit, N, matIdx))
                {
                    droppedMissed++;
                    break;
                }

                const Material &material = scene.materials[matIdx];

                bool entering = photonRay.dir.dot(N) < 0.f;
                if (!entering) N = N * -1.f;

                if (material.isEmissive())
                {
                    // Photon ran back into a light (light-to-light
                    // path). No caustic deposit; drop it.
                    droppedHitLight++;
                    break;
                }

                if (material.metallic)
                {
                    // Mirror reflection. Tint the photon by albedo
                    // and continue. Russian roulette on the chained
                    // throughput so geometric series of mirror
                    // bounces terminate.
                    float cosI = -photonRay.dir.dot(N);
                    Vec3f reflectedDir = photonRay.dir + N * (2.f * cosI);
                    photonPower = Vec3f(photonPower[0] * material.albedo[0],
                                        photonPower[1] * material.albedo[1],
                                        photonPower[2] * material.albedo[2]);

                    float pSurv = std::min(0.95f, std::max(
                        photonPower[0],
                        std::max(photonPower[1], photonPower[2])));
                    if (bounce >= 1)
                    {
                        if (rng.next() > pSurv) { droppedRR++; pathAlive = false; break; }
                        photonPower /= pSurv;
                    }

                    Vec3f newOrigin = hit + N * 1e-3f;
                    photonRay = Ray(reflectedDir, newOrigin);
                    hasSpecular = true;
                    continue;
                }

                if (material.transparent)
                {
                    auto b = Optics::dielectricBounce(photonRay.dir, N, hit,
                                                     entering, material.ior, rng.next());
                    photonPower = Vec3f(photonPower[0] * material.albedo[0],
                                        photonPower[1] * material.albedo[1],
                                        photonPower[2] * material.albedo[2]);

                    float pSurv = std::min(0.95f, std::max(
                        photonPower[0],
                        std::max(photonPower[1], photonPower[2])));
                    if (bounce >= 1)
                    {
                        if (rng.next() > pSurv) { droppedRR++; pathAlive = false; break; }
                        photonPower /= pSurv;
                    }

                    photonRay = Ray(b.dir, b.origin);
                    hasSpecular = true;
                    continue;
                }

                // Diffuse hit. Caustic-only policy: deposit a
                // photon iff the path has touched at least one
                // specular surface. Otherwise drop (NEE handles
                // direct light and the eye-path indirect bounces
                // handle diffuse-diffuse already).
                if (hasSpecular)
                {
                    Record rec;
                    rec.position = hit;
                    rec.wi       = photonRay.dir;  // direction of travel at deposit
                    rec.power    = photonPower;
                    map.insert(rec);
                    deposited++;
                }
                else
                {
                    droppedNonCaustic++;
                }
                pathAlive = false;
                break;
            }
        }

        map.build();

        PCR_LOG << "Photon::shootCaustic: deposited=" << deposited
                  << " noncaustic=" << droppedNonCaustic
                  << " hitlight=" << droppedHitLight
                  << " rr=" << droppedRR
                  << " missed=" << droppedMissed
                  << std::endl;

        return map;
    }

    // Spectral caustic shoot. Same structure as shootCaustic but carries
    // per-hero-wavelength power (the map's 4 lambdas) instead of RGB, and
    // handles dispersion: at a dispersive glass surface (cauchyB > 0) a
    // still-polychromatic photon collapses to one hero wavelength (uniform
    // pick, power x4 to stay unbiased) so the refraction uses that
    // wavelength's IOR -- different wavelengths bend differently, which is
    // the rainbow. The deposited photon carries power in its collapsed
    // channel (or all four for a non-dispersive mirror-only caustic).
    Map shootCausticSpectral(const Scenes::SceneData &scene,
                             int photonCount,
                             float radius,
                             int maxBounces,
                             uint64_t seed,
                             const float lambdas[4])
    {
        Map map(radius);
        map.setSpectralLambdas(lambdas);

        if (photonCount <= 0 || scene.areaLights.empty())
            return map;

        std::vector<Plane> planes;
        for (const auto &L : scene.areaLights)
            if (L.kind == Scenes::AreaLightKind::Plane) planes.push_back(L.plane);
        for (const auto &w : scene.walls) planes.push_back(w);

        float totalLightArea = 0.f;
        for (const auto &L : scene.areaLights) totalLightArea += L.totalArea;
        if (totalLightArea <= 0.f) return map;

        const float fluxScale = (float)std::numbers::pi * totalLightArea
                                / (float)photonCount;

        PCG rng(seed);

        int deposited = 0, droppedNonCaustic = 0, droppedHitLight = 0,
            droppedRR = 0, droppedMissed = 0;

        for (int i = 0; i < photonCount; i++)
        {
            float pickTarget = rng.next() * totalLightArea;
            const Scenes::AreaLight *picked = &scene.areaLights.front();
            float cumul = 0.f;
            for (const auto &L : scene.areaLights)
            {
                cumul += L.totalArea;
                if (pickTarget <= cumul) { picked = &L; break; }
            }

            Vec3f startPos, startN;
            int lightMatIdx;
            if (picked->kind == Scenes::AreaLightKind::Plane)
            {
                const Plane &p = picked->plane;
                float ru = rng.next(), rv = rng.next();
                startPos = p.origin + p.getU() * ru + p.getV() * rv;
                startN = p.N;
                lightMatIdx = p.matIdx;
            }
            else
            {
                float rtri = rng.next() * picked->totalArea;
                auto it = std::lower_bound(picked->cumulativeArea.begin(),
                                           picked->cumulativeArea.end(), rtri);
                int triIdx = std::min((int)(it - picked->cumulativeArea.begin()),
                                      (int)picked->triangles.size() - 1);
                const Triangle &tri = picked->triangles[triIdx];
                float r1 = rng.next(), r2 = rng.next();
                if (r1 + r2 > 1.f) { r1 = 1.f - r1; r2 = 1.f - r2; }
                startPos = tri.v0 + (tri.v1 - tri.v0) * r1 + (tri.v2 - tri.v0) * r2;
                startN = tri.flatN;
                lightMatIdx = tri.matIdx;
            }

            Vec3f startDir = cosineHemisphere(startN, rng.next(), rng.next());
            Ray   photonRay(startDir, startPos + startN * 1e-3f);

            const Material &lightMat = scene.materials[lightMatIdx];
            float specPower[4];
            for (int k = 0; k < 4; k++)
                specPower[k] = lightMat.emissiveAt(lambdas[k]) * fluxScale;

            int  monoIdx = -1;     // -1 = still polychromatic (hero-4)
            bool hasSpecular = false;
            bool pathAlive = true;

            auto maxPower = [&]() {
                float m = 0.f;
                for (int k = 0; k < 4; k++) m = std::max(m, specPower[k]);
                return m;
            };

            for (int bounce = 0; bounce < maxBounces && pathAlive; bounce++)
            {
                Vec3f hit, N;
                int matIdx = -1;
                if (!sceneIntersectPhoton(photonRay, scene.spheres, planes,
                                          scene.triangles, scene.triangleBvh,
                                          hit, N, matIdx))
                { droppedMissed++; break; }

                const Material &material = scene.materials[matIdx];
                bool entering = photonRay.dir.dot(N) < 0.f;
                if (!entering) N = N * -1.f;

                if (material.isEmissive()) { droppedHitLight++; break; }

                if (material.metallic)
                {
                    float cosI = -photonRay.dir.dot(N);
                    Vec3f reflectedDir = photonRay.dir + N * (2.f * cosI);
                    for (int k = 0; k < 4; k++)
                        specPower[k] *= material.albedoAt(lambdas[k]);
                    float pSurv = std::min(0.95f, std::max(0.f, maxPower()));
                    if (bounce >= 1)
                    {
                        if (rng.next() > pSurv) { droppedRR++; pathAlive = false; break; }
                        for (int k = 0; k < 4; k++) specPower[k] /= pSurv;
                    }
                    photonRay = Ray(reflectedDir, hit + N * 1e-3f);
                    hasSpecular = true;
                    continue;
                }

                if (material.transparent)
                {
                    float ior = material.ior;
                    if (material.cauchyB > 0.f)
                    {
                        if (monoIdx < 0)
                        {
                            monoIdx = std::min(3, (int)(rng.next() * 4.f));
                            for (int k = 0; k < 4; k++)
                                if (k != monoIdx) specPower[k] = 0.f;
                            specPower[monoIdx] *= 4.f; // 1 / uniform pick prob
                        }
                        ior = Optics::cauchyIor(material.ior, material.cauchyB,
                                                lambdas[monoIdx]);
                    }
                    auto b = Optics::dielectricBounce(photonRay.dir, N, hit,
                                                      entering, ior, rng.next());
                    for (int k = 0; k < 4; k++)
                        specPower[k] *= material.albedoAt(lambdas[k]);
                    float pSurv = std::min(0.95f, std::max(0.f, maxPower()));
                    if (bounce >= 1)
                    {
                        if (rng.next() > pSurv) { droppedRR++; pathAlive = false; break; }
                        for (int k = 0; k < 4; k++) specPower[k] /= pSurv;
                    }
                    photonRay = Ray(b.dir, b.origin);
                    hasSpecular = true;
                    continue;
                }

                if (hasSpecular)
                {
                    Record rec;
                    rec.position = hit;
                    rec.wi       = photonRay.dir;
                    rec.power    = Vec3f(0.f, 0.f, 0.f);
                    for (int k = 0; k < 4; k++) rec.specPower[k] = specPower[k];
                    map.insert(rec);
                    deposited++;
                }
                else droppedNonCaustic++;
                pathAlive = false;
                break;
            }
        }

        map.build();

        PCR_LOG << "Photon::shootCausticSpectral: deposited=" << deposited
                  << " noncaustic=" << droppedNonCaustic
                  << " hitlight=" << droppedHitLight
                  << " rr=" << droppedRR
                  << " missed=" << droppedMissed
                  << std::endl;

        return map;
    }
}
