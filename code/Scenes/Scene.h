#pragma once

#include <string>
#include <vector>

#include "../Bvh/Bvh.h"
#include "../Includes/Sphere.h"
#include "../Includes/Plane.h"
#include "../Includes/Triangle.h"
#include "../Includes/Vec3f.h"

namespace Scenes
{
    // Camera lives with the scene, not the renderer. Each scene declares
    // where the camera sits and what FOV it sees, so a single binary can
    // render scenes with different framings without recompiling.
    //
    // The viewing direction is fixed (-Z forward, +Y up). the renderer
    // hasn't grown a lookAt/up basis yet. Adding those fields here later
    // is non-breaking; renderers can default them when absent.
    struct Camera
    {
        Vec3f position{0, 0, 0};
        float fov = 65.f;
    };

    // An area light. Either a single Plane (parallelogram) or a set of
    // emissive Triangles (for mesh / hand-rolled triangle lights). Sampling
    // is uniform over surface area; the renderer picks one light from the
    // scene's list proportional to totalArea, then samples within it.
    //
    // The triangles in a TriangleSet light are *copies* of the underlying
    // mesh triangles, not indices into SceneData::triangles. This avoids
    // the BVH builder (which permutes SceneData::triangles in place)
    // invalidating any references the lights held. ~6 MB of duplication
    // for the bunny is a fair trade for not having to thread the BVH
    // permutation through the lights.
    enum class AreaLightKind { Plane, TriangleSet };

    struct AreaLight
    {
        AreaLightKind kind = AreaLightKind::Plane;

        // kind == Plane:
        Plane plane;

        // kind == TriangleSet:
        std::vector<Triangle> triangles;
        // Cumulative area of triangles[0..i], inclusive. cumulativeArea.back()
        // equals totalArea. Used for binary-search picking by area.
        std::vector<float> cumulativeArea;

        // Cached. For Plane: plane.getArea(). For TriangleSet: sum.
        float totalArea = 0.f;
    };

    struct SceneData
    {
        std::string name;
        std::string version;
        Camera camera;
        std::vector<Sphere> spheres;
        std::vector<Plane> walls;
        std::vector<Triangle> triangles;
        // BVH over `triangles`. Built at scene-load time after triangles are
        // populated. Empty when triangles is empty (the BVH builder permutes
        // `triangles` in place, so `triangles` is in BVH-leaf order after build).
        std::vector<Bvh::Node> triangleBvh;

        // At least one area light. Hardcoded scenes ship one Plane light to
        // match historical behavior; JSON scenes can mark planes, triangles,
        // and meshes with light:true and end up with multiple lights here.
        std::vector<AreaLight> areaLights;
    };
}

namespace Scenes
{
    // Build an AreaLight from a single Plane primitive. Caches totalArea.
    inline AreaLight makePlaneLight(const Plane &p)
    {
        AreaLight L;
        L.kind = AreaLightKind::Plane;
        L.plane = p;
        L.totalArea = p.getArea();
        return L;
    }

    // Walk every Material referenced by the scene and populate its
    // spectral counterparts (albedoSpectrum, emissiveSpectrum) from
    // the RGB values. Called at the end of every scene-load path.
    // Idempotent. Cheap (~10 microseconds per material via Newton-
    // Raphson, and Cornell-class scenes have under ten materials).
    inline void populateSpectra(SceneData &s)
    {
        for (auto &sphere : s.spheres) sphere.material.populateSpectra();
        for (auto &wall : s.walls) wall.material.populateSpectra();
        for (auto &tri : s.triangles) tri.material.populateSpectra();
        for (auto &L : s.areaLights)
        {
            if (L.kind == AreaLightKind::Plane)
            {
                L.plane.material.populateSpectra();
            }
            else
            {
                for (auto &tri : L.triangles) tri.material.populateSpectra();
            }
        }
    }

    // Build an AreaLight from a contiguous list of triangles. Caches per-tri
    // cumulative area for binary-search sampling.
    inline AreaLight makeTriangleSetLight(std::vector<Triangle> tris)
    {
        AreaLight L;
        L.kind = AreaLightKind::TriangleSet;
        L.triangles = std::move(tris);
        L.cumulativeArea.reserve(L.triangles.size());
        float acc = 0.f;
        for (const auto &t : L.triangles)
        {
            // Triangle area = 0.5 * |e1 x e2|.
            Vec3f e1 = t.v1 - t.v0;
            Vec3f e2 = t.v2 - t.v0;
            Vec3f cross = e1.cross(e2);
            float area = 0.5f * cross.length();
            acc += area;
            L.cumulativeArea.push_back(acc);
        }
        L.totalArea = acc;
        return L;
    }
}
