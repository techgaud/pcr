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
    // The viewing direction is fixed (-Z forward, +Y up) — the renderer
    // hasn't grown a lookAt/up basis yet. Adding those fields here later
    // is non-breaking; renderers can default them when absent.
    struct Camera
    {
        Vec3f position{0, 0, 0};
        float fov = 65.f;
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
        Plane lightSource;
    };
}
