// cornell-spheres: the standard Cornell Box but with a cluster of small
// spheres scattered around the floor in addition to the centered big one.
// Sphere positions inherited from a commented-out vector in the original
// pre-refactor Main.cpp (commit 1f89fa9), restored here as a distinct scene.
// All spheres are non-emissive cream/grey to keep the area-light sampling
// clean (the renderer only does explicit sampling of the single Plane light).

#include "CornellSpheres.h"

namespace Scenes
{
    SceneData makeCornellSpheres()
    {
        SceneData s;
        s.name = "cornell-spheres";
        s.version = CORNELL_SPHERES_VERSION;

        Material nonemissive{Vec3f(0.4f, 0.4f, 0.3f), Vec3f(0.f, 0.f, 0.f)};
        Material emissive{Vec3f(0.f, 0.f, 0.f), Vec3f{1.0f, 0.85f, 0.6f}};
        emissive.emissive *= 80.f;

        s.areaLights.push_back(makePlaneLight(Plane(
            Vec3f{-0.375f, 2.f, -4.25f},
            Vec3f{0.75f, 0, 0},
            Vec3f{0, 0, 0.4f},
            emissive)));

        // Cornell-box walls (same as base cornell scene).
        Material cream{Vec3f(0.74f, 0.74f, 0.64f), Vec3f(0, 0, 0)};
        Material red{Vec3f(0.63f, 0.06f, 0.05f), Vec3f(0, 0, 0)};
        Material green{Vec3f(0.13f, 0.45f, 0.1f), Vec3f(0, 0, 0)};

        s.walls.emplace_back(Vec3f{-2.f, 2.f, -6.f}, Vec3f{4, 0, 0}, Vec3f{0, 0, 7}, cream);    // ceiling
        s.walls.emplace_back(Vec3f{2.f, 2.f, -6.f}, Vec3f{-6, 0, 0}, Vec3f{0, -6, 0}, cream);   // back wall
        s.walls.emplace_back(Vec3f{2.f, -2.f, 0.f}, Vec3f{0, 0, -7}, Vec3f{-4, 0, 0}, cream);   // floor
        s.walls.emplace_back(Vec3f{2.f, -2.f, 0.f}, Vec3f{0, 4, 0}, Vec3f{0, 0, -6}, green);    // right wall
        s.walls.emplace_back(Vec3f{-2.f, -2.f, -6.f}, Vec3f{0, 4, 0}, Vec3f{0, 0, 6}, red);     // left wall

        // Center big sphere (same as base cornell), then four smaller spheres
        // scattered around it on the floor — recovered from the original
        // commented-out vector in the pre-refactor Main.cpp.
        s.spheres.push_back(Sphere(Vec3f(0.f, -1.f, -4.5f),     0.75f, nonemissive));
        s.spheres.push_back(Sphere(Vec3f(-0.75f, -1.5f, -3.75f), 0.25f, nonemissive));
        s.spheres.push_back(Sphere(Vec3f(-0.75f, -1.45f, -4.75f), 0.65f, nonemissive));
        s.spheres.push_back(Sphere(Vec3f(0.3f, -1.6f, -4.45f),   0.4f,  nonemissive));
        s.spheres.push_back(Sphere(Vec3f(0.75f, -0.75f, -4.75f), 0.35f, nonemissive));

        return s;
    }
}
