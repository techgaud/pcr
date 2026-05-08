// cornell-large-light: same Cornell Box as the base scene, but the ceiling
// area light is ~2.7x larger (2.0 wide instead of 0.75, same depth). Larger
// light means softer shadows, brighter overall ambient, and faster
// convergence at any sample count. Useful for A/B testing direct-lighting
// noise vs the small-light variant.

#include "CornellLargeLight.h"

namespace Scenes
{
    SceneData makeCornellLargeLight()
    {
        SceneData s;
        s.name = "cornell-large-light";
        s.version = CORNELL_LARGE_LIGHT_VERSION;

        Material nonemissive{Vec3f(0.4f, 0.4f, 0.3f), Vec3f(0.f, 0.f, 0.f)};
        Material emissive{Vec3f(0.f, 0.f, 0.f), Vec3f{1.0f, 0.85f, 0.6f}};
        emissive.emissive *= 80.f;

        // Larger ceiling light: 2.0 wide x 0.4 deep instead of 0.75 x 0.4.
        s.areaLights.push_back(makePlaneLight(Plane(
            Vec3f{-1.f, 2.f, -4.25f},
            Vec3f{2.f, 0, 0},
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

        s.spheres.push_back(Sphere(Vec3f(0.f, -1.f, -4.5f), 0.75f, nonemissive));

        populateSpectra(s);
        return s;
    }
}
