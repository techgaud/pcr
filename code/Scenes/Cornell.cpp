#include "Cornell.h"

namespace Scenes
{
    SceneData makeCornell()
    {
        SceneData s;
        s.name = "cornell";
        s.version = CORNELL_VERSION;

        auto addMat = [&](Material m) -> int {
            int idx = (int)s.materials.size();
            s.materials.push_back(std::move(m));
            return idx;
        };

        int nonemissiveIdx = addMat({Vec3f(0.4f, 0.4f, 0.3f), Vec3f(0.f, 0.f, 0.f)});
        Material emissive{Vec3f(0.f, 0.f, 0.f), Vec3f{1.0f, 0.85f, 0.6f}};
        emissive.emissive *= 80.f;
        int emissiveIdx = addMat(std::move(emissive));

        int creamIdx = addMat({Vec3f(0.74f, 0.74f, 0.64f), Vec3f(0, 0, 0)});
        int redIdx   = addMat({Vec3f(0.63f, 0.06f, 0.05f), Vec3f(0, 0, 0)});
        int greenIdx = addMat({Vec3f(0.13f, 0.45f, 0.1f),  Vec3f(0, 0, 0)});

        s.areaLights.push_back(makePlaneLight(Plane(
            Vec3f{-0.375f, 2.f, -4.25f},
            Vec3f{0.75f, 0, 0},
            Vec3f{0, 0, 0.4f},
            emissiveIdx)));

        s.walls.emplace_back(Vec3f{-2.f, 2.f, -6.f}, Vec3f{4, 0, 0}, Vec3f{0, 0, 7}, creamIdx);    // ceiling
        s.walls.emplace_back(Vec3f{2.f, 2.f, -6.f}, Vec3f{-6, 0, 0}, Vec3f{0, -6, 0}, creamIdx);   // back wall
        s.walls.emplace_back(Vec3f{2.f, -2.f, 0.f}, Vec3f{0, 0, -7}, Vec3f{-4, 0, 0}, creamIdx);   // floor
        s.walls.emplace_back(Vec3f{2.f, -2.f, 0.f}, Vec3f{0, 4, 0}, Vec3f{0, 0, -6}, greenIdx);    // right wall
        s.walls.emplace_back(Vec3f{-2.f, -2.f, -6.f}, Vec3f{0, 4, 0}, Vec3f{0, 0, 6}, redIdx);     // left wall

        s.spheres.push_back(Sphere(Vec3f(0.f, -1.f, -4.5f), 0.75f, nonemissiveIdx));

        populateSpectra(s);
        return s;
    }
}
