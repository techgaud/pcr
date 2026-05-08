#pragma once
#include <cmath>

#include "Ray.h"
#include "Vec3f.h"

// Material is referenced by index into SceneData::materials, not stored
// by value. The indirection saves substantial memory on mesh-heavy
// scenes (a Triangle was ~600 bytes when it carried a full Material;
// it's ~100 bytes now) and lets the GPU upload skip per-primitive
// material deduplication.
struct Sphere
{
public:
    Sphere(Vec3f c, float r, int materialIdx) : center{c}, matIdx{materialIdx}, _radius{r} {}

    Vec3f center;
    int matIdx;

    bool intersect(const Ray &ray, float &t0) const
    {
        auto cp = center - ray.origin;
        auto rayLen = ray.dir.dot(cp);
        auto tSq = cp.dot(cp) - rayLen * rayLen;

        if (tSq > _radius * _radius)
            return false;

        auto tDist = std::sqrt(_radius * _radius - tSq);
        t0 = rayLen - tDist;
        auto t1 = rayLen + tDist;

        if (t0 < 0)
            t0 = t1;
        if (t0 < 0)
            return false;

        return true;
    }

    float radius() const { return _radius; }

private:
    float _radius;
};
