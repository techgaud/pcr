#pragma once
#include <cmath>

#include "Ray.h"
#include "Vec3f.h"
#include "Material.h"

struct Sphere
{
public:
    Sphere(Vec3f &&c, float r, Material &m) : center{c}, _radius{r}, material{m} {}

    Vec3f center;
    Material material;

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

private:
    float _radius;
};