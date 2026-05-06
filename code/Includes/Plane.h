#pragma once

#include "Vec3f.h"
#include "Ray.h"
#include "Material.h"

class Plane
{
public:
    Plane() {}
    Plane(Vec3f o, Vec3f l, Vec3f w, Material mat) : material{mat}, origin{o}, u{l}, v{w}
    {
        N = u.cross(v).normalize();
    }
    Plane(Vec3f &&o, Vec3f &&l, Vec3f &&w, Material &&mat) : material{mat}, origin{o}, u{l}, v{w}
    {
        auto cross = u.cross(v);
        area = cross.length();
        N = cross.normalize();
    }

    Material material;
    Vec3f origin;
    Vec3f N;

private:
    Vec3f u;
    Vec3f v;
    float area = 0;

public:
    inline float getArea() { return area == 0 ? u.cross(v).length() : area; }

    bool intersect(const Ray &ray, Vec3f &hit, float &t0, const float &closest_t) const
    {
        constexpr float EPS = 1e-6;
        float denom = ray.dir.dot(N);
        if (std::fabs(denom) <= EPS)
            return false;

        t0 = (origin - ray.origin).dot(N) / denom;
        if (t0 <= EPS || t0 > closest_t)
            return false;

        Vec3f tempHit = ray.origin + ray.dir * t0;
        Vec3f localHit = tempHit - origin;

        float s = localHit.dot(u) / u.dot(u);
        float q = localHit.dot(v) / v.dot(v);
        if (s > 1 || s < 0 || q > 1 || q < 0)
            return false;

        hit = tempHit;
        return true;
    }

    Vec3f getVecToPlaneFromHit(const Vec3f &hit) const
    {
        return (origin + u * NumGen::Epsilon() + v * NumGen::Epsilon()) - hit;
    }
};