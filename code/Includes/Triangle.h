#pragma once

#include <cmath>

#include "Vec3f.h"
#include "Ray.h"

// Triangle primitive with Moller-Trumbore ray intersect. Material is
// referenced by index into SceneData::materials (see Sphere note).
// Optional per-vertex normals enable smooth (interpolated) shading;
// without them the triangle uses the flat geometric normal (cross
// product of two edges, computed once at construction).
class Triangle
{
public:
    // Flat-shaded constructor: normal is the geometric normal from edges.
    Triangle(Vec3f a, Vec3f b, Vec3f c, int materialIdx)
        : v0{a}, v1{b}, v2{c},
          n0{0,0,0}, n1{0,0,0}, n2{0,0,0},
          smooth{false},
          matIdx{materialIdx}
    {
        Vec3f e1 = v1 - v0;
        Vec3f e2 = v2 - v0;
        flatN = e1.cross(e2).normalize();
    }

    // Smooth-shaded constructor: per-vertex normals get barycentric-interpolated
    // at the hit point. Caller is responsible for passing already-normalized
    // normals; we don't re-normalize inside the ctor.
    Triangle(Vec3f a, Vec3f b, Vec3f c,
             Vec3f na, Vec3f nb, Vec3f nc, int materialIdx)
        : v0{a}, v1{b}, v2{c},
          n0{na}, n1{nb}, n2{nc},
          smooth{true},
          matIdx{materialIdx}
    {
        Vec3f e1 = v1 - v0;
        Vec3f e2 = v2 - v0;
        flatN = e1.cross(e2).normalize();
    }

    Vec3f v0, v1, v2;
    Vec3f n0, n1, n2;
    Vec3f flatN;
    bool smooth;
    int matIdx;

    // Moller-Trumbore. Returns true if the ray hits the triangle within
    // (EPS, closest_t). On hit, fills t0 (param distance), hit (world-space
    // hit position), and N (shading normal. interpolated when smooth, flat
    // otherwise; the renderer flips it to face the ray afterward).
    bool intersect(const Ray &ray, Vec3f &hit, Vec3f &N,
                   float &t0, const float &closest_t) const
    {
        constexpr float EPS = 1e-6f;
        Vec3f e1 = v1 - v0;
        Vec3f e2 = v2 - v0;
        Vec3f pvec = ray.dir.cross(e2);
        float det = e1.dot(pvec);
        if (std::fabs(det) < EPS) return false;

        float invDet = 1.f / det;
        Vec3f tvec = ray.origin - v0;
        float u = tvec.dot(pvec) * invDet;
        if (u < 0.f || u > 1.f) return false;

        Vec3f qvec = tvec.cross(e1);
        float v = ray.dir.dot(qvec) * invDet;
        if (v < 0.f || u + v > 1.f) return false;

        float t = e2.dot(qvec) * invDet;
        if (t <= EPS || t >= closest_t) return false;

        t0 = t;
        hit = ray.origin + ray.dir * t;
        if (smooth)
        {
            float w = 1.f - u - v;
            // Interpolated barycentric normal. Re-normalize because barycentric
            // averaging of unit vectors does not, in general, yield a unit vector.
            Vec3f interp{n0[0]*w + n1[0]*u + n2[0]*v,
                         n0[1]*w + n1[1]*u + n2[1]*v,
                         n0[2]*w + n1[2]*u + n2[2]*v};
            N = interp.normalize();
        }
        else
        {
            N = flatN;
        }
        return true;
    }
};
