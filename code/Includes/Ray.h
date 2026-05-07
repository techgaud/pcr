#pragma once

#include <math.h>
#include <numbers>

#include "Vec3f.h"
#include "NumGen.h"

class Ray
{
public:
    Ray() {}
    Ray(Vec3f &dir, Vec3f &origin) : dir{dir}, origin{origin} {}
    Ray(Vec3f &&dir, Vec3f &origin) : dir{dir}, origin{origin} {}

    Vec3f dir;
    Vec3f origin;

    // Random-sample variant. Caller-blind: pulls two PRNG values internally.
    static Ray genRayFromIntersection(Vec3f &N, Vec3f &o)
    {
        return genRayFromIntersection(N, o, NumGen::Epsilon(), NumGen::Epsilon());
    }

    // Stratified-friendly variant. Caller supplies two values in [0,1).
    // Used by stratified sampling so the renderer can place the (r1,r2)
    // pairs on a jittered grid rather than fully random.
    static Ray genRayFromIntersection(Vec3f &N, Vec3f &o, float r1, float r2)
    {
        Vec3f T, B;
        createBasis(N, T, B);
        auto v = sampleDiskFrom(r1, r2);
        return Ray(T * v[0] + B * v[1] + N * v[2], o);
    }

private:
    static Vec3f sampleDiskFrom(float r1, float r2)
    {
        auto r = std::sqrt(r1);                            // cosine weighted
        auto phi = 2.f * (float)std::numbers::pi * r2;
        auto x = r * std::cos(phi);
        auto y = r * std::sin(phi);
        auto z = std::sqrt(std::max(0.f, 1.f - x * x - y * y));
        return {x, y, z};
    }

    static void createBasis(const Vec3f &N, Vec3f &T, Vec3f &B)
    {
        auto helper = std::abs(N[0]) <= std::abs(N[1]) ? Vec3f(1, 0, 0) : Vec3f(0, 1, 0);

        T = N.cross(helper).normalize();
        B = N.cross(T);
    }
};