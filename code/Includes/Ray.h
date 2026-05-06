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

    static Ray genRayFromIntersection(Vec3f &N, Vec3f &o)
    {
        Vec3f T, B;
        createBasis(N, T, B); // effectively a transformation matrix for our random sample

        auto v = sampleDisk(); // random vector (in local space)

        return Ray(T * v[0] + B * v[1] + N * v[2], o); // converts the random sample to world space
    }

private:
    static Vec3f sampleDisk()
    {
        // auto r = NumGen::Epsilon();                 // uniform sampling
        auto r = std::sqrt(NumGen::Epsilon());   // cosine weighted
        auto row = 2.f * (float)std::numbers::pi * NumGen::Epsilon();

        auto x = r * std::cos(row);
        auto y = r * std::sin(row);
        auto z = std::sqrt(1 - x * x - y * y);
        return {x, y, z};
    }

    static void createBasis(const Vec3f &N, Vec3f &T, Vec3f &B)
    {
        auto helper = std::abs(N[0]) <= std::abs(N[1]) ? Vec3f(1, 0, 0) : Vec3f(0, 1, 0);

        T = N.cross(helper).normalize();
        B = N.cross(T);
    }
};