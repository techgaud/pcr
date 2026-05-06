#pragma once

#include "Vec3f.h"

struct Material
{
    Material() {}
    Material(Vec3f &&a) : albedo{a}, emissive{0, 0, 0} {}
    Material(Vec3f &&a, Vec3f &&e) : albedo{a}, emissive{e} {}

    Vec3f albedo{0, 0, 0};
    Vec3f emissive{0, 0, 0};

    bool isEmissive()
    {
        return emissive[0] > 0 || emissive[1] > 0 || emissive[2] > 0;
    }
};