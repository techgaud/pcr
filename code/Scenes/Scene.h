#pragma once

#include <string>
#include <vector>

#include "../Includes/Sphere.h"
#include "../Includes/Plane.h"

namespace Scenes
{
    struct SceneData
    {
        std::string name;
        std::string version;
        std::vector<Sphere> spheres;
        std::vector<Plane> walls;
        Plane lightSource;
    };
}
