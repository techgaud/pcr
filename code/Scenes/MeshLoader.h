#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../Includes/Material.h"
#include "../Includes/Triangle.h"
#include "../Includes/Vec3f.h"

namespace Scenes
{
    // Per-mesh transform + material override settings parsed from the JSON.
    struct MeshOptions
    {
        Vec3f position{0, 0, 0};
        Vec3f rotation{0, 0, 0}; // Euler degrees, applied X then Y then Z
        Vec3f scale{1, 1, 1};

        // If set, every triangle in the mesh uses this material name (must
        // exist in the registry already, or be added by MTL parsing). If
        // unset, OBJ usemtl directives drive per-face material selection.
        std::string materialOverride;
        bool hasMaterialOverride = false;

        // OBJ files often ship vertex normals; when present the mesh shades
        // smooth by default. Set to false to force flat shading even if the
        // OBJ has normals.
        bool smoothOverride = true;
        bool hasSmoothOverride = false;
    };

    // Loads the OBJ at `objPath`, applies transform from `opts`, returns
    // the resulting triangles with their matIdx fields populated. Any MTL
    // materials referenced by the OBJ are pushed onto `materials` (the
    // SceneData registry vector) and added to `nameToIdx` for subsequent
    // resolution. Throws Scenes::SceneLoaderError on file-not-found, parse
    // errors, or unresolvable material references.
    std::vector<Triangle> loadMesh(const std::string &objPath,
                                   const MeshOptions &opts,
                                   std::vector<Material> &materials,
                                   std::unordered_map<std::string, int> &nameToIdx);
}
