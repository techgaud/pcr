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

    // Loads the OBJ at `objPath`, applies transform from `opts`, returns the
    // resulting triangles. Triangle materials reference names in `materials`;
    // any MTL materials referenced by the OBJ are added to `materials` under
    // their MTL-given names (overwriting any existing entry with that name —
    // JSON authors who care about the resolution rule should override per-
    // mesh via opts.materialOverride or pre-define materials in JSON to
    // shadow MTL-side names).
    //
    // Throws Scenes::SceneLoaderError on file-not-found, parse errors, or
    // unresolvable material references.
    std::vector<Triangle> loadMesh(const std::string &objPath,
                                   const MeshOptions &opts,
                                   std::unordered_map<std::string, Material> &materials);
}
