// MeshLoader: OBJ + MTL ingestion via tinyobjloader. Translates each face
// into a Triangle (CCW -> Triangle ctor's vertex order), applies the per-
// mesh transform, and feeds MTL diffuse/emissive colors into the shared
// material registry.
//
// Phase 3a constraints:
//   - Triangulates non-triangle faces via tinyobjloader's built-in (the
//     library's default config triangulates).
//   - map_Kd and other texture maps in MTL are ignored with a warning;
//     phase 5 lands texture support.
//   - light:true on a mesh primitive is rejected upstream in SceneLoader;
//     phase 3b adds triangle area-light sampling so meshes can be lights.
//   - Non-uniform scale is supported but normals get only rotated (not
//     inverse-transposed scaled) — sheared meshes will have subtly wrong
//     shading. Uniform scale is correct.

#include "MeshLoader.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "SceneLoader.h" // for SceneLoaderError

#define TINYOBJLOADER_IMPLEMENTATION
// tinyobjloader has unused static helpers that fire -Wunused-function on
// gcc/clang. They're an implementation detail of the upstream library, not
// something we want to vendor-edit, so silence them locally.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../Includes/tiny_obj_loader.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace Scenes
{
    namespace
    {
        constexpr float kPi = 3.14159265358979323846f;

        // Apply Euler-XYZ rotation to a 3-vector. Angles are degrees.
        Vec3f rotateXYZ(Vec3f v, const Vec3f &eulerDeg)
        {
            float rx = eulerDeg[0] * (kPi / 180.f);
            float ry = eulerDeg[1] * (kPi / 180.f);
            float rz = eulerDeg[2] * (kPi / 180.f);

            // Rotate around X
            if (rx != 0.f)
            {
                float c = std::cos(rx), s = std::sin(rx);
                float y = v[1] * c - v[2] * s;
                float z = v[1] * s + v[2] * c;
                v[1] = y; v[2] = z;
            }
            // Rotate around Y
            if (ry != 0.f)
            {
                float c = std::cos(ry), s = std::sin(ry);
                float x = v[0] * c + v[2] * s;
                float z = -v[0] * s + v[2] * c;
                v[0] = x; v[2] = z;
            }
            // Rotate around Z
            if (rz != 0.f)
            {
                float c = std::cos(rz), s = std::sin(rz);
                float x = v[0] * c - v[1] * s;
                float y = v[0] * s + v[1] * c;
                v[0] = x; v[1] = y;
            }
            return v;
        }

        Vec3f transformPosition(Vec3f v, const MeshOptions &opts)
        {
            v[0] *= opts.scale[0]; v[1] *= opts.scale[1]; v[2] *= opts.scale[2];
            v = rotateXYZ(v, opts.rotation);
            v[0] += opts.position[0]; v[1] += opts.position[1]; v[2] += opts.position[2];
            return v;
        }

        // Normals: rotation only. For uniform scale this is correct; for
        // non-uniform scale a proper inverse-transpose would be needed and
        // is left for a later phase if a real use case turns up.
        Vec3f transformNormal(Vec3f n, const MeshOptions &opts)
        {
            return rotateXYZ(n, opts.rotation);
        }
    } // namespace

    std::vector<Triangle> loadMesh(const std::string &objPath,
                                   const MeshOptions &opts,
                                   std::unordered_map<std::string, Material> &materials)
    {
        std::vector<Triangle> result;

        if (!std::filesystem::exists(objPath))
            throw SceneLoaderError("Mesh file not found: " + objPath);

        // tinyobjloader: ParseFromFile triangulates by default. mtl_search_path
        // points at the OBJ's parent dir so MTL-relative texture paths
        // (which we don't use anyway) and `mtllib foo.mtl` directives resolve.
        tinyobj::ObjReaderConfig cfg;
        cfg.mtl_search_path = std::filesystem::path(objPath).parent_path().string();
        cfg.triangulate = true;
        cfg.vertex_color = false;

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(objPath, cfg))
        {
            std::string err = reader.Error();
            throw SceneLoaderError("Failed to parse OBJ " + objPath + ": " +
                                   (err.empty() ? "(no detail)" : err));
        }
        if (!reader.Warning().empty())
            std::fprintf(stderr, "warning: parsing %s: %s",
                         objPath.c_str(), reader.Warning().c_str());

        const auto &attrib = reader.GetAttrib();
        const auto &shapes = reader.GetShapes();
        const auto &mtls = reader.GetMaterials();

        // Bring MTL materials into the shared registry. tinyobjloader gives
        // `name`, `diffuse[3]` (Kd), `emission[3]` (Ke). We map the rest of
        // MTL (specular, shininess, illum) to nothing because the renderer
        // is Lambert-only.
        for (const auto &m : mtls)
        {
            if (!m.diffuse_texname.empty() ||
                !m.emissive_texname.empty() ||
                !m.specular_texname.empty())
            {
                std::fprintf(stderr,
                             "warning: %s material '%s' references textures (map_Kd/etc) — "
                             "ignored; texture support lands in phase 5\n",
                             objPath.c_str(), m.name.c_str());
            }
            Material mat;
            mat.albedo = Vec3f(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
            mat.emissive = Vec3f(m.emission[0], m.emission[1], m.emission[2]);
            materials[m.name] = mat;
        }

        // Resolve override material once if set.
        Material overrideMat;
        if (opts.hasMaterialOverride)
        {
            auto it = materials.find(opts.materialOverride);
            if (it == materials.end())
                throw SceneLoaderError("Mesh material override '" + opts.materialOverride +
                                       "' not in registry (declare it in JSON \"materials\" or "
                                       "via MTL before referencing)");
            overrideMat = it->second;
        }

        // Walk shapes -> faces. tinyobjloader has triangulated for us so each
        // face is exactly 3 indices. A "vertex" here is a (pos_idx, normal_idx,
        // texcoord_idx) triple; we read only pos and normal.
        for (const auto &shape : shapes)
        {
            const auto &mesh = shape.mesh;
            size_t indexOffset = 0;
            for (size_t f = 0; f < mesh.num_face_vertices.size(); f++)
            {
                int fv = (int)mesh.num_face_vertices[f]; // always 3 with cfg.triangulate
                if (fv != 3) { indexOffset += fv; continue; } // be safe

                tinyobj::index_t idx0 = mesh.indices[indexOffset + 0];
                tinyobj::index_t idx1 = mesh.indices[indexOffset + 1];
                tinyobj::index_t idx2 = mesh.indices[indexOffset + 2];

                auto getPos = [&](int vi) {
                    return Vec3f(attrib.vertices[3 * vi + 0],
                                 attrib.vertices[3 * vi + 1],
                                 attrib.vertices[3 * vi + 2]);
                };
                auto getNorm = [&](int ni) {
                    return Vec3f(attrib.normals[3 * ni + 0],
                                 attrib.normals[3 * ni + 1],
                                 attrib.normals[3 * ni + 2]);
                };

                Vec3f v0 = transformPosition(getPos(idx0.vertex_index), opts);
                Vec3f v1 = transformPosition(getPos(idx1.vertex_index), opts);
                Vec3f v2 = transformPosition(getPos(idx2.vertex_index), opts);

                bool haveNormals = idx0.normal_index >= 0 &&
                                   idx1.normal_index >= 0 &&
                                   idx2.normal_index >= 0;
                bool wantSmooth = haveNormals;
                if (opts.hasSmoothOverride)
                    wantSmooth = haveNormals && opts.smoothOverride;

                // Resolve material: per-face material from MTL unless an
                // override is set. tinyobjloader assigns -1 when no MTL.
                Material faceMat = overrideMat;
                if (!opts.hasMaterialOverride)
                {
                    int mid = mesh.material_ids[f];
                    if (mid >= 0 && mid < (int)mtls.size())
                    {
                        auto it = materials.find(mtls[mid].name);
                        if (it != materials.end())
                            faceMat = it->second;
                    }
                    // If no MTL material at all and no override, the JSON
                    // author hasn't told us what color the mesh is. Fall
                    // back to mid-grey rather than aborting — they'll see
                    // it and add an override. Loud silence.
                    if (mtls.empty() && !opts.hasMaterialOverride)
                    {
                        faceMat = Material{};
                        faceMat.albedo = Vec3f(0.7f, 0.7f, 0.7f);
                    }
                }

                if (wantSmooth)
                {
                    Vec3f n0 = transformNormal(getNorm(idx0.normal_index), opts).normalize();
                    Vec3f n1 = transformNormal(getNorm(idx1.normal_index), opts).normalize();
                    Vec3f n2 = transformNormal(getNorm(idx2.normal_index), opts).normalize();
                    result.emplace_back(v0, v1, v2, n0, n1, n2, faceMat);
                }
                else
                {
                    result.emplace_back(v0, v1, v2, faceMat);
                }

                indexOffset += fv;
            }
        }

        return result;
    }
}
