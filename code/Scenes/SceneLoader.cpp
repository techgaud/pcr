#include "SceneLoader.h"

#include <fstream>
#include <sstream>
#include <unordered_map>

#include "../Includes/json.hpp"
#include "../Includes/Material.h"
#include "../Includes/Plane.h"
#include "../Includes/SpdLoader.h"
#include "../Includes/Sphere.h"
#include "../Includes/Triangle.h"
#include "../Includes/Vec3f.h"
#include "MeshLoader.h"

namespace Scenes
{
    namespace
    {
        using nlohmann::json;

        // Schema version this loader understands. Files declaring a higher
        // major version are rejected; higher-minor is permitted (forward-
        // compat for additive schema changes).
        constexpr int kSchemaMajor = 1;

        // Convert a byte offset into a 1-indexed line/column pair by
        // re-scanning the source text. nlohmann's parse_error gives byte
        // position; humans read line:col.
        std::pair<int, int> byteToLineCol(const std::string &text, std::size_t byte)
        {
            int line = 1;
            int col = 1;
            std::size_t end = std::min(byte, text.size());
            for (std::size_t i = 0; i < end; i++)
            {
                if (text[i] == '\n') { line++; col = 1; }
                else                  { col++; }
            }
            return {line, col};
        }

        std::string readFile(const std::string &path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in)
                throw SceneLoaderError("Cannot open scene file: " + path);
            std::ostringstream ss;
            ss << in.rdbuf();
            return ss.str();
        }

        Vec3f vec3FromJson(const json &j, const std::string &fieldPath)
        {
            if (!j.is_array() || j.size() != 3)
                throw SceneLoaderError(fieldPath + ": expected [x, y, z] array of 3 numbers");
            try
            {
                return Vec3f(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
            }
            catch (const json::exception &e)
            {
                throw SceneLoaderError(fieldPath + ": " + e.what());
            }
        }

        // Parse the schema field. Format is "<major>.<minor>" as a string.
        // We accept only major == kSchemaMajor; reject higher major outright,
        // tolerate higher minor (forward-compatible additive changes).
        void validateSchema(const json &j, const std::string &path)
        {
            auto it = j.find("schema");
            if (it == j.end())
                throw SceneLoaderError(path + ": missing required \"schema\" field");
            if (!it->is_string())
                throw SceneLoaderError(path + ": \"schema\" must be a string like \"1.0\"");
            std::string s = it->get<std::string>();
            int major = 0, minor = 0;
            if (std::sscanf(s.c_str(), "%d.%d", &major, &minor) != 2)
                throw SceneLoaderError(path + ": malformed schema \"" + s + "\" (expected major.minor)");
            if (major != kSchemaMajor)
                throw SceneLoaderError(path + ": schema major version " + std::to_string(major) +
                                       " is not supported by this binary (expected " +
                                       std::to_string(kSchemaMajor) + ".x)");
        }

        // Parse materials block, pushing each into out.materials and
        // recording its name -> index mapping in nameToIdx.
        void parseMaterials(const json &j, const std::string &path,
                            SceneData &out,
                            std::unordered_map<std::string, int> &nameToIdx)
        {
            auto it = j.find("materials");
            if (it == j.end() || !it->is_object())
                throw SceneLoaderError(path + ": missing or non-object \"materials\" map");

            for (auto kv = it->begin(); kv != it->end(); ++kv)
            {
                const std::string &name = kv.key();
                const json &m = kv.value();
                if (!m.is_object())
                    throw SceneLoaderError(path + ": material \"" + name + "\" must be an object");

                Material mat;
                if (auto ait = m.find("albedo"); ait != m.end())
                    mat.albedo = vec3FromJson(*ait, "materials." + name + ".albedo");
                if (auto eit = m.find("emissive"); eit != m.end())
                    mat.emissive = vec3FromJson(*eit, "materials." + name + ".emissive");

                // Tabulated SPDs override the RGB upsampler. albedo_spd
                // is a name like "cornell/white-paint" resolved through
                // the spd/ search path; the file must exist or scene
                // load fails. Mutually exclusive with albedo (the loader
                // accepts both but the SPD wins; an explicit RGB albedo
                // becomes redundant when an SPD is also provided, so we
                // reject the combination to keep scene files honest).
                if (auto sit = m.find("albedo_spd"); sit != m.end())
                {
                    if (!sit->is_string())
                        throw SceneLoaderError(path + ": materials." + name + ".albedo_spd must be a string");
                    if (m.find("albedo") != m.end())
                        throw SceneLoaderError(path + ": materials." + name + " has both albedo and albedo_spd; pick one");
                    std::string spdName = sit->get<std::string>();
                    std::string err, resolvedPath;
                    if (!SpdLoader::loadSpdByName(spdName, {}, mat.tabulatedAlbedo, &resolvedPath, &err))
                        throw SceneLoaderError(path + ": materials." + name + ".albedo_spd: " + err);
                    mat.useTabulatedAlbedo = true;
                }
                if (auto sit = m.find("emissive_spd"); sit != m.end())
                {
                    if (!sit->is_string())
                        throw SceneLoaderError(path + ": materials." + name + ".emissive_spd must be a string");
                    if (m.find("emissive") != m.end())
                        throw SceneLoaderError(path + ": materials." + name + " has both emissive and emissive_spd; pick one");
                    std::string spdName = sit->get<std::string>();
                    std::string err, resolvedPath;
                    if (!SpdLoader::loadSpdByName(spdName, {}, mat.tabulatedEmissive, &resolvedPath, &err))
                        throw SceneLoaderError(path + ": materials." + name + ".emissive_spd: " + err);
                    mat.useTabulatedEmissive = true;
                    // Optional brightness multiplier baked into the
                    // stored spectrum so the renderer doesn't carry a
                    // separate scale field. Cornell's published light
                    // SPD is in arbitrary radiance units; scenes apply
                    // a multiplier to land at scene-appropriate
                    // brightness (cornell-spec uses ~5x).
                    if (auto cit = m.find("emissive_scale"); cit != m.end())
                    {
                        if (!cit->is_number())
                            throw SceneLoaderError(path + ": materials." + name + ".emissive_scale must be a number");
                        float scale = cit->get<float>();
                        for (int i = 0; i < Spectrum::kSamples; i++)
                            mat.tabulatedEmissive[i] *= scale;
                    }
                }
                // Specular extensions: metallic = perfect mirror,
                // transparent = glass dielectric with Fresnel + Snell.
                // ior is the index of refraction; only matters when
                // transparent is true.
                if (auto it = m.find("metallic"); it != m.end())
                {
                    if (!it->is_boolean())
                        throw SceneLoaderError(path + ": materials." + name + ".metallic must be a boolean");
                    mat.metallic = it->get<bool>();
                }
                if (auto it = m.find("transparent"); it != m.end())
                {
                    if (!it->is_boolean())
                        throw SceneLoaderError(path + ": materials." + name + ".transparent must be a boolean");
                    mat.transparent = it->get<bool>();
                }
                if (auto it = m.find("ior"); it != m.end())
                {
                    if (!it->is_number())
                        throw SceneLoaderError(path + ": materials." + name + ".ior must be a number");
                    mat.ior = it->get<float>();
                }
                if (auto it = m.find("cauchyB"); it != m.end())
                {
                    if (!it->is_number())
                        throw SceneLoaderError(path + ": materials." + name + ".cauchyB must be a number");
                    mat.cauchyB = it->get<float>();
                }
                if (mat.metallic && mat.transparent)
                    throw SceneLoaderError(path + ": materials." + name + " cannot be both metallic and transparent");
                int idx = (int)out.materials.size();
                out.materials.push_back(std::move(mat));
                nameToIdx[name] = idx;
            }
            if (out.materials.empty())
                throw SceneLoaderError(path + ": \"materials\" map is empty");
        }

        Camera parseCamera(const json &j, const std::string &path)
        {
            Camera c;
            auto it = j.find("camera");
            if (it == j.end())
                return c; // defaults: position {0,0,0}, fov 65
            if (!it->is_object())
                throw SceneLoaderError(path + ": \"camera\" must be an object");
            if (auto pit = it->find("position"); pit != it->end())
                c.position = vec3FromJson(*pit, "camera.position");
            if (auto fit = it->find("fov"); fit != it->end())
            {
                if (!fit->is_number())
                    throw SceneLoaderError(path + ": camera.fov must be a number");
                c.fov = fit->get<float>();
            }
            return c;
        }

        int resolveMaterialIdx(const std::unordered_map<std::string, int> &nameToIdx,
                               const std::string &ref,
                               const std::string &fieldPath,
                               const std::string &path)
        {
            auto it = nameToIdx.find(ref);
            if (it == nameToIdx.end())
                throw SceneLoaderError(path + ": " + fieldPath + " references unknown material \"" + ref + "\"");
            return it->second;
        }

        std::string requireString(const json &p, const char *key,
                                  const std::string &fieldPath, const std::string &path)
        {
            auto it = p.find(key);
            if (it == p.end() || !it->is_string())
                throw SceneLoaderError(path + ": " + fieldPath + " missing string \"" + key + "\"");
            return it->get<std::string>();
        }

        Sphere parseSphere(const json &p,
                           const std::unordered_map<std::string, int> &nameToIdx,
                           const std::string &fieldPath, const std::string &path)
        {
            auto cit = p.find("center");
            auto rit = p.find("radius");
            if (cit == p.end() || rit == p.end() || !rit->is_number())
                throw SceneLoaderError(path + ": " + fieldPath + " sphere needs center [x,y,z] and numeric radius");
            Vec3f center = vec3FromJson(*cit, fieldPath + ".center");
            float radius = rit->get<float>();
            std::string matRef = requireString(p, "material", fieldPath, path);
            int matIdx = resolveMaterialIdx(nameToIdx, matRef, fieldPath + ".material", path);
            return Sphere(center, radius, matIdx);
        }

        // Triangle parser. v0/v1/v2 required. Optional smooth-shading normals
        // n0/n1/n2 (all three together or none. partial smooth normals are
        // an authoring mistake we'd rather flag than silently zero out).
        Triangle parseTriangle(const json &p,
                               const std::unordered_map<std::string, int> &nameToIdx,
                               const std::string &fieldPath, const std::string &path)
        {
            auto v0it = p.find("v0");
            auto v1it = p.find("v1");
            auto v2it = p.find("v2");
            if (v0it == p.end() || v1it == p.end() || v2it == p.end())
                throw SceneLoaderError(path + ": " + fieldPath +
                                       " triangle needs v0, v1, v2 vertices");
            Vec3f v0 = vec3FromJson(*v0it, fieldPath + ".v0");
            Vec3f v1 = vec3FromJson(*v1it, fieldPath + ".v1");
            Vec3f v2 = vec3FromJson(*v2it, fieldPath + ".v2");

            std::string matRef = requireString(p, "material", fieldPath, path);
            int matIdx = resolveMaterialIdx(nameToIdx, matRef, fieldPath + ".material", path);

            auto n0it = p.find("n0");
            auto n1it = p.find("n1");
            auto n2it = p.find("n2");
            int normalCount = (n0it != p.end()) + (n1it != p.end()) + (n2it != p.end());
            if (normalCount == 0)
                return Triangle(v0, v1, v2, matIdx);
            if (normalCount != 3)
                throw SceneLoaderError(path + ": " + fieldPath +
                                       " triangle normals require all of n0, n1, n2 or none");
            Vec3f n0 = vec3FromJson(*n0it, fieldPath + ".n0");
            Vec3f n1 = vec3FromJson(*n1it, fieldPath + ".n1");
            Vec3f n2 = vec3FromJson(*n2it, fieldPath + ".n2");
            return Triangle(v0, v1, v2, n0, n1, n2, matIdx);
        }

        Plane parsePlane(const json &p,
                         const std::unordered_map<std::string, int> &nameToIdx,
                         const std::string &fieldPath, const std::string &path)
        {
            auto oit = p.find("origin");
            auto uit = p.find("u");
            auto vit = p.find("v");
            if (oit == p.end() || uit == p.end() || vit == p.end())
                throw SceneLoaderError(path + ": " + fieldPath + " plane needs origin, u, v");
            Vec3f origin = vec3FromJson(*oit, fieldPath + ".origin");
            Vec3f u = vec3FromJson(*uit, fieldPath + ".u");
            Vec3f v = vec3FromJson(*vit, fieldPath + ".v");
            std::string matRef = requireString(p, "material", fieldPath, path);
            int matIdx = resolveMaterialIdx(nameToIdx, matRef, fieldPath + ".material", path);
            return Plane(origin, u, v, matIdx);
        }

        // Mesh primitive parser. Reads file/transform/material override,
        // delegates the OBJ parse + transform + triangle generation to
        // MeshLoader, and appends the resulting triangles to SceneData.
        // Resolves "file" relative to the JSON scene's directory so a JSON
        // can reference ../models/foo.obj predictably regardless of cwd.
        //
        // Returns the (begin, count) range of newly-appended triangles in
        // out.triangles so the caller can build a TriangleSet AreaLight
        // from them when the mesh is flagged as a light.
        std::pair<int, int> parseMesh(const json &p,
                       std::unordered_map<std::string, int> &nameToIdx,
                       SceneData &out,
                       const std::string &fieldPath,
                       const std::string &scenePath)
        {
            std::string fileRel = requireString(p, "file", fieldPath, scenePath);
            std::filesystem::path objPath = std::filesystem::path(scenePath).parent_path() / fileRel;

            MeshOptions opts;
            if (auto it = p.find("position"); it != p.end())
                opts.position = vec3FromJson(*it, fieldPath + ".position");
            if (auto it = p.find("rotation"); it != p.end())
                opts.rotation = vec3FromJson(*it, fieldPath + ".rotation");
            if (auto it = p.find("scale"); it != p.end())
            {
                if (it->is_number())
                {
                    float s = it->get<float>();
                    opts.scale = Vec3f(s, s, s);
                }
                else
                {
                    opts.scale = vec3FromJson(*it, fieldPath + ".scale");
                }
            }
            if (auto it = p.find("material"); it != p.end() && it->is_string())
            {
                opts.materialOverride = it->get<std::string>();
                opts.hasMaterialOverride = true;
            }
            if (auto it = p.find("smooth"); it != p.end())
            {
                if (!it->is_boolean())
                    throw SceneLoaderError(scenePath + ": " + fieldPath +
                                           ".smooth must be a boolean");
                opts.smoothOverride = it->get<bool>();
                opts.hasSmoothOverride = true;
            }

            auto tris = loadMesh(objPath.string(), opts, out.materials, nameToIdx);
            int begin = (int)out.triangles.size();
            int count = (int)tris.size();
            out.triangles.insert(out.triangles.end(),
                                 std::make_move_iterator(tris.begin()),
                                 std::make_move_iterator(tris.end()));
            return {begin, count};
        }

        // Walk the primitives array. Each primitive contributes geometry
        // (spheres / walls / triangles) and may additionally contribute an
        // AreaLight when flagged "light": true. The "exactly one light" rule
        // changed to "at least one light" in phase 3b. multiple plane,
        // triangle, and mesh lights can coexist now, and the renderer picks
        // among them proportional to area.
        //
        // Light primitives also contribute to scene geometry for ray
        // intersection: a plane light is added to walls (so it's hittable
        // and can occlude other lights), a triangle light is added to
        // triangles, mesh lights' triangles are already in triangles. The
        // AreaLight stores a copy of the geometry for sampling, decoupled
        // from the BVH builder's triangle permutation.
        void parsePrimitives(const json &j,
                             std::unordered_map<std::string, int> &nameToIdx,
                             SceneData &out, const std::string &path)
        {
            auto it = j.find("primitives");
            if (it == j.end() || !it->is_array())
                throw SceneLoaderError(path + ": missing or non-array \"primitives\"");

            for (std::size_t i = 0; i < it->size(); i++)
            {
                const json &p = (*it)[i];
                std::string fieldPath = "primitives[" + std::to_string(i) + "]";
                if (!p.is_object())
                    throw SceneLoaderError(path + ": " + fieldPath + " must be an object");

                std::string type = requireString(p, "type", fieldPath, path);
                bool isLight = p.value("light", false);

                if (type == "sphere")
                {
                    if (isLight)
                        throw SceneLoaderError(path + ": " + fieldPath +
                                               ": light flag is only valid on plane, triangle, "
                                               "or mesh primitives (sphere area-light sampling "
                                               "not implemented)");
                    out.spheres.push_back(parseSphere(p, nameToIdx, fieldPath, path));
                }
                else if (type == "triangle")
                {
                    Triangle tri = parseTriangle(p, nameToIdx, fieldPath, path);
                    if (isLight)
                    {
                        std::vector<Triangle> setTris;
                        setTris.push_back(tri);
                        out.areaLights.push_back(makeTriangleSetLight(std::move(setTris)));
                    }
                    out.triangles.push_back(std::move(tri));
                }
                else if (type == "plane")
                {
                    Plane pl = parsePlane(p, nameToIdx, fieldPath, path);
                    if (isLight)
                    {
                        // Light planes go ONLY to areaLights. the renderer
                        // promotes them to _planes ahead of walls so coplanar
                        // ties (e.g. a ceiling-cutout light) are won by the
                        // light. Adding them to walls too would lose the tie
                        // to whichever wall is iterated first.
                        out.areaLights.push_back(makePlaneLight(pl));
                    }
                    else
                    {
                        out.walls.push_back(std::move(pl));
                    }
                }
                else if (type == "mesh")
                {
                    auto [begin, count] = parseMesh(p, nameToIdx, out, fieldPath, path);
                    if (isLight)
                    {
                        // Copy this mesh's triangle range out into the light.
                        // The BVH builder is allowed to permute out.triangles
                        // afterward without disturbing the light's copy.
                        std::vector<Triangle> setTris(out.triangles.begin() + begin,
                                                       out.triangles.begin() + begin + count);
                        out.areaLights.push_back(makeTriangleSetLight(std::move(setTris)));
                    }
                }
                else
                {
                    throw SceneLoaderError(path + ": " + fieldPath +
                                           ": unknown primitive type \"" + type +
                                           "\" (expected sphere, plane, triangle, or mesh)");
                }
            }
            if (out.areaLights.empty())
                throw SceneLoaderError(path +
                    ": at least one primitive must have \"light\": true; found none");
        }
    } // namespace

    SceneData loadSceneFromFile(const std::string &path)
    {
        std::string text = readFile(path);
        json j;
        try
        {
            // Args: input, callback, allow_exceptions, ignore_comments.
            // Comment support is the reason we go through the parse() overload
            // rather than operator>>.
            j = json::parse(text, nullptr, true, true);
        }
        catch (const json::parse_error &e)
        {
            auto [line, col] = byteToLineCol(text, e.byte);
            throw SceneLoaderError(path + ":" + std::to_string(line) + ":" + std::to_string(col) +
                                   ": JSON parse error: " + e.what());
        }

        validateSchema(j, path);

        SceneData out;
        out.name = requireString(j, "name", "(root)", path);
        out.version = requireString(j, "version", "(root)", path);
        out.camera = parseCamera(j, path);

        std::unordered_map<std::string, int> nameToIdx;
        parseMaterials(j, path, out, nameToIdx);
        parsePrimitives(j, nameToIdx, out, path);

        // Build the BVH now so renders don't pay the cost (and so the GPU
        // upload path in phase 4 can ship pre-built nodes verbatim). The
        // builder permutes out.triangles in place into leaf order; from this
        // point on the order in `out.triangles` is what the BVH expects.
        out.triangleBvh = Bvh::build(out.triangles);

        // Spectral upsample after BVH build because BVH permutes
        // out.triangles, and we want every Material on every primitive
        // (post-permutation) to carry its fitted spectrum.
        populateSpectra(out);

        return out;
    }
}
