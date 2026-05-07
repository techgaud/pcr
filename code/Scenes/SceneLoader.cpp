#include "SceneLoader.h"

#include <fstream>
#include <sstream>
#include <unordered_map>

#include "../Includes/json.hpp"
#include "../Includes/Material.h"
#include "../Includes/Plane.h"
#include "../Includes/Sphere.h"
#include "../Includes/Vec3f.h"

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

        std::unordered_map<std::string, Material> parseMaterials(const json &j, const std::string &path)
        {
            auto it = j.find("materials");
            if (it == j.end() || !it->is_object())
                throw SceneLoaderError(path + ": missing or non-object \"materials\" map");

            std::unordered_map<std::string, Material> out;
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
                out.emplace(name, mat);
            }
            if (out.empty())
                throw SceneLoaderError(path + ": \"materials\" map is empty");
            return out;
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

        const Material &resolveMaterial(const std::unordered_map<std::string, Material> &mats,
                                        const std::string &ref,
                                        const std::string &fieldPath,
                                        const std::string &path)
        {
            auto it = mats.find(ref);
            if (it == mats.end())
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

        Sphere parseSphere(const json &p, const std::unordered_map<std::string, Material> &mats,
                           const std::string &fieldPath, const std::string &path)
        {
            auto cit = p.find("center");
            auto rit = p.find("radius");
            if (cit == p.end() || rit == p.end() || !rit->is_number())
                throw SceneLoaderError(path + ": " + fieldPath + " sphere needs center [x,y,z] and numeric radius");
            Vec3f center = vec3FromJson(*cit, fieldPath + ".center");
            float radius = rit->get<float>();
            std::string matRef = requireString(p, "material", fieldPath, path);
            Material mat = resolveMaterial(mats, matRef, fieldPath + ".material", path);
            return Sphere(std::move(center), radius, mat);
        }

        Plane parsePlane(const json &p, const std::unordered_map<std::string, Material> &mats,
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
            Material mat = resolveMaterial(mats, matRef, fieldPath + ".material", path);
            return Plane(origin, u, v, mat);
        }

        // Walk the primitives array. Builds spheres + walls + identifies the
        // single light-flagged primitive (must be a plane).
        void parsePrimitives(const json &j, const std::unordered_map<std::string, Material> &mats,
                             SceneData &out, const std::string &path)
        {
            auto it = j.find("primitives");
            if (it == j.end() || !it->is_array())
                throw SceneLoaderError(path + ": missing or non-array \"primitives\"");

            int lightCount = 0;
            for (std::size_t i = 0; i < it->size(); i++)
            {
                const json &p = (*it)[i];
                std::string fieldPath = "primitives[" + std::to_string(i) + "]";
                if (!p.is_object())
                    throw SceneLoaderError(path + ": " + fieldPath + " must be an object");

                std::string type = requireString(p, "type", fieldPath, path);
                bool isLight = p.value("light", false);

                if (type == "mesh")
                    throw SceneLoaderError(path + ": " + fieldPath +
                                           ": mesh primitives are reserved in the schema "
                                           "but not yet supported by this binary");
                if (type == "sphere")
                {
                    if (isLight)
                        throw SceneLoaderError(path + ": " + fieldPath +
                                               ": light flag is only valid on plane primitives "
                                               "(sphere area-light sampling not implemented)");
                    out.spheres.push_back(parseSphere(p, mats, fieldPath, path));
                }
                else if (type == "plane")
                {
                    Plane pl = parsePlane(p, mats, fieldPath, path);
                    if (isLight)
                    {
                        out.lightSource = pl;
                        lightCount++;
                    }
                    else
                    {
                        out.walls.push_back(std::move(pl));
                    }
                }
                else
                {
                    throw SceneLoaderError(path + ": " + fieldPath +
                                           ": unknown primitive type \"" + type +
                                           "\" (expected sphere, plane, or mesh)");
                }
            }
            if (lightCount == 0)
                throw SceneLoaderError(path + ": exactly one primitive must have \"light\": true; found none");
            if (lightCount > 1)
                throw SceneLoaderError(path + ": exactly one primitive must have \"light\": true; found " +
                                       std::to_string(lightCount));
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

        auto materials = parseMaterials(j, path);
        parsePrimitives(j, materials, out, path);

        return out;
    }
}
