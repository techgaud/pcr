// GpuRenderer — OpenGL 4.3 compute shader path tracer.
//
// Architecture:
//   - Scene -> SSBOs (one each for spheres, planes, materials)
//   - Compute shader operates on a vec4 image2D, one work-item per pixel
//   - Per-pixel: loop `samples` paths of up to `depth` bounces each, with
//     `shadowSamples` direct-light shadow rays per non-emissive hit
//   - Tone-map (Reinhard) inside the shader, output 8-bit RGBA to the image
//   - Dispatch in horizontal row strips so cancel and progress are responsive
//   - Read pixels back, save as PNG via lodepng (same path as CPU)
//
// Threading:
//   - render() is called on the GUI's render-worker thread
//   - Worker calls glfwMakeContextCurrent(_sharedContext) on first render() so
//     all GL calls hit the dedicated worker context (the GUI keeps drawing on
//     its own context; resources are shared via GLFW's share=mainWindow)
//
// Why no glad/glew: we only need ~40 GL entry points and GLFW already provides
// glfwGetProcAddress. ~80 lines of typedefs + loaders is cheaper than another
// vendored library.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "Gpu/GpuRenderer.h"
#include "Includes/lodepng.h"
#include "Includes/Denoise.h"

#include <cmath>

namespace fs = std::filesystem;

// ---------- Minimal GL types + constants ---------------------------------
//
// We define these locally to avoid pulling in <GL/gl.h>, which pulls in
// <windows.h> on Windows and we already had to fight that fight with NOMINMAX.
// Values are from the OpenGL spec and are stable.

using GLenum   = unsigned int;
using GLbitfield = unsigned int;
using GLuint   = unsigned int;
using GLint    = int;
using GLsizei  = int;
using GLfloat  = float;
using GLchar   = char;
using GLboolean = unsigned char;
using GLsizeiptr = ptrdiff_t;
using GLintptr = ptrdiff_t;

#define GL_FALSE 0
#define GL_TRUE 1
#define GL_NO_ERROR 0
#define GL_COMPUTE_SHADER 0x91B9
#define GL_LINK_STATUS 0x8B82
#define GL_COMPILE_STATUS 0x8B81
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_TEXTURE_2D 0x0DE1
#define GL_RGBA8 0x8058
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_LINEAR 0x2601
#define GL_NEAREST 0x2600
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_WRITE_ONLY 0x88B9
#define GL_READ_ONLY 0x88B8
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#define GL_ALL_BARRIER_BITS 0xFFFFFFFF

#if defined(_WIN32)
#define GL_APIENTRY __stdcall
#else
#define GL_APIENTRY
#endif

// ---------- GL function pointers (loaded via glfwGetProcAddress) ---------

#define PCR_GL_FUNCS(X)                                                          \
    X(GLuint, CreateShader,         GLenum)                                       \
    X(void,   ShaderSource,         GLuint, GLsizei, const GLchar *const *, const GLint *) \
    X(void,   CompileShader,        GLuint)                                       \
    X(void,   GetShaderiv,          GLuint, GLenum, GLint *)                      \
    X(void,   GetShaderInfoLog,     GLuint, GLsizei, GLsizei *, GLchar *)         \
    X(GLuint, CreateProgram,        void)                                         \
    X(void,   AttachShader,         GLuint, GLuint)                               \
    X(void,   LinkProgram,          GLuint)                                       \
    X(void,   GetProgramiv,         GLuint, GLenum, GLint *)                      \
    X(void,   GetProgramInfoLog,    GLuint, GLsizei, GLsizei *, GLchar *)         \
    X(void,   DeleteShader,         GLuint)                                       \
    X(void,   DeleteProgram,        GLuint)                                       \
    X(void,   UseProgram,           GLuint)                                       \
    X(GLint,  GetUniformLocation,   GLuint, const GLchar *)                       \
    X(void,   Uniform1i,            GLint, GLint)                                 \
    X(void,   Uniform1f,            GLint, GLfloat)                               \
    X(void,   Uniform2i,            GLint, GLint, GLint)                          \
    X(void,   GenBuffers,           GLsizei, GLuint *)                            \
    X(void,   DeleteBuffers,        GLsizei, const GLuint *)                      \
    X(void,   BindBuffer,           GLenum, GLuint)                               \
    X(void,   BufferData,           GLenum, GLsizeiptr, const void *, GLenum)     \
    X(void,   BindBufferBase,       GLenum, GLuint, GLuint)                       \
    X(void,   GenTextures,          GLsizei, GLuint *)                            \
    X(void,   DeleteTextures,       GLsizei, const GLuint *)                      \
    X(void,   BindTexture,          GLenum, GLuint)                               \
    X(void,   TexImage2D,           GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *) \
    X(void,   TexParameteri,        GLenum, GLenum, GLint)                        \
    X(void,   GetTexImage,          GLenum, GLint, GLenum, GLenum, void *)        \
    X(void,   BindImageTexture,     GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum) \
    X(void,   DispatchCompute,      GLuint, GLuint, GLuint)                       \
    X(void,   MemoryBarrier,        GLbitfield)                                   \
    X(void,   Finish,               void)                                         \
    X(GLenum, GetError,             void)

#define PCR_DECLARE_GL_FN(ret, name, ...)                                         \
    using PFN_pcr_gl##name = ret(GL_APIENTRY *)(__VA_ARGS__);                     \
    static PFN_pcr_gl##name pcr_gl##name = nullptr;
PCR_GL_FUNCS(PCR_DECLARE_GL_FN)
#undef PCR_DECLARE_GL_FN

// Use the gl-prefixed names inside the file for readability.
#define glCreateShader        pcr_glCreateShader
#define glShaderSource        pcr_glShaderSource
#define glCompileShader       pcr_glCompileShader
#define glGetShaderiv         pcr_glGetShaderiv
#define glGetShaderInfoLog    pcr_glGetShaderInfoLog
#define glCreateProgram       pcr_glCreateProgram
#define glAttachShader        pcr_glAttachShader
#define glLinkProgram         pcr_glLinkProgram
#define glGetProgramiv        pcr_glGetProgramiv
#define glGetProgramInfoLog   pcr_glGetProgramInfoLog
#define glDeleteShader        pcr_glDeleteShader
#define glDeleteProgram       pcr_glDeleteProgram
#define glUseProgram          pcr_glUseProgram
#define glGetUniformLocation  pcr_glGetUniformLocation
#define glUniform1i           pcr_glUniform1i
#define glUniform1f           pcr_glUniform1f
#define glUniform2i           pcr_glUniform2i
#define glGenBuffers          pcr_glGenBuffers
#define glDeleteBuffers       pcr_glDeleteBuffers
#define glBindBuffer          pcr_glBindBuffer
#define glBufferData          pcr_glBufferData
#define glBindBufferBase      pcr_glBindBufferBase
#define glGenTextures         pcr_glGenTextures
#define glDeleteTextures      pcr_glDeleteTextures
#define glBindTexture         pcr_glBindTexture
#define glTexImage2D          pcr_glTexImage2D
#define glTexParameteri       pcr_glTexParameteri
#define glGetTexImage         pcr_glGetTexImage
#define glBindImageTexture    pcr_glBindImageTexture
#define glDispatchCompute     pcr_glDispatchCompute
#define glMemoryBarrier       pcr_glMemoryBarrier
#define glFinish              pcr_glFinish
#define glGetError            pcr_glGetError

static bool loadGlFunctions()
{
    bool ok = true;
#define PCR_LOAD_GL_FN(ret, name, ...)                                            \
    pcr_gl##name = (PFN_pcr_gl##name)glfwGetProcAddress("gl" #name);              \
    if (!pcr_gl##name) {                                                          \
        std::fprintf(stderr, "GL function gl" #name " not available\n");          \
        ok = false;                                                               \
    }
    PCR_GL_FUNCS(PCR_LOAD_GL_FN)
#undef PCR_LOAD_GL_FN
    return ok;
}

// ---------- Embedded compute shader source --------------------------------

static const char *kComputeShaderSrc = R"GLSL(
#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba8, binding = 0) uniform writeonly image2D uOutput;

uniform int   uWidth;
uniform int   uHeight;
uniform float uFov;
uniform int   uDepth;
uniform int   uSamples;
uniform int   uShadowSamples;
uniform int   uSphereCount;
uniform int   uPlaneCount;
uniform int   uLightPlaneIdx;
uniform int   uYOffset;
uniform int   uYEnd;
uniform int   uFrameSeed;
uniform int   uUseMIS;        // 0/1
uniform int   uUseRussian;    // 0/1
uniform int   uUseStratified; // 0/1
uniform int   uStrata;        // round(sqrt(uSamples)) when stratified, else 0

const float PI = 3.14159265358979323846;

struct GpuSphere {
    vec4 center_radius;  // xyz=center, w=radius
    int  matIdx;
    int  _pad0, _pad1, _pad2;
};

struct GpuPlane {
    vec4 origin;          // xyz=origin
    vec4 u;               // xyz=u
    vec4 v;               // xyz=v
    vec4 normal_area;     // xyz=N, w=area
    int  matIdx;
    int  _pad0, _pad1, _pad2;
};

struct GpuMaterial {
    vec4 albedo;          // rgb
    vec4 emissive;        // rgb
};

layout(std430, binding = 1) buffer Spheres   { GpuSphere   spheres[];   };
layout(std430, binding = 2) buffer Planes    { GpuPlane    planes[];    };
layout(std430, binding = 3) buffer Materials { GpuMaterial materials[]; };

// PCG random number generator. Each pixel gets its own seeded state.
uint pcg(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rand(inout uint seed) {
    return float(pcg(seed)) / 4294967295.0;
}

vec3 sampleHemisphereFrom(vec3 N, float r1, float r2) {
    float r = sqrt(r1);
    float phi = 2.0 * PI * r2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - x*x - y*y));
    vec3 helper = abs(N.x) <= abs(N.y) ? vec3(1, 0, 0) : vec3(0, 1, 0);
    vec3 T = normalize(cross(N, helper));
    vec3 B = cross(N, T);
    return T * x + B * y + N * z;
}

vec3 sampleHemisphere(vec3 N, inout uint seed) {
    return sampleHemisphereFrom(N, rand(seed), rand(seed));
}

bool intersectSphere(vec3 ro, vec3 rd, vec3 center, float radius, out float t) {
    vec3 cp = center - ro;
    float rayLen = dot(rd, cp);
    float tSq = dot(cp, cp) - rayLen * rayLen;
    if (tSq > radius * radius) return false;
    float tDist = sqrt(radius * radius - tSq);
    float t0 = rayLen - tDist;
    float t1 = rayLen + tDist;
    if (t0 < 0.0) t0 = t1;
    if (t0 < 0.0) return false;
    t = t0;
    return true;
}

bool intersectPlane(vec3 ro, vec3 rd, GpuPlane p, out float t, out vec3 hitOut) {
    const float EPS = 1e-6;
    vec3 N = p.normal_area.xyz;
    float denom = dot(rd, N);
    if (abs(denom) <= EPS) return false;
    float tt = dot(p.origin.xyz - ro, N) / denom;
    if (tt <= EPS) return false;
    vec3 tempHit = ro + rd * tt;
    vec3 localHit = tempHit - p.origin.xyz;
    float s = dot(localHit, p.u.xyz) / dot(p.u.xyz, p.u.xyz);
    float q = dot(localHit, p.v.xyz) / dot(p.v.xyz, p.v.xyz);
    if (s < 0.0 || s > 1.0 || q < 0.0 || q > 1.0) return false;
    t = tt;
    hitOut = tempHit;
    return true;
}

bool sceneIntersect(vec3 ro, vec3 rd, out vec3 hit, out vec3 N, out int matIdx) {
    float closest = 1e30;
    bool found = false;

    for (int i = 0; i < uSphereCount; i++) {
        float t;
        if (intersectSphere(ro, rd, spheres[i].center_radius.xyz, spheres[i].center_radius.w, t)) {
            if (t < closest) {
                closest = t;
                hit = ro + rd * t;
                N = normalize(hit - spheres[i].center_radius.xyz);
                matIdx = spheres[i].matIdx;
                found = true;
            }
        }
    }
    for (int i = 0; i < uPlaneCount; i++) {
        float t;
        vec3 ph;
        if (intersectPlane(ro, rd, planes[i], t, ph)) {
            if (t < closest) {
                closest = t;
                hit = ph;
                N = planes[i].normal_area.xyz;
                matIdx = planes[i].matIdx;
                found = true;
            }
        }
    }
    return found;
}

vec3 tracePath(vec2 pix, float pr1, float pr2, inout uint seed) {
    float aspect = float(uWidth) / float(uHeight);
    float scale = tan(PI / 180.0 * 0.5 * uFov);
    float x = ((2.0 * (pix.x + 0.5) / float(uWidth)) - 1.0) * scale * aspect;
    float y = -((2.0 * (pix.y + 0.5) / float(uHeight)) - 1.0) * scale;
    vec3 rd = normalize(vec3(x, y, -1.0));
    vec3 ro = vec3(0.0);

    vec3 throughput = vec3(1.0);
    vec3 radiance = vec3(0.0);

    GpuPlane light = planes[uLightPlaneIdx];
    vec3 lightEmissive = materials[light.matIdx].emissive.rgb;

    // pr1, pr2 only seed the FIRST indirect bounce when stratified is on; later
    // bounces fall back to PCG. Stratifying every bounce would require N^depth
    // strata, which isn't worth the bookkeeping.
    bool firstBounce = true;

    for (int bounce = 0; bounce < uDepth; bounce++) {
        vec3 hit, N;
        int matIdx;
        if (!sceneIntersect(ro, rd, hit, N, matIdx)) break;

        if (dot(rd, N) > 0.0) N = -N;

        GpuMaterial mat = materials[matIdx];

        if (any(greaterThan(mat.emissive.rgb, vec3(0.0)))) {
            radiance += throughput * mat.emissive.rgb;
            break;
        }

        // Direct lighting
        vec3 directLo = vec3(0.0);
        for (int s = 0; s < uShadowSamples; s++) {
            vec3 lpos = light.origin.xyz + light.u.xyz * rand(seed) + light.v.xyz * rand(seed);
            vec3 Li = lpos - hit;
            vec3 wi = normalize(Li);
            float cosTheta = max(0.0, dot(wi, N));
            float lightDist2 = dot(Li, Li);
            vec3 shadowOrigin = (cosTheta <= 0.0) ? hit - N * 1e-3 : hit + N * 1e-3;

            vec3 sh, sN;
            int sMat;
            bool occluded = false;
            if (sceneIntersect(shadowOrigin, wi, sh, sN, sMat)) {
                vec3 d = sh - shadowOrigin;
                float occluderDist2 = dot(d, d);
                if (occluderDist2 < lightDist2 - 1e-3) {
                    GpuMaterial om = materials[sMat];
                    if (!any(greaterThan(om.emissive.rgb, vec3(0.0)))) occluded = true;
                }
            }

            if (!occluded) {
                float cosLight = max(0.0, dot(light.normal_area.xyz, -wi));
                float G = (cosTheta * cosLight) / lightDist2;
                vec3 directContrib = (mat.albedo.rgb / PI) * lightEmissive * G * light.normal_area.w;

                // Partial MIS: down-weight by balance heuristic on the
                // light-side. Same caveat as the CPU renderer about the
                // BRDF-side weighting on emissive returns being omitted.
                if (uUseMIS != 0 && cosLight > 1e-6) {
                    float pdfLight = lightDist2 / (cosLight * light.normal_area.w);
                    float pdfBrdf  = cosTheta / PI;
                    float w = (pdfLight * pdfLight) /
                              (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                    directContrib *= w;
                }
                directLo += directContrib;
            }
        }
        directLo /= float(uShadowSamples);
        radiance += throughput * directLo;

        // Russian roulette: at bounce >= 1, terminate with prob (1 - p) where
        // p reflects surface reflectance. Survivors get scaled to compensate.
        if (uUseRussian != 0 && bounce >= 1) {
            float p = clamp(max(max(mat.albedo.r, mat.albedo.g), mat.albedo.b), 0.05, 0.95);
            if (rand(seed) > p) break;
            throughput /= p;
        }

        // Indirect bounce: cosine-weighted hemisphere. Use stratified r1/r2
        // for the first bounce only; subsequent bounces use plain random.
        vec3 newDir;
        if (uUseStratified != 0 && firstBounce) {
            newDir = sampleHemisphereFrom(N, pr1, pr2);
        } else {
            newDir = sampleHemisphere(N, seed);
        }
        firstBounce = false;

        ro = hit + N * 1e-3;
        rd = newDir;
        throughput *= mat.albedo.rgb;
    }
    return radiance;
}

vec3 reinhard(vec3 c) {
    return c / (c + vec3(1.0));
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.x, int(gl_GlobalInvocationID.y) + uYOffset);
    if (pix.x >= uWidth || pix.y >= uYEnd || pix.y >= uHeight) return;

    uint seed = uint(pix.x) * 1973u + uint(pix.y) * 9277u + uint(uFrameSeed) * 26699u;

    vec3 accum = vec3(0.0);
    for (int s = 0; s < uSamples; s++) {
        // Stratified sample placement on a sqrt(N) x sqrt(N) jittered grid
        // for the first bounce of each path.
        float r1, r2;
        if (uUseStratified != 0 && uStrata > 0) {
            int sx = s % uStrata;
            int sy = (s / uStrata) % uStrata;
            r1 = (float(sx) + rand(seed)) / float(uStrata);
            r2 = (float(sy) + rand(seed)) / float(uStrata);
        } else {
            r1 = rand(seed);
            r2 = rand(seed);
        }
        accum += tracePath(vec2(pix), r1, r2, seed);
    }
    accum /= float(uSamples);
    accum = reinhard(accum);

    imageStore(uOutput, pix, vec4(accum, 1.0));
}
)GLSL";

// ---------- Helpers -------------------------------------------------------

namespace
{
    bool checkGl(const char *where)
    {
        GLenum err = glGetError();
        if (err == GL_NO_ERROR) return true;
        std::fprintf(stderr, "GL error 0x%x at %s\n", err, where);
        return false;
    }

    GLuint compileComputeShader(const char *src)
    {
        GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            GLint len = 0;
            glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
            std::vector<char> log(len > 0 ? (size_t)len : (size_t)1, 0);
            glGetShaderInfoLog(sh, len, nullptr, log.data());
            std::fprintf(stderr, "Compute shader compile failed:\n%s\n", log.data());
            glDeleteShader(sh);
            return 0;
        }
        GLuint prog = glCreateProgram();
        glAttachShader(prog, sh);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            GLint len = 0;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
            std::vector<char> log(len > 0 ? (size_t)len : (size_t)1, 0);
            glGetProgramInfoLog(prog, len, nullptr, log.data());
            std::fprintf(stderr, "Program link failed:\n%s\n", log.data());
            glDeleteShader(sh);
            glDeleteProgram(prog);
            return 0;
        }
        glDeleteShader(sh);
        return prog;
    }

    // GPU-friendly layouts. std430 packs vec3 awkwardly so we use vec4 +
    // explicit padding everywhere to keep alignment predictable.
    struct GpuSphere
    {
        float center[4];   // xyz=center, w=radius
        int   matIdx;
        int   _pad[3];
    };
    struct GpuPlane
    {
        float origin[4];
        float u[4];
        float v[4];
        float normal_area[4]; // xyz=N, w=area
        int   matIdx;
        int   _pad[3];
    };
    struct GpuMaterial
    {
        float albedo[4];
        float emissive[4];
    };

    std::string formatTimestamp(bool utc)
    {
        std::time_t now = std::time(nullptr);
        std::tm tm{};
#ifdef _MSC_VER
        if (utc) gmtime_s(&tm, &now);
        else     localtime_s(&tm, &now);
#else
        if (utc) gmtime_r(&now, &tm);
        else     localtime_r(&now, &tm);
#endif
        char buf[64];
        if (utc) std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S-UTC", &tm);
        else     std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S-%Z", &tm);
        return buf;
    }
}

// ---------- GpuRenderer ---------------------------------------------------

GpuRenderer::GpuRenderer(int width, int height, float fov,
                         int depth, int samples, int shadowSamples,
                         GLFWwindow *sharedContext)
    : _width{width}, _height{height}, _fov{fov},
      _maxDepth{depth}, _samples{samples}, _shadowSamples{shadowSamples},
      _sharedContext{sharedContext}
{
}

GpuRenderer::~GpuRenderer()
{
    // We don't tear down GL state in the destructor because the GL context
    // probably isn't current on the calling thread. The OS reclaims it on
    // process exit; the cost is negligible for a single-render lifetime.
}

bool GpuRenderer::initGL()
{
    if (_initialized) return true;

    if (!loadGlFunctions())
    {
        std::cerr << "GpuRenderer: failed to load GL functions" << std::endl;
        return false;
    }

    _program = compileComputeShader(kComputeShaderSrc);
    if (!_program) return false;

    glGenTextures(1, &_outputTex);
    glBindTexture(GL_TEXTURE_2D, _outputTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    if (!checkGl("texture allocation")) return false;

    glGenBuffers(1, &_sphereSSBO);
    glGenBuffers(1, &_planeSSBO);
    glGenBuffers(1, &_materialSSBO);

    _initialized = true;
    return true;
}

void GpuRenderer::uploadScene(const Scenes::SceneData &scene, int &outLightIdx)
{
    // Build a flat material table: one material per scene material seen.
    // For simplicity, materials are deduplicated by pointer identity (we
    // copy them in sequence and reference by index). Walls and the sphere
    // each contribute one material; the light is its own material.
    std::vector<GpuMaterial> mats;
    auto addMaterial = [&](const Material &m) -> int {
        GpuMaterial gm{};
        gm.albedo[0] = m.albedo[0];
        gm.albedo[1] = m.albedo[1];
        gm.albedo[2] = m.albedo[2];
        gm.emissive[0] = m.emissive[0];
        gm.emissive[1] = m.emissive[1];
        gm.emissive[2] = m.emissive[2];
        mats.push_back(gm);
        return (int)mats.size() - 1;
    };

    std::vector<GpuSphere> gpuSpheres;
    for (const auto &s : scene.spheres)
    {
        int mi = addMaterial(s.material);
        GpuSphere gs{};
        gs.center[0] = s.center[0];
        gs.center[1] = s.center[1];
        gs.center[2] = s.center[2];
        gs.center[3] = s.radius();
        gs.matIdx = mi;
        gpuSpheres.push_back(gs);
    }

    std::vector<GpuPlane> gpuPlanes;

    auto addPlane = [&](const Plane &p, int matIdx) {
        const Vec3f &u = p.getU();
        const Vec3f &v = p.getV();
        GpuPlane gp{};
        gp.origin[0] = p.origin[0];
        gp.origin[1] = p.origin[1];
        gp.origin[2] = p.origin[2];
        gp.u[0] = u[0]; gp.u[1] = u[1]; gp.u[2] = u[2];
        gp.v[0] = v[0]; gp.v[1] = v[1]; gp.v[2] = v[2];
        gp.normal_area[0] = p.N[0];
        gp.normal_area[1] = p.N[1];
        gp.normal_area[2] = p.N[2];
        gp.normal_area[3] = p.getArea();
        gp.matIdx = matIdx;
        gpuPlanes.push_back(gp);
    };

    // Add light as planes[0] so we can refer to it by index.
    int lightMatIdx = addMaterial(scene.lightSource.material);
    addPlane(scene.lightSource, lightMatIdx);
    outLightIdx = 0;

    for (const auto &w : scene.walls)
    {
        int wmi = addMaterial(w.material);
        addPlane(w, wmi);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _sphereSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(gpuSpheres.size() * sizeof(GpuSphere)),
                 gpuSpheres.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _planeSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(gpuPlanes.size() * sizeof(GpuPlane)),
                 gpuPlanes.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _materialSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(mats.size() * sizeof(GpuMaterial)),
                 mats.data(), GL_DYNAMIC_DRAW);
}

void GpuRenderer::render(const Scenes::SceneData &scene,
                         std::chrono::steady_clock::time_point start,
                         const std::string &outputDir)
{
    lastOutputPath.clear();

    if (!_sharedContext)
    {
        std::cerr << "GpuRenderer: no shared OpenGL context" << std::endl;
        return;
    }

    glfwMakeContextCurrent(_sharedContext);

    if (!initGL())
    {
        glfwMakeContextCurrent(nullptr);
        return;
    }

    int lightIdx = 0;
    uploadScene(scene, lightIdx);

    glUseProgram(_program);

    // Bind output image
    glBindImageTexture(0, _outputTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _sphereSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _planeSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _materialSSBO);

    // Set uniforms
    auto setI = [&](const char *name, int v) {
        glUniform1i(glGetUniformLocation(_program, name), v);
    };
    auto setF = [&](const char *name, float v) {
        glUniform1f(glGetUniformLocation(_program, name), v);
    };

    setI("uWidth",          _width);
    setI("uHeight",         _height);
    setF("uFov",            _fov);
    setI("uDepth",          _maxDepth);
    setI("uSamples",        _samples);
    setI("uShadowSamples",  _shadowSamples);
    setI("uSphereCount",    (int)scene.spheres.size());
    setI("uPlaneCount",     (int)(scene.walls.size() + 1)); // walls + light
    setI("uLightPlaneIdx",  lightIdx);
    setI("uFrameSeed",      (int)(std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count() & 0x7FFFFFFF));
    setI("uUseMIS",         useMIS ? 1 : 0);
    setI("uUseRussian",     useRussian ? 1 : 0);
    setI("uUseStratified",  useStratified ? 1 : 0);
    setI("uStrata",         useStratified
                            ? std::max(1, (int)std::round(std::sqrt((float)_samples)))
                            : 0);

    // Dispatch in row strips so cancel + progress are responsive and we stay
    // well under any GPU TDR (Timeout Detection and Recovery) window.
    const int STRIP_HEIGHT = 32;
    int doneRows = 0;
    for (int yStart = 0; yStart < _height; yStart += STRIP_HEIGHT)
    {
        if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
        {
            std::cout << "GPU render cancelled." << std::endl;
            glfwMakeContextCurrent(nullptr);
            return;
        }
        int yEnd = std::min(yStart + STRIP_HEIGHT, _height);
        int stripH = yEnd - yStart;
        setI("uYOffset", yStart);
        setI("uYEnd", yEnd);

        // 16x16 work group, one item per pixel. ceil division.
        GLuint gx = (GLuint)((_width  + 15) / 16);
        GLuint gy = (GLuint)((stripH + 15) / 16);
        glDispatchCompute(gx, gy, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Force completion of this strip before moving on. Without this the
        // driver might queue many strips and wash out per-strip cancel
        // responsiveness. glFinish is expensive but tolerable per-strip.
        glFinish();

        doneRows += stripH;
        if (progressRows)
            progressRows->store(doneRows, std::memory_order_relaxed);
    }

    // Read pixels back to CPU as 8-bit RGBA. Convert to RGB for lodepng.
    std::vector<unsigned char> rgba((size_t)_width * _height * 4);
    glBindTexture(GL_TEXTURE_2D, _outputTex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    if (!checkGl("readback")) {
        glfwMakeContextCurrent(nullptr);
        return;
    }

    glfwMakeContextCurrent(nullptr);

    auto end = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "GPU render took " << elapsedMs << " ms" << std::endl;

    std::vector<unsigned char> rgb((size_t)_width * _height * 3);
    for (size_t i = 0; i < (size_t)_width * _height; i++)
    {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }

    if (useDenoise)
        Denoise::bilateralRGB(rgb, _width, _height);

    std::string timestamp = formatTimestamp(false);
    std::string filename = scene.name + "-" + scene.version + "-"
                         + timestamp
                         + "-d" + std::to_string(_maxDepth)
                         + "-s" + std::to_string(_samples)
                         + "-S" + std::to_string(_shadowSamples)
                         + "-w" + std::to_string(_width);
    if (_width != _height)
        filename += "-h" + std::to_string(_height);
    filename += "-t" + std::to_string(elapsedMs) + "-gpu.png";

    fs::path outputPath = fs::path(outputDir) / filename;
    fs::create_directories(outputPath.parent_path());

    lodepng::State state;
    state.info_raw.colortype = LCT_RGB;
    state.info_raw.bitdepth = 8;
    state.info_png.color.colortype = LCT_RGB;
    state.info_png.color.bitdepth = 8;
    state.encoder.text_compression = 0;

    auto addText = [&](const char *key, const std::string &val) {
        lodepng_add_text(&state.info_png, key, val.c_str());
    };
    addText("Software",      "physically-cringe-rendering");
    addText("Backend",       "GPU OpenGL 4.3 compute");
    addText("Scene",         scene.name);
    addText("SceneVersion",  scene.version);
    addText("CreationTime",  timestamp);
    addText("Depth",         std::to_string(_maxDepth));
    addText("Samples",       std::to_string(_samples));
    addText("ShadowSamples", std::to_string(_shadowSamples));
    addText("Width",         std::to_string(_width));
    addText("Height",        std::to_string(_height));
    addText("RenderTimeMs",  std::to_string(elapsedMs));
    addText("Denoise",       useDenoise    ? "1" : "0");
    addText("MIS",           useMIS        ? "1" : "0");
    addText("Russian",       useRussian    ? "1" : "0");
    addText("Stratified",    useStratified ? "1" : "0");

    std::vector<unsigned char> pngBuffer;
    unsigned encErr = lodepng::encode(pngBuffer, rgb, _width, _height, state);
    if (encErr) {
        std::cerr << "lodepng encode error " << encErr << ": "
                  << lodepng_error_text(encErr) << std::endl;
        return;
    }

    unsigned saveErr = lodepng::save_file(pngBuffer, outputPath.string());
    if (saveErr) {
        std::cerr << "lodepng save error " << saveErr << ": "
                  << lodepng_error_text(saveErr) << std::endl;
        return;
    }

    std::cout << "Wrote " << outputPath << std::endl;
    lastOutputPath = outputPath.string();
}

void GpuRenderer::destroyGL()
{
    if (_program) { glDeleteProgram(_program); _program = 0; }
    if (_outputTex) { glDeleteTextures(1, &_outputTex); _outputTex = 0; }
    if (_sphereSSBO)   { glDeleteBuffers(1, &_sphereSSBO);   _sphereSSBO = 0; }
    if (_planeSSBO)    { glDeleteBuffers(1, &_planeSSBO);    _planeSSBO = 0; }
    if (_materialSSBO) { glDeleteBuffers(1, &_materialSSBO); _materialSSBO = 0; }
    _initialized = false;
}
