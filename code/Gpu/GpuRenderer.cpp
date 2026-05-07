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

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "Gpu/GpuRenderer.h"
#include "Includes/lodepng.h"
#include "Includes/Denoise.h"
#include "Includes/OidnDenoise.h"

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
    X(void,   Uniform3f,            GLint, GLfloat, GLfloat, GLfloat)             \
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
#define glUniform3f           pcr_glUniform3f
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

layout(rgba8, binding = 0)   uniform writeonly image2D uOutput;

uniform int   uWidth;
uniform int   uHeight;
uniform float uFov;
uniform vec3  uOrigin;
uniform int   uDepth;
uniform int   uSamples;
uniform int   uShadowSamples;
uniform int   uSphereCount;
uniform int   uPlaneCount;
uniform int   uTriangleCount;
uniform int   uBvhNodeCount;
uniform int   uLightCount;
uniform float uTotalLightArea;
uniform int   uXOffset;
uniform int   uXEnd;
uniform int   uYOffset;
uniform int   uYEnd;
uniform int   uFrameSeed;
uniform int   uUseMIS;        // 0/1
uniform int   uUseRussian;    // 0/1
uniform int   uUseStratified; // 0/1
uniform int   uStrata;        // round(sqrt(uSamples)) when stratified, else 0
uniform int   uUseACES;       // 0 = Reinhard, 1 = ACES filmic (Narkowicz)
uniform int   uAaSamples;     // 1 = no AA; >1 = jittered primary rays per pixel
uniform int   uUseAdaptive;   // 0/1; meaningful only when uAaSamples > 1

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
    int  metallic;        // 0 = diffuse/glass; 1 = perfect mirror
    int  transparent;     // 0 = opaque; 1 = glass dielectric
    float ior;            // index of refraction (transparent only)
    int  _pad;
};

struct GpuTriangle {
    vec4 v0;              // xyz=v0 vertex position
    vec4 v1;              // xyz=v1
    vec4 v2;              // xyz=v2
    vec4 n0;              // xyz=per-vertex normal at v0 (smooth shading)
    vec4 n1;
    vec4 n2;
    vec4 flatN;           // xyz=geometric normal (cross product of edges)
    int  matIdx;
    int  smooth_;         // 1 = use n0/n1/n2 with barycentric interp, 0 = use flatN
    int  _pad0, _pad1;
};

// BVH node: mirrors CPU Bvh::Node with two extra pad floats per AABB so
// boxMin/boxMax sit on vec4 boundaries (std430 alignment).
struct GpuBvhNode {
    vec4 boxMin;          // xyz=AABB min
    vec4 boxMax;          // xyz=AABB max
    int  leftOrFirst;     // internal: left-child idx; leaf: first triangle idx
    int  rightChild;      // internal: right-child idx; leaf: unused
    int  count;           // 0 = internal; > 0 = leaf with this many tris
    int  _pad;
};

// Multi-light SSBO. Plane and TriangleSet kinds share one struct via tag
// discriminator. firstTri/count index into LightTriangles[] for the
// TriangleSet kind (Plane lights ignore those fields).
struct GpuLight {
    vec4 origin;          // xyz=plane origin (Plane kind only)
    vec4 u;               // xyz=plane u    (Plane kind only)
    vec4 v;               // xyz=plane v    (Plane kind only)
    vec4 normal_area;     // xyz=plane N, w=this light's total area
    vec4 emissive;        // representative emissive (Plane: that material's;
                          // TriangleSet: first tri's, used as fallback only)
    int  kind;            // 0 = Plane, 1 = TriangleSet
    int  firstTri;        // TriangleSet only
    int  count;           // TriangleSet only
    int  _pad;
};

// Compact triangle for area-light sampling. flatN.w stores the cumulative
// area of triangles 0..i within the parent light, so binary search on .w
// picks a triangle proportional to area in O(log count).
struct GpuLightTriangle {
    vec4 v0;
    vec4 v1;
    vec4 v2;
    vec4 flatN;           // xyz=N, w=cumulative area within parent light
    vec4 emissive;
};

layout(std430, binding = 1) buffer Spheres        { GpuSphere        spheres[];        };
layout(std430, binding = 2) buffer Planes         { GpuPlane         planes[];         };
layout(std430, binding = 3) buffer Materials      { GpuMaterial      materials[];      };
layout(std430, binding = 4) buffer Triangles      { GpuTriangle      triangles[];      };
layout(std430, binding = 5) buffer BvhNodes       { GpuBvhNode       bvhNodes[];       };
layout(std430, binding = 6) buffer Lights         { GpuLight         lights[];         };
layout(std430, binding = 7) buffer LightTriangles { GpuLightTriangle lightTriangles[]; };

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

// Möller-Trumbore. Mirrors CPU code in Includes/Triangle.h. Returns the
// shading normal (smooth-interpolated when t.smooth_ != 0, flat otherwise);
// the renderer flips it to face the ray after the call, like every other
// primitive in this scene.
bool intersectTriangle(vec3 ro, vec3 rd, GpuTriangle t,
                       out float tt, out vec3 hitOut, out vec3 nOut) {
    const float EPS = 1e-6;
    vec3 e1 = t.v1.xyz - t.v0.xyz;
    vec3 e2 = t.v2.xyz - t.v0.xyz;
    vec3 pvec = cross(rd, e2);
    float det = dot(e1, pvec);
    if (abs(det) < EPS) return false;
    float invDet = 1.0 / det;
    vec3 tvec = ro - t.v0.xyz;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) return false;
    vec3 qvec = cross(tvec, e1);
    float v = dot(rd, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0) return false;
    float thit = dot(e2, qvec) * invDet;
    if (thit <= EPS) return false;
    tt = thit;
    hitOut = ro + rd * thit;
    if (t.smooth_ != 0) {
        float w = 1.0 - u - v;
        nOut = normalize(t.n0.xyz * w + t.n1.xyz * u + t.n2.xyz * v);
    } else {
        nOut = t.flatN.xyz;
    }
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

// Slab-method ray-AABB. Returns true if the ray segment (0, segMax)
// intersects the box. Mirrors CPU Bvh::rayAabb.
bool intersectAabb(vec3 ro, vec3 rd, vec3 mn, vec3 mx, float segMax) {
    float tmin = 0.0;
    float tmax = segMax;
    for (int i = 0; i < 3; i++) {
        float invD = 1.0 / rd[i];
        float t1 = (mn[i] - ro[i]) * invD;
        float t2 = (mx[i] - ro[i]) * invD;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
        if (tmin > tmax) return false;
    }
    return true;
}

// Stack-based BVH closest-hit traversal. Mirrors CPU Bvh::intersect.
// Returns true on hit; out params (hit, N, matIdx, t_out) are set to the
// closest intersection within (EPS, closest_t).
bool intersectBvh(vec3 ro, vec3 rd, float closest_t,
                  out vec3 hit, out vec3 N, out int matIdx, out float t_out) {
    if (uBvhNodeCount == 0) return false;

    // 32-deep stack covers BVH trees up to 2^32 triangles, which is far
    // past anything we'd actually render. Smaller-than-CPU stack because
    // GLSL allocates one of these per pixel-thread in private memory; a
    // 64-deep version was tipping bunny + Picture-class scenes over GPU
    // scratch-space limits on some Windows drivers.
    int stack[32];
    int top = 0;
    stack[top++] = 0; // root

    bool anyHit = false;
    float closest = closest_t;

    while (top > 0) {
        int idx = stack[top - 1];
        top -= 1;
        GpuBvhNode n = bvhNodes[idx];

        if (!intersectAabb(ro, rd, n.boxMin.xyz, n.boxMax.xyz, closest)) continue;

        if (n.count > 0) {
            for (int i = 0; i < n.count; i++) {
                int triIdx = n.leftOrFirst + i;
                float tt;
                vec3 ph, pn;
                if (intersectTriangle(ro, rd, triangles[triIdx], tt, ph, pn) && tt < closest) {
                    closest = tt;
                    hit = ph;
                    N = pn;
                    matIdx = triangles[triIdx].matIdx;
                    anyHit = true;
                }
            }
        } else {
            // Internal: push both children. Bounds-check the stack push.
            if (top + 2 <= 32) {
                stack[top++] = n.leftOrFirst;
                stack[top++] = n.rightChild;
            }
        }
    }

    if (anyHit) t_out = closest;
    return anyHit;
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

    // Triangles go through the BVH (built once at scene load and uploaded
    // verbatim from the CPU's Bvh::Node array). Empty BVH = no triangles.
    if (uTriangleCount > 0) {
        vec3 triHit, triN;
        int triMat;
        float triT;
        if (intersectBvh(ro, rd, closest, triHit, triN, triMat, triT) && triT < closest) {
            closest = triT;
            hit = triHit;
            N = triN;
            matIdx = triMat;
            found = true;
        }
    }
    return found;
}

// Pick one area light proportional to area, then sample uniformly within
// it. Mirrors CPU pick-and-sample. PDF over total light surface area =
// 1/uTotalLightArea regardless of which light got picked.
void sampleAreaLight(inout uint seed,
                     out vec3 sampleP, out vec3 sampleN, out vec3 sampleEmissive) {
    float pickTarget = rand(seed) * uTotalLightArea;
    int lightIdx = 0;
    float cumul = 0.0;
    for (int i = 0; i < uLightCount; i++) {
        cumul += lights[i].normal_area.w;
        if (pickTarget <= cumul) { lightIdx = i; break; }
    }

    GpuLight L = lights[lightIdx];
    if (L.kind == 0) {
        // Plane kind: uniform on parallelogram.
        float ru = rand(seed);
        float rv = rand(seed);
        sampleP = L.origin.xyz + L.u.xyz * ru + L.v.xyz * rv;
        sampleN = L.normal_area.xyz;
        sampleEmissive = L.emissive.rgb;
    } else {
        // TriangleSet: binary-search the cumulative area (stored in flatN.w
        // of each light triangle), then uniform-sample within that triangle.
        float rtri = rand(seed) * L.normal_area.w;
        int lo = L.firstTri;
        int hi = L.firstTri + L.count - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (lightTriangles[mid].flatN.w < rtri) lo = mid + 1;
            else hi = mid;
        }
        int triIdx = lo;

        float r1 = rand(seed);
        float r2 = rand(seed);
        if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }
        vec3 v0 = lightTriangles[triIdx].v0.xyz;
        vec3 v1 = lightTriangles[triIdx].v1.xyz;
        vec3 v2 = lightTriangles[triIdx].v2.xyz;
        sampleP = v0 + (v1 - v0) * r1 + (v2 - v0) * r2;
        sampleN = lightTriangles[triIdx].flatN.xyz;
        sampleEmissive = lightTriangles[triIdx].emissive.rgb;
    }
}
)GLSL"
// MSVC has a 16380-byte limit on a single string literal. The GLSL source
// crosses that around phase 4 once BVH traversal + multi-light sampling
// land, so split into two adjacent literals — C++ concatenates them at
// compile time. Pick a clean function boundary so the source still reads
// linearly.
R"GLSL(
vec3 tracePath(vec2 pix, float pr1, float pr2, inout uint seed) {
    float aspect = float(uWidth) / float(uHeight);
    float scale = tan(PI / 180.0 * 0.5 * uFov);
    float x = ((2.0 * (pix.x + 0.5) / float(uWidth)) - 1.0) * scale * aspect;
    float y = -((2.0 * (pix.y + 0.5) / float(uHeight)) - 1.0) * scale;
    vec3 rd = normalize(vec3(x, y, -1.0));
    vec3 ro = uOrigin;

    vec3 throughput = vec3(1.0);
    vec3 radiance = vec3(0.0);

    // pr1, pr2 only seed the FIRST indirect bounce when stratified is on; later
    // bounces fall back to PCG. Stratifying every bounce would require N^depth
    // strata, which isn't worth the bookkeeping.
    bool firstBounce = true;

    for (int bounce = 0; bounce < uDepth; bounce++) {
        vec3 hit, N;
        int matIdx;
        if (!sceneIntersect(ro, rd, hit, N, matIdx)) break;

        // Capture entering before flipping N — glass refraction needs to
        // know whether we're going air->glass or glass->air, which the
        // post-flip orientation alone can't tell us.
        bool entering = dot(rd, N) < 0.0;
        if (!entering) N = -N;

        GpuMaterial mat = materials[matIdx];

        if (any(greaterThan(mat.emissive.rgb, vec3(0.0)))) {
            radiance += throughput * mat.emissive.rgb;
            break;
        }

        // Specular: mirror = perfect reflection along the geometric normal.
        // No diffuse contribution, no shadow rays — albedo just tints the
        // recursive path. Continue to the next bounce.
        if (mat.metallic != 0) {
            rd = reflect(rd, N);
            ro = hit + N * 1e-3;
            throughput *= mat.albedo.rgb;
            firstBounce = false;
            continue;
        }

        // Specular: glass = Fresnel-weighted reflection vs refraction
        // (Schlick's approximation), Snell's law for refraction direction.
        // Total internal reflection when refraction is impossible.
        if (mat.transparent != 0) {
            float n1 = entering ? 1.0 : mat.ior;
            float n2 = entering ? mat.ior : 1.0;
            float eta = n1 / n2;
            float cosI = -dot(rd, N);
            float sinT2 = eta * eta * (1.0 - cosI * cosI);
            vec3 newDir;
            vec3 newOrigin;
            if (sinT2 >= 1.0) {
                newDir = reflect(rd, N);
                newOrigin = hit + N * 1e-3;
            } else {
                float cosT = sqrt(1.0 - sinT2);
                float F0 = (n1 - n2) / (n1 + n2);
                F0 = F0 * F0;
                float F = F0 + (1.0 - F0) * pow(1.0 - cosI, 5.0);
                if (rand(seed) < F) {
                    newDir = reflect(rd, N);
                    newOrigin = hit + N * 1e-3;
                } else {
                    newDir = rd * eta + N * (eta * cosI - cosT);
                    newOrigin = hit - N * 1e-3;
                }
            }
            rd = newDir;
            ro = newOrigin;
            throughput *= mat.albedo.rgb;
            firstBounce = false;
            continue;
        }

        // Direct lighting (multi-light: pick by area, sample within).
        vec3 directLo = vec3(0.0);
        if (uTotalLightArea > 0.0) {
            for (int s = 0; s < uShadowSamples; s++) {
                vec3 sampleP, sampleN, sampleEmissive;
                sampleAreaLight(seed, sampleP, sampleN, sampleEmissive);

                vec3 Li = sampleP - hit;
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
                    float cosLight = max(0.0, dot(sampleN, -wi));
                    float G = (cosTheta * cosLight) / lightDist2;
                    // pdf = 1/uTotalLightArea, so divide-by-pdf = uTotalLightArea.
                    vec3 directContrib = (mat.albedo.rgb / PI) * sampleEmissive * G * uTotalLightArea;

                    // Partial MIS: light-side balance heuristic, same caveat
                    // as CPU renderer (BRDF-side weighting omitted).
                    if (uUseMIS != 0 && cosLight > 1e-6) {
                        float pdfLight = lightDist2 / (cosLight * uTotalLightArea);
                        float pdfBrdf  = cosTheta / PI;
                        float w = (pdfLight * pdfLight) /
                                  (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                        directContrib *= w;
                    }
                    directLo += directContrib;
                }
            }
            directLo /= float(uShadowSamples);
        }
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

// Narkowicz 2015 ACES filmic approximation. Per-channel S-curve with a
// gentle toe and shoulder; preserves midtone contrast better than the
// Reinhard concave curve at the cost of mild hue shifts in saturated
// highlights. Cheap (handful of muls/divs) and visually close to the
// full ACES RRT+ODT pipeline.
vec3 aces(vec3 x) {
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
}

void main() {
    ivec2 pix = ivec2(int(gl_GlobalInvocationID.x) + uXOffset,
                      int(gl_GlobalInvocationID.y) + uYOffset);
    // 2D tile dispatch: bounds-check against the tile rectangle AND the
    // image extent (the tile may end mid-image with extra invocations).
    if (pix.x >= uXEnd || pix.x >= uWidth ||
        pix.y >= uYEnd || pix.y >= uHeight) return;

    uint seed = uint(pix.x) * 1973u + uint(pix.y) * 9277u + uint(uFrameSeed) * 26699u;

    // AA outer loop with optional adaptive early-exit. Welford's algorithm
    // tracks running mean + M2 (sum of squared deviations) so we can
    // check relative variance after each sample without recomputing.
    int aaN = max(1, uAaSamples);
    vec3 mean = vec3(0.0);
    vec3 M2 = vec3(0.0);
    int taken = 0;
    for (int aa = 0; aa < aaN; aa++) {
        vec2 jpix = vec2(pix);
        if (aaN > 1) {
            jpix.x += rand(seed) - 0.5;
            jpix.y += rand(seed) - 0.5;
        }
        vec3 accum = vec3(0.0);
        for (int s = 0; s < uSamples; s++) {
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
            accum += tracePath(jpix, r1, r2, seed);
        }
        accum /= float(uSamples);

        // Welford update.
        taken += 1;
        vec3 delta = accum - mean;
        mean += delta / float(taken);
        vec3 delta2 = accum - mean;
        M2 += delta * delta2;

        if (uUseAdaptive != 0 && taken >= 4) {
            vec3 variance = M2 / float(taken - 1);
            vec3 rel = variance / (mean * mean + vec3(0.01));
            if (max(max(rel.r, rel.g), rel.b) < 0.05) break;
        }
    }
    vec3 outColor = (uUseACES != 0) ? aces(mean) : reinhard(mean);

    imageStore(uOutput, pix, vec4(outColor, 1.0));
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
        int   metallic;
        int   transparent;
        float ior;
        int   _pad;
    };
    struct GpuTriangle
    {
        float v0[4];
        float v1[4];
        float v2[4];
        float n0[4];
        float n1[4];
        float n2[4];
        float flatN[4];
        int   matIdx;
        int   smooth_;
        int   _pad[2];
    };
    struct GpuBvhNode
    {
        float boxMin[4];   // xyz=AABB min, w=padding
        float boxMax[4];   // xyz=AABB max, w=padding
        int   leftOrFirst;
        int   rightChild;
        int   count;
        int   _pad;
    };
    struct GpuLight
    {
        float origin[4];      // Plane only
        float u[4];           // Plane only
        float v[4];           // Plane only
        float normal_area[4]; // xyz=N (Plane only), w=this light's totalArea
        float emissive[4];    // representative
        int   kind;           // 0 = Plane, 1 = TriangleSet
        int   firstTri;       // TriangleSet only
        int   count;          // TriangleSet only
        int   _pad;
    };
    struct GpuLightTriangle
    {
        float v0[4];
        float v1[4];
        float v2[4];
        float flatN[4];   // xyz=N, w=cumulative area within parent light
        float emissive[4];
    };

    // Mirrors Renderer.cpp's compressZone — strftime("%Z") on Windows
    // returns long Windows-registry names like "Eastern Daylight Time"
    // instead of POSIX abbreviations. Compress to "EDT" etc. so the
    // filename slug is the same shape across platforms.
    std::string compressZone(const std::string &z)
    {
        static const std::unordered_map<std::string, std::string> map = {
            {"Eastern Standard Time",       "EST"},
            {"Eastern Daylight Time",       "EDT"},
            {"Central Standard Time",       "CST"},
            {"Central Daylight Time",       "CDT"},
            {"Mountain Standard Time",      "MST"},
            {"Mountain Daylight Time",      "MDT"},
            {"Pacific Standard Time",       "PST"},
            {"Pacific Daylight Time",       "PDT"},
            {"Alaskan Standard Time",       "AKST"},
            {"Alaskan Daylight Time",       "AKDT"},
            {"Hawaiian Standard Time",      "HST"},
            {"GMT Standard Time",           "GMT"},
            {"GMT Daylight Time",           "BST"},
            {"Coordinated Universal Time",  "UTC"},
        };
        auto it = map.find(z);
        if (it != map.end()) return it->second;
        // POSIX returns short forms already ("EDT", "UTC", "PST"); only
        // multi-word inputs (Windows long names) need the fallback below.
        if (z.find(' ') == std::string::npos) return z;
        std::string abbrev;
        bool atWordStart = true;
        for (char c : z)
        {
            if (c == ' ') { atWordStart = true; continue; }
            if (atWordStart && std::isalpha((unsigned char)c))
                abbrev += (char)std::toupper((unsigned char)c);
            atWordStart = false;
        }
        return abbrev.empty() ? z : abbrev;
    }

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
        std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
        std::string out(buf);
        if (utc)
        {
            out += "-UTC";
        }
        else
        {
            char zone[64];
            std::strftime(zone, sizeof(zone), "%Z", &tm);
            out += "-";
            out += compressZone(zone);
        }
        return out;
    }
}

// ---------- GpuRenderer ---------------------------------------------------

GpuRenderer::GpuRenderer(int width, int height,
                         int depth, int samples, int shadowSamples,
                         GLFWwindow *sharedContext)
    : _width{width}, _height{height},
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
    glGenBuffers(1, &_triangleSSBO);
    glGenBuffers(1, &_materialSSBO);
    glGenBuffers(1, &_bvhSSBO);
    glGenBuffers(1, &_lightSSBO);
    glGenBuffers(1, &_lightTriSSBO);

    _initialized = true;
    return true;
}

void GpuRenderer::uploadScene(const Scenes::SceneData &scene, float &outTotalLightArea)
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
        gm.metallic = m.metallic ? 1 : 0;
        gm.transparent = m.transparent ? 1 : 0;
        gm.ior = m.ior;
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

    // Plane lights go into the GpuPlane SSBO front-loaded so coplanar ties
    // (cornell ceiling-cutout case) are won by the light, mirroring the CPU
    // _planes ordering. Walls follow.
    for (const auto &L : scene.areaLights)
    {
        if (L.kind != Scenes::AreaLightKind::Plane) continue;
        int mi = addMaterial(L.plane.material);
        addPlane(L.plane, mi);
    }
    for (const auto &w : scene.walls)
    {
        int wmi = addMaterial(w.material);
        addPlane(w, wmi);
    }

    std::vector<GpuTriangle> gpuTris;
    for (const auto &t : scene.triangles)
    {
        int mi = addMaterial(t.material);
        GpuTriangle gt{};
        for (int i = 0; i < 3; i++)
        {
            gt.v0[i] = t.v0[i];
            gt.v1[i] = t.v1[i];
            gt.v2[i] = t.v2[i];
            gt.n0[i] = t.n0[i];
            gt.n1[i] = t.n1[i];
            gt.n2[i] = t.n2[i];
            gt.flatN[i] = t.flatN[i];
        }
        gt.matIdx = mi;
        gt.smooth_ = t.smooth ? 1 : 0;
        gpuTris.push_back(gt);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _sphereSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(gpuSpheres.size() * sizeof(GpuSphere)),
                 gpuSpheres.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _planeSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(gpuPlanes.size() * sizeof(GpuPlane)),
                 gpuPlanes.data(), GL_DYNAMIC_DRAW);

    // Empty SSBO with non-zero size: GL allows zero-byte BufferData but some
    // drivers emit a warning. Use a 1-element dummy when there are no
    // triangles in the scene.
    if (gpuTris.empty())
    {
        GpuTriangle dummy{};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _triangleSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GpuTriangle),
                     &dummy, GL_DYNAMIC_DRAW);
    }
    else
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _triangleSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     (GLsizeiptr)(gpuTris.size() * sizeof(GpuTriangle)),
                     gpuTris.data(), GL_DYNAMIC_DRAW);
    }

    // Convert the CPU BVH node array (3-component AABBs) into the GPU
    // layout (4-component AABBs with explicit pad). The CPU code is
    // free to use the smaller layout because std::vector handles its
    // own alignment; std430 forces the vec4 boundaries.
    std::vector<GpuBvhNode> gpuBvh;
    gpuBvh.reserve(scene.triangleBvh.size());
    for (const auto &n : scene.triangleBvh)
    {
        GpuBvhNode gn{};
        for (int i = 0; i < 3; i++)
        {
            gn.boxMin[i] = n.boxMin[i];
            gn.boxMax[i] = n.boxMax[i];
        }
        gn.leftOrFirst = n.leftOrFirst;
        gn.rightChild = n.rightChild;
        gn.count = n.count;
        gpuBvh.push_back(gn);
    }

    // Build the multi-light buffers. Plane lights store geometry in the
    // GpuLight struct directly; TriangleSet lights store firstTri/count
    // pointing into the parallel GpuLightTriangle SSBO. flatN.w on each
    // light triangle holds the cumulative area within its parent light
    // for binary-search picking on the GPU.
    std::vector<GpuLight> gpuLights;
    std::vector<GpuLightTriangle> gpuLightTris;
    float totalLightArea = 0.f;
    for (const auto &L : scene.areaLights)
    {
        GpuLight gl{};
        gl.normal_area[3] = L.totalArea;
        totalLightArea += L.totalArea;
        if (L.kind == Scenes::AreaLightKind::Plane)
        {
            gl.kind = 0;
            const Vec3f &u = L.plane.getU();
            const Vec3f &v = L.plane.getV();
            for (int i = 0; i < 3; i++)
            {
                gl.origin[i] = L.plane.origin[i];
                gl.u[i] = u[i];
                gl.v[i] = v[i];
                gl.normal_area[i] = L.plane.N[i];
                gl.emissive[i] = L.plane.material.emissive[i];
            }
            gl.firstTri = 0;
            gl.count = 0;
        }
        else
        {
            gl.kind = 1;
            gl.firstTri = (int)gpuLightTris.size();
            gl.count = (int)L.triangles.size();
            // First-triangle emissive as a representative; the GLSL
            // sampler reads per-triangle emissive directly.
            if (!L.triangles.empty())
                for (int i = 0; i < 3; i++)
                    gl.emissive[i] = L.triangles.front().material.emissive[i];

            for (size_t ti = 0; ti < L.triangles.size(); ti++)
            {
                const Triangle &t = L.triangles[ti];
                GpuLightTriangle glt{};
                for (int i = 0; i < 3; i++)
                {
                    glt.v0[i] = t.v0[i];
                    glt.v1[i] = t.v1[i];
                    glt.v2[i] = t.v2[i];
                    glt.flatN[i] = t.flatN[i];
                    glt.emissive[i] = t.material.emissive[i];
                }
                glt.flatN[3] = L.cumulativeArea[ti];
                gpuLightTris.push_back(glt);
            }
        }
        gpuLights.push_back(gl);
    }
    outTotalLightArea = totalLightArea;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _materialSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(mats.size() * sizeof(GpuMaterial)),
                 mats.data(), GL_DYNAMIC_DRAW);

    // BVH and light SSBOs use the same dummy-1-element trick as the
    // triangle SSBO so drivers don't choke on zero-byte allocations.
    auto uploadOrDummy = [&](unsigned ssbo, const void *data, size_t bytes,
                             const void *dummy, size_t dummyBytes) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        if (bytes == 0)
            glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)dummyBytes,
                         dummy, GL_DYNAMIC_DRAW);
        else
            glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes,
                         data, GL_DYNAMIC_DRAW);
    };
    GpuBvhNode bvhDummy{};
    GpuLight lightDummy{};
    GpuLightTriangle ltDummy{};
    uploadOrDummy(_bvhSSBO, gpuBvh.data(), gpuBvh.size() * sizeof(GpuBvhNode),
                  &bvhDummy, sizeof(bvhDummy));
    uploadOrDummy(_lightSSBO, gpuLights.data(), gpuLights.size() * sizeof(GpuLight),
                  &lightDummy, sizeof(lightDummy));
    uploadOrDummy(_lightTriSSBO, gpuLightTris.data(),
                  gpuLightTris.size() * sizeof(GpuLightTriangle),
                  &ltDummy, sizeof(ltDummy));
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

    if (scene.areaLights.empty())
    {
        std::cerr << "physically-cringe-rendering: scene has no area lights"
                  << std::endl;
        return;
    }

    glfwMakeContextCurrent(_sharedContext);

    if (!initGL())
    {
        glfwMakeContextCurrent(nullptr);
        return;
    }

    float totalLightArea = 0.f;
    uploadScene(scene, totalLightArea);

    glUseProgram(_program);

    // Bind output image
    glBindImageTexture(0, _outputTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _sphereSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _planeSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _materialSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _triangleSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _bvhSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _lightSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _lightTriSSBO);

    // Set uniforms
    auto setI = [&](const char *name, int v) {
        glUniform1i(glGetUniformLocation(_program, name), v);
    };
    auto setF = [&](const char *name, float v) {
        glUniform1f(glGetUniformLocation(_program, name), v);
    };
    auto setV3 = [&](const char *name, float x, float y, float z) {
        glUniform3f(glGetUniformLocation(_program, name), x, y, z);
    };

    setI("uWidth",          _width);
    setI("uHeight",         _height);
    setF("uFov",            scene.camera.fov);
    setV3("uOrigin",        scene.camera.position[0],
                            scene.camera.position[1],
                            scene.camera.position[2]);
    setI("uDepth",          _maxDepth);
    setI("uSamples",        _samples);
    setI("uShadowSamples",  _shadowSamples);
    // Plane count = walls + plane-kind area lights (TriangleSet lights
    // don't contribute to the plane SSBO; their geometry lives in the
    // light-triangle SSBO and the main triangle SSBO).
    int planeLightCount = 0;
    for (const auto &L : scene.areaLights)
        if (L.kind == Scenes::AreaLightKind::Plane) planeLightCount++;

    setI("uSphereCount",    (int)scene.spheres.size());
    setI("uPlaneCount",     (int)scene.walls.size() + planeLightCount);
    setI("uTriangleCount",  (int)scene.triangles.size());
    setI("uBvhNodeCount",   (int)scene.triangleBvh.size());
    setI("uLightCount",     (int)scene.areaLights.size());
    setF("uTotalLightArea", totalLightArea);
    setI("uFrameSeed",      (int)(std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count() & 0x7FFFFFFF));
    setI("uUseMIS",         useMIS ? 1 : 0);
    setI("uUseRussian",     useRussian ? 1 : 0);
    setI("uUseStratified",  useStratified ? 1 : 0);
    setI("uUseACES",        useACES ? 1 : 0);
    setI("uAaSamples",      std::max(1, aaSamples));
    setI("uUseAdaptive",    useAdaptive ? 1 : 0);
    setI("uStrata",         useStratified
                            ? std::max(1, (int)std::round(std::sqrt((float)_samples)))
                            : 0);

    // Dispatch in 2D tiles so we stay well under any GPU TDR (Timeout
    // Detection and Recovery) window. Windows defaults to a 2-second TDR;
    // a single dispatch that exceeds that is killed by the driver, taking
    // the whole process with it (no exception, no GL error — the kernel
    // just resets the GPU).
    //
    // 1D row-strip dispatch (the previous approach) hits a floor of 1 row
    // per dispatch on heavy presets, and at high resolutions one row can
    // still be too much work in a single submit. 2D tiles let us shrink
    // both dimensions, so a Picture-class render on a 70k-tri mesh at 1080
    // dispatches small enough chunks regardless of resolution.
    //
    // Tile size is computed from per-pixel work × BVH overhead. Target
    // ~0.15 sec per dispatch on the calibration GPU — a 13x safety margin
    // from the 2-sec Windows TDR cliff. The previous 0.5-sec target hit
    // TDR in practice because (a) GPU clock varies under thermal load
    // (the same shader runs ~2x slower a few seconds in than at peak),
    // (b) first-dispatch driver overhead inflates the early tiles, and
    // (c) workload variance across the image means some tiles do more
    // work than the average.
    //
    // Calibration source: cornell-spheres at 1080² × 393216 work units
    // took 245 sec ≈ 1.87e9 work-units-per-second. 0.15 sec target =
    // ~2.8e8 work units per dispatch.
    // aaSamples multiplies primary-ray count; folds into per-pixel work
    // for tile sizing so AA-enabled renders shrink tiles to compensate.
    int aaMult = std::max(1, aaSamples);
    int workPerPixel = std::max(1, _samples * _maxDepth * _shadowSamples * aaMult);

    // Per-ray cost depends on how many primitives sceneIntersect tests.
    // Spheres + planes are linear in count; triangles go through the BVH
    // at log(N/leafSize) cost. Without this multiplier, tile sizing
    // assumes a baseline-cornell scene (~7 primitives) and over-sizes
    // tiles for plane-heavy scenes — cornell-spec with 18 primitives
    // hits TDR at the same tile size that cornell handles fine.
    constexpr double kBaselinePrimitives = 7.0; // cornell baseline
    int primCount = (int)scene.spheres.size() + (int)scene.walls.size();
    for (const auto &L : scene.areaLights)
        if (L.kind == Scenes::AreaLightKind::Plane) primCount++;
    double primMult = std::max(1.0, (double)primCount / kBaselinePrimitives);
    double bvhMult = 0.0;
    if ((int)scene.triangles.size() > 4)
    {
        // Each ray traverses ~log2(N/leafSize) BVH nodes. AABB tests are
        // cheap (~1/4 of a triangle test); the 4-tri leaves add a few
        // full triangle tests per ray.
        double depth = std::log2((double)scene.triangles.size() / 4.0);
        bvhMult = depth * 0.25;
    }
    double effectivePerPixel = (double)workPerPixel * (primMult + bvhMult);
    constexpr double kTargetWorkPerDispatch = 2.8e8;
    double maxPixelsD = kTargetWorkPerDispatch / effectivePerPixel;
    int maxPixelsPerDispatch = std::max(64, (int)maxPixelsD);
    int tileSide = (int)std::sqrt((double)maxPixelsPerDispatch);
    // Tile floor: heavy per-pixel work needs aggressive small tiles even at
    // the cost of more dispatch overhead. Light work uses bigger tiles to
    // avoid drowning in glFinish overhead.
    int minTileSide = (effectivePerPixel > 5e5) ? 8 : 16;
    tileSide = std::clamp(tileSide, minTileSide, 256);
    // No work-group-multiple rounding; the shader's per-axis bounds check
    // handles non-multiples. The 16x16 work-group still dispatches a few
    // wasted invocations at the tile edge, but only ~6% on average.

    int tilesX = (_width  + tileSide - 1) / tileSide;
    int tilesY = (_height + tileSide - 1) / tileSide;
    int totalTiles = tilesX * tilesY;
    int doneTiles = 0;

#ifdef _WIN32
    // Activity log: overwritten before each dispatch so the post-mortem
    // MessageBox on the next launch knows exactly which tile was running
    // when the GPU died. The CPU GUI's runRender() stamps a render-level
    // activity message at the start; this replaces it with finer
    // granularity once we're in the dispatch loop.
    fs::path activityPath = fs::current_path() /
                            (std::string(PCR_BINARY_NAME) + ".lastrun.txt");
#endif

    for (int yStart = 0; yStart < _height; yStart += tileSide)
    {
        for (int xStart = 0; xStart < _width; xStart += tileSide)
        {
            if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
            {
                std::cout << "GPU render cancelled." << std::endl;
                glfwMakeContextCurrent(nullptr);
                return;
            }
            int xEnd = std::min(xStart + tileSide, _width);
            int yEnd = std::min(yStart + tileSide, _height);
            int tileW = xEnd - xStart;
            int tileH = yEnd - yStart;

            setI("uXOffset", xStart);
            setI("uXEnd", xEnd);
            setI("uYOffset", yStart);
            setI("uYEnd", yEnd);

#ifdef _WIN32
            {
                char act[640];
                std::snprintf(act, sizeof(act),
                    "Rendering '%s' v%s at d=%d s=%d S=%d w=%d h=%d (GPU)\n"
                    "Triangles: %d (BVH nodes: %d). Lights: %d.\n"
                    "Tile size: %d. Tile %d/%d, "
                    "pixels x=[%d..%d) y=[%d..%d).",
                    scene.name.c_str(), scene.version.c_str(),
                    _maxDepth, _samples, _shadowSamples, _width, _height,
                    (int)scene.triangles.size(), (int)scene.triangleBvh.size(),
                    (int)scene.areaLights.size(),
                    tileSide, doneTiles + 1, totalTiles,
                    xStart, xEnd, yStart, yEnd);
                std::ofstream f(activityPath);
                if (f) f << act;
            }
#endif

            // 16x16 work group, one item per pixel. ceil division.
            GLuint gx = (GLuint)((tileW + 15) / 16);
            GLuint gy = (GLuint)((tileH + 15) / 16);
            glDispatchCompute(gx, gy, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            // glFinish per tile so cancel + progress are responsive and
            // each dispatch exits the GPU before the next starts (else
            // the driver might queue many at once and the TDR watchdog
            // could see them as one long submission).
            glFinish();

            doneTiles++;
            if (progressRows)
            {
                // Map tiles-completed to rows-completed for the existing
                // progress bar. Approximate but linear, which is what the
                // user sees.
                int approxRows = (int)((double)doneTiles / totalTiles * _height);
                progressRows->store(approxRows, std::memory_order_relaxed);
            }
        }
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

    // OIDN on the GPU path: the shader has already tone-mapped to RGBA8,
    // so we run OIDN in LDR mode (no "hdr" flag) on the 8-bit output. No
    // aux buffers either — capturing albedo + normal in a separate pass
    // would need RGBA16F output textures and a second readback, and the
    // existing tile dispatch is already at the TDR cliff. So GPU OIDN is
    // best-effort denoise on already-tone-mapped data; the CPU --oidn
    // path is the high-quality choice (HDR pre-tone-map + aux).
    if (useOIDN)
    {
        if (!OidnDenoise::isAvailable())
        {
            std::cerr << "warning: --oidn requested but binary was not "
                         "built with PCR_USE_OIDN=ON; skipping.\n";
        }
        else
        {
            std::vector<Vec3f> floatBuf((size_t)_width * _height);
            for (size_t i = 0; i < floatBuf.size(); i++)
                floatBuf[i] = Vec3f(rgb[i * 3 + 0] / 255.f,
                                    rgb[i * 3 + 1] / 255.f,
                                    rgb[i * 3 + 2] / 255.f);
            std::vector<Vec3f> empty;
            OidnDenoise::denoise(floatBuf, empty, empty, _width, _height);
            for (size_t i = 0; i < floatBuf.size(); i++)
            {
                auto cl = [](float v) {
                    if (v < 0.f) v = 0.f;
                    if (v > 1.f) v = 1.f;
                    return (unsigned char)(v * 255.f + 0.5f);
                };
                rgb[i * 3 + 0] = cl(floatBuf[i][0]);
                rgb[i * 3 + 1] = cl(floatBuf[i][1]);
                rgb[i * 3 + 2] = cl(floatBuf[i][2]);
            }
        }
    }
    else if (useDenoise)
    {
        Denoise::bilateralRGB(rgb, _width, _height);
    }

    std::string timestamp = formatTimestamp(false);
    std::string filename = scene.name + "-" + scene.version + "-"
                         + timestamp
                         + "-d" + std::to_string(_maxDepth)
                         + "-s" + std::to_string(_samples)
                         + "-S" + std::to_string(_shadowSamples)
                         + "-w" + std::to_string(_width);
    if (_width != _height)
        filename += "-h" + std::to_string(_height);
    if (aaSamples > 1)
    {
        filename += "-aa" + std::to_string(aaSamples);
        if (useAdaptive) filename += "adaptive";
    }
    if (useACES)
        filename += "-aces";
    if (useOIDN)
        filename += "-oidn";
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
    addText("Tonemap",       useACES       ? "ACES" : "Reinhard");
    addText("AASamples",     std::to_string(aaSamples));
    addText("Adaptive",      useAdaptive ? "1" : "0");
    addText("OIDN",          useOIDN     ? "1" : "0");

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
