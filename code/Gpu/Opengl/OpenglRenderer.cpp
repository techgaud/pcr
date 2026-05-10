// OpenglRenderer. OpenGL 4.3 compute shader path tracer.
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

#include <algorithm>
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

#include "Gpu/Opengl/OpenglRenderer.h"
#include "Includes/lodepng.h"
#include "Includes/Denoise.h"
#include "Includes/OidnDenoise.h"
#include "Includes/ToneMap.h"

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
#define GL_RGBA16F 0x881A
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_FLOAT 0x1406
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

// HDR pre-tone-map output. Float because path-trace radiance values
// commonly exceed 1.0 (direct hits on emissive surfaces, bright indirect
// bounces). CPU reads this back as float and tone-maps after OIDN
// denoise so we get the full HDR signal into the neural net.
layout(rgba16f, binding = 0) uniform writeonly image2D uOutput;
// Aux outputs for OIDN. Float16 because albedo can exceed 1.0 (when
// material albedo > 1.0, rare but legal) and normal components are
// in [-1, 1]. Only written when uWriteAux != 0; the textures are
// allocated unconditionally because reallocating per render would
// stutter.
layout(rgba16f, binding = 1) uniform writeonly image2D uAlbedoOut;
layout(rgba16f, binding = 2) uniform writeonly image2D uNormalOut;

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
uniform int   uAaSamples;     // 1 = no AA; >1 = jittered primary rays per pixel
uniform int   uUseAdaptive;   // 0/1; meaningful only when uAaSamples > 1
uniform int   uWriteAux;      // 0/1; populate uAlbedoOut + uNormalOut for OIDN
uniform int   uUseSpectral;   // 0 = RGB path tracer, 1 = hero-wavelength spectral
uniform int   uHeroSamples;   // 4 = hero default; 1 = single-wavelength legacy

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
    vec4 albedo;          // rgb (used in RGB rendering mode)
    vec4 emissive;        // rgb (used in RGB rendering mode)
    int  metallic;        // 0 = diffuse/glass; 1 = perfect mirror
    int  transparent;     // 0 = opaque; 1 = glass dielectric
    float ior;            // index of refraction (transparent only)
    float cauchyB;        // dispersion: ior_at_lambda = ior + cauchyB*1e4/lambda^2
    // Spectral mode: 61-sample tabulated reflectance / emission. Values
    // 0..60 cover 400 nm to 700 nm at 5 nm spacing; values 61..63 are
    // unused, present only to keep the float[64] array naturally
    // 16-byte-aligned (256 bytes -> multiple of vec4) so the struct's
    // total size stays a multiple of 16 bytes for std430 array stride.
    //
    // CPU Material::populateSpectraGpu fills these from EITHER the
    // material's loaded SPD (for measured-data materials, e.g. cornell-spec
    // with its albedo_spd: "cornell/white-paint") OR by evaluating the
    // material's Jakob SigmoidFit at every wavelength (for RGB-described
    // materials where the upsampler is the only source).
    //
    // Albedo samples are clamped to [0, 1] at upload time (the CPU side
    // does the same in Material::albedoAt). Emissive samples are not
    // clamped because radiance is HDR.
    float albedoSpectrum[64];
    float emissiveSpectrum[64];
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
    int  matIdx;          // index into materials[] for spectral emission
                          // lookup (Plane kind only; TriangleSet uses the
                          // per-triangle matIdx in lightTriangles[]).
};

// Compact triangle for area-light sampling. flatN.w stores the cumulative
// area of triangles 0..i within the parent light, so binary search on .w
// picks a triangle proportional to area in O(log count).
struct GpuLightTriangle {
    vec4 v0;
    vec4 v1;
    vec4 v2;
    vec4 flatN;           // xyz=N, w=cumulative area within parent light
    vec4 emissive;        // representative RGB (used by RGB path tracer)
    int  matIdx;          // index into materials[] for spectral emission
                          // lookup (TriangleSet kind).
    int  _pad0, _pad1, _pad2;
};

layout(std430, binding = 1) buffer Spheres        { GpuSphere        spheres[];        };
layout(std430, binding = 2) buffer Planes         { GpuPlane         planes[];         };
layout(std430, binding = 3) buffer Materials      { GpuMaterial      materials[];      };
layout(std430, binding = 4) buffer Triangles      { GpuTriangle      triangles[];      };
layout(std430, binding = 5) buffer BvhNodes       { GpuBvhNode       bvhNodes[];       };
layout(std430, binding = 6) buffer Lights         { GpuLight         lights[];         };
layout(std430, binding = 7) buffer LightTriangles { GpuLightTriangle lightTriangles[]; };

// Spectral helpers, used only when uUseSpectral != 0. Mirror the
// CPU-side Spectrum / CIE / RGBToSpectrum modules, but cheaper to
// evaluate in shader: spectrum lookups are direct array indexing,
// CIE observer functions use the Wyman 2013 piecewise-Gaussian fit,
// and the XYZ -> linear sRGB matrix is the D65 standard.

const float kLambdaMin = 400.0;
const float kLambdaMax = 700.0;
const int   kSpecSamples = 61;
const float kSpecStep = 5.0; // (700-400) / (61-1)

// Linearly-interpolated lookup into a 61-sample tabulated spectrum
// (400 nm to 700 nm at 5 nm spacing). Mirrors CPU's Spectrum::operator()
// at the same wavelength. Out-of-range wavelengths return 0; in-range
// wavelengths interpolate between the two bracketing samples.
//
// The 'src' parameter is 0 for albedo and 1 for emissive. Inlining the
// dispatch here keeps the per-bounce hot path branch-free relative to
// driver-specific decisions about array passing.
float lookupTabulated(int matIdx, int src, float lambda) {
    if (lambda < kLambdaMin || lambda > kLambdaMax) return 0.0;
    float t = (lambda - kLambdaMin) / kSpecStep;
    int   i = int(t);
    if (i >= kSpecSamples - 1) {
        return src == 0 ? materials[matIdx].albedoSpectrum[kSpecSamples - 1]
                        : materials[matIdx].emissiveSpectrum[kSpecSamples - 1];
    }
    float f = t - float(i);
    if (src == 0) {
        float a = materials[matIdx].albedoSpectrum[i];
        float b = materials[matIdx].albedoSpectrum[i + 1];
        return mix(a, b, f);
    } else {
        float a = materials[matIdx].emissiveSpectrum[i];
        float b = materials[matIdx].emissiveSpectrum[i + 1];
        return mix(a, b, f);
    }
}

float albedoAt(int matIdx, float lambda) {
    // CPU clamps to [0, 1] in Material::albedoAt; we clamp at upload
    // time so the GPU side doesn't need a per-bounce min().
    return lookupTabulated(matIdx, 0, lambda);
}

float emissiveAt(int matIdx, float lambda) {
    // No clamp - emissive radiance is HDR.
    return lookupTabulated(matIdx, 1, lambda);
}

// Wyman 2013 CIE 1931 2-deg observer fit. ~1% accurate vs the
// tabulated CMFs, ~30 lines of analytic code, no LUT. Same fit the
// CPU side uses (Includes/CIE.h).
float wymanG(float lambda, float mu, float s1, float s2) {
    float t = (lambda < mu) ? (lambda - mu) * s1 : (lambda - mu) * s2;
    return exp(-0.5 * t * t);
}

vec3 cieObserverAt(float lambda) {
    float xb =  0.362 * wymanG(lambda, 442.0, 0.0624, 0.0374)
              + 1.056 * wymanG(lambda, 599.8, 0.0264, 0.0323)
              - 0.065 * wymanG(lambda, 501.1, 0.0490, 0.0382);
    float yb =  0.821 * wymanG(lambda, 568.8, 0.0213, 0.0247)
              + 0.286 * wymanG(lambda, 530.9, 0.0613, 0.0322);
    float zb =  1.217 * wymanG(lambda, 437.0, 0.0845, 0.0278)
              + 0.681 * wymanG(lambda, 459.0, 0.0385, 0.0725);
    return vec3(xb, yb, zb);
}

// Single-lambda XYZ contribution. Mirrors CIE::singleLambdaXYZ on
// the CPU side. The kLambdaMax-kLambdaMin scaling cancels with the
// uniform-pdf weight in the per-pixel estimator so absolute
// brightness matches the full-spectrum case. The 1/kYBarIntegral
// term converts back from the physical reflectance convention used
// in the spectrum buffers (s = 1 for a perfect white reflector) into
// linear-sRGB-comparable XYZ where Y(white) ~= 1; see CIE.h on the
// CPU side. The hardcoded 106.895 is the integral of the Wyman 2013
// yBar approximation over 400-700 nm at 5 nm steps, the same
// quantity CIE::yBarIntegral() computes at startup.
vec3 singleLambdaXYZ(float lambda, float radiance) {
    const float kYBarIntegral = 106.895210;
    return cieObserverAt(lambda) * radiance
         * (kLambdaMax - kLambdaMin) / kYBarIntegral;
}

// CIE XYZ to linear sRGB (D65). Standard 3x3 from IEC 61966-2-1.
vec3 xyzToLinearSRGB(vec3 xyz) {
    return vec3(
         3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
        -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
         0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z
    );
}

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

// Moller-Trumbore. Mirrors CPU code in Includes/Triangle.h. Returns the
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

)GLSL"
R"GLSL(

// Dielectric (glass) optics. Mirrors code/Includes/Optics.h on the
// CPU side line-for-line. The block was reproduced inline at every
// glass-handling callsite before this lived as a helper.
float schlickFresnel(float cosTheta, float n1, float n2) {
    float F0 = (n1 - n2) / (n1 + n2);
    F0 = F0 * F0;
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float cauchyIor(float baseIor, float cb, float lambdaNm) {
    return baseIor + cb * 1e4 / (lambdaNm * lambdaNm);
}

struct DielectricOut { vec3 dir; vec3 origin; };

DielectricOut dielectricBounce(vec3 rayDir, vec3 N, vec3 hit,
                               bool entering, float ior, float fresnelRand) {
    float cosI = -dot(rayDir, N);
    float n1 = entering ? 1.0 : ior;
    float n2 = entering ? ior : 1.0;
    float eta = n1 / n2;
    float sinT2 = eta * eta * (1.0 - cosI * cosI);
    DielectricOut o;
    if (sinT2 >= 1.0) {
        o.dir = reflect(rayDir, N);
        o.origin = hit + N * 1e-3;
        return o;
    }
    float F = schlickFresnel(cosI, n1, n2);
    if (fresnelRand < F) {
        o.dir = reflect(rayDir, N);
        o.origin = hit + N * 1e-3;
    } else {
        float cosT = sqrt(1.0 - sinT2);
        o.dir = rayDir * eta + N * (eta * cosI - cosT);
        o.origin = hit - N * 1e-3;
    }
    return o;
}

// Slab-method ray-AABB. Returns true if the ray segment (0, segMax)
// intersects the box; tNear is the ray-parametric distance at entry,
// used by intersectBvh's ordered traversal to push the farther child
// first. Mirrors CPU Bvh::rayAabb.
bool intersectAabb(vec3 ro, vec3 rd, vec3 mn, vec3 mx, float segMax, out float tNear) {
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
    tNear = tmin;
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
    // scratch-space limits on some Windows drivers. Each entry caches
    // the AABB tNear so we can drop entries pointing at farther subtrees
    // once a closer hit gets found in a near subtree (same trick the
    // CPU traversal uses).
    int   stackIdx[32];
    float stackTNear[32];
    int top = 0;
    {
        float tRoot;
        if (!intersectAabb(ro, rd, bvhNodes[0].boxMin.xyz, bvhNodes[0].boxMax.xyz, closest_t, tRoot))
            return false;
        stackIdx[top] = 0;
        stackTNear[top] = tRoot;
        top += 1;
    }

    bool anyHit = false;
    float closest = closest_t;

    while (top > 0) {
        top -= 1;
        if (stackTNear[top] > closest) continue;
        GpuBvhNode n = bvhNodes[stackIdx[top]];

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
            continue;
        }

        // Internal: test both children, push them in far-first order so
        // the near child is popped (and traversed) first.
        GpuBvhNode cl = bvhNodes[n.leftOrFirst];
        GpuBvhNode cr = bvhNodes[n.rightChild];
        float tL, tR;
        bool hitL = intersectAabb(ro, rd, cl.boxMin.xyz, cl.boxMax.xyz, closest, tL);
        bool hitR = intersectAabb(ro, rd, cr.boxMin.xyz, cr.boxMax.xyz, closest, tR);

        if (hitL && hitR) {
            if (top + 2 <= 32) {
                if (tL <= tR) {
                    stackIdx[top]   = n.rightChild;  stackTNear[top]   = tR; top += 1;
                    stackIdx[top]   = n.leftOrFirst; stackTNear[top]   = tL; top += 1;
                } else {
                    stackIdx[top]   = n.leftOrFirst; stackTNear[top]   = tL; top += 1;
                    stackIdx[top]   = n.rightChild;  stackTNear[top]   = tR; top += 1;
                }
            }
        } else if (hitL && top + 1 <= 32) {
            stackIdx[top] = n.leftOrFirst; stackTNear[top] = tL; top += 1;
        } else if (hitR && top + 1 <= 32) {
            stackIdx[top] = n.rightChild; stackTNear[top] = tR; top += 1;
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
                     out vec3 sampleP, out vec3 sampleN, out vec3 sampleEmissive,
                     out int sampleMatIdx) {
    int lightIdx = 0;
    if (uLightCount > 1) {
        // Multi-light: linear walk over the light list, weighted by area.
        // O(L) per shadow ray; with many lights this becomes the inner-loop
        // bottleneck and a precomputed CDF + binary search is the standard
        // upgrade. cornell-class scenes ship with one light and never hit
        // the loop; we keep this branch tight rather than build the CDF.
        float pickTarget = rand(seed) * uTotalLightArea;
        float cumul = 0.0;
        for (int i = 0; i < uLightCount; i++) {
            cumul += lights[i].normal_area.w;
            if (pickTarget <= cumul) { lightIdx = i; break; }
        }
    }

    GpuLight L = lights[lightIdx];
    if (L.kind == 0) {
        // Plane kind: uniform on parallelogram.
        float ru = rand(seed);
        float rv = rand(seed);
        sampleP = L.origin.xyz + L.u.xyz * ru + L.v.xyz * rv;
        sampleN = L.normal_area.xyz;
        sampleEmissive = L.emissive.rgb;
        sampleMatIdx = L.matIdx;
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
        sampleMatIdx = lightTriangles[triIdx].matIdx;
    }
}
)GLSL"
// MSVC has a 16380-byte limit on a single string literal. The GLSL source
// crosses that around phase 4 once BVH traversal + multi-light sampling
// land, so split into two adjacent literals. C++ concatenates them at
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

        // Capture entering before flipping N. glass refraction needs to
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
        // No diffuse contribution, no shadow rays. albedo just tints the
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
            DielectricOut b = dielectricBounce(rd, N, hit, entering, mat.ior, rand(seed));
            rd = b.dir;
            ro = b.origin;
            throughput *= mat.albedo.rgb;
            firstBounce = false;
            continue;
        }

        // Direct lighting (multi-light: pick by area, sample within).
        vec3 directLo = vec3(0.0);
        if (uTotalLightArea > 0.0) {
            for (int s = 0; s < uShadowSamples; s++) {
                vec3 sampleP, sampleN, sampleEmissive;
                int sampleMatIdx;
                sampleAreaLight(seed, sampleP, sampleN, sampleEmissive, sampleMatIdx);

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

)GLSL"
R"GLSL(

// Single-wavelength continuation, used by tracePathSpectral when it
// hits glass and forks into per-channel sub-paths. Mirrors
// tracePathSpectral but with scalar throughput / radiance and a
// single lambda passed in. Does not generate primary rays (those
// come from the caller after Snell's-law refraction at glass).
//
// remainingDepth bounds how many more bounces this sub-path can
// take. parentBounce is the global path-depth count from the parent
// at the point of the glass split, so Russian roulette in the sub-
// path fires at the same global depth as the CPU recursion-based
// castRaySpectral does. Without it, sub-paths effectively reset RR
// depth and run with lower termination probability than a non-glass
// path of the same length, biasing variance asymmetrically across
// glass surfaces.
float tracePathSpectralSingle(vec3 ro, vec3 rd, float lambda,
                              int remainingDepth, int parentBounce,
                              inout uint seed) {
    float throughput = 1.0;
    float radiance = 0.0;
    bool firstBounce = true;

    for (int bounce = 0; bounce < remainingDepth; bounce++) {
        vec3 hit, N;
        int matIdx;
        if (!sceneIntersect(ro, rd, hit, N, matIdx)) break;

        bool entering = dot(rd, N) < 0.0;
        if (!entering) N = -N;

        if (any(greaterThan(materials[matIdx].emissive.rgb, vec3(0.0)))) {
            radiance += throughput * emissiveAt(matIdx, lambda);
            break;
        }

        if (materials[matIdx].metallic != 0) {
            rd = reflect(rd, N);
            ro = hit + N * 1e-3;
            throughput *= albedoAt(matIdx, lambda);
            firstBounce = false;
            continue;
        }

        // Glass continuation. Each subsequent glass surface uses
        // this same single-wavelength's IOR; no further forking
        // because we're already on a single-channel path.
        if (materials[matIdx].transparent != 0) {
            float ior = cauchyIor(materials[matIdx].ior, materials[matIdx].cauchyB, lambda);
            DielectricOut b = dielectricBounce(rd, N, hit, entering, ior, rand(seed));
            rd = b.dir;
            ro = b.origin;
            throughput *= albedoAt(matIdx, lambda);
            firstBounce = false;
            continue;
        }

        // Diffuse: per-lambda direct lighting + indirect.
        float albedoLam = albedoAt(matIdx, lambda);

        float directLo = 0.0;
        if (uTotalLightArea > 0.0) {
            for (int s = 0; s < uShadowSamples; s++) {
                vec3 sampleP, sampleN, sampleEmissiveRGB;
                int sampleMatIdx;
                sampleAreaLight(seed, sampleP, sampleN, sampleEmissiveRGB, sampleMatIdx);

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
                        if (!any(greaterThan(materials[sMat].emissive.rgb, vec3(0.0))))
                            occluded = true;
                    }
                }

                if (!occluded) {
                    float cosLight = max(0.0, dot(sampleN, -wi));
                    float G = (cosTheta * cosLight) / lightDist2;
                    float misWeight = 1.0;
                    if (uUseMIS != 0 && cosLight > 1e-6) {
                        float pdfLight = lightDist2 / (cosLight * uTotalLightArea);
                        float pdfBrdf  = cosTheta / PI;
                        misWeight = (pdfLight * pdfLight) /
                                    (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                    }
                    float emitL;
                    if (sampleMatIdx >= 0) {
                        emitL = emissiveAt(sampleMatIdx, lambda);
                    } else {
                        emitL = sampleEmissiveRGB.x * 0.30
                              + sampleEmissiveRGB.y * 0.59
                              + sampleEmissiveRGB.z * 0.11;
                    }
                    directLo += (albedoLam / PI) * emitL * G * uTotalLightArea * misWeight;
                }
            }
            directLo /= float(uShadowSamples);
        }
        radiance += throughput * directLo;

        if (uUseRussian != 0 && (bounce + parentBounce) >= 1) {
            float p = clamp(albedoLam, 0.05, 0.95);
            if (rand(seed) > p) break;
            throughput /= p;
        }

        // Indirect bounce. Sub-paths skip stratification (no clean
        // way to thread pr1/pr2 across the boundary); plain random
        // here is fine since stratification was a first-bounce
        // optimization for the primary ray.
        vec3 newDir = sampleHemisphere(N, seed);
        firstBounce = false;
        ro = hit + N * 1e-3;
        rd = newDir;
        throughput *= albedoLam;
    }
    return radiance;
}

)GLSL"
R"GLSL(

// Spectral path tracer: 4 correlated wavelengths (Wilkie 2014 hero
// wavelength sampling) share the same path geometry. throughput and
// radiance are vec4, indexed [0..3] for the 4 hero channels.
// Wavelength values for those channels are passed in via lambdas.
//
// Mirrors tracePath above but with per-channel scalar math instead
// of per-channel-of-RGB componentwise math, and per-bounce material
// spectrum lookups via albedoAt/emissiveAt instead of mat.albedo.rgb.
//
// Russian roulette uses the hero (.x) channel reflectance so all 4
// channels share the same termination decision (single-distribution
// path sampling). MIS weight is wavelength-independent (depends only
// on path geometry) so it scales the per-channel direct contribution
// uniformly.
vec4 tracePathSpectral(vec2 pix, float pr1, float pr2, vec4 lambdas, inout uint seed) {
    float aspect = float(uWidth) / float(uHeight);
    float scale = tan(PI / 180.0 * 0.5 * uFov);
    float x = ((2.0 * (pix.x + 0.5) / float(uWidth)) - 1.0) * scale * aspect;
    float y = -((2.0 * (pix.y + 0.5) / float(uHeight)) - 1.0) * scale;
    vec3 rd = normalize(vec3(x, y, -1.0));
    vec3 ro = uOrigin;

    vec4 throughput = vec4(1.0);
    vec4 radiance = vec4(0.0);
    bool firstBounce = true;

    for (int bounce = 0; bounce < uDepth; bounce++) {
        vec3 hit, N;
        int matIdx;
        if (!sceneIntersect(ro, rd, hit, N, matIdx)) break;

        bool entering = dot(rd, N) < 0.0;
        if (!entering) N = -N;

        // Emissive surface: terminate, accumulate per-channel emission.
        // We check the RGB emissive flag (cheap) before paying for 4
        // spectrum lookups; if it's all-zero, the spectrum is too.
        if (any(greaterThan(materials[matIdx].emissive.rgb, vec3(0.0)))) {
            vec4 emit = vec4(emissiveAt(matIdx, lambdas.x),
                             emissiveAt(matIdx, lambdas.y),
                             emissiveAt(matIdx, lambdas.z),
                             emissiveAt(matIdx, lambdas.w));
            radiance += throughput * emit;
            break;
        }

        // Mirror: per-channel albedo at each lambda tints the
        // recursive throughput. Path direction is identical across
        // channels (no chromatic dispersion in mirrors).
        if (materials[matIdx].metallic != 0) {
            rd = reflect(rd, N);
            ro = hit + N * 1e-3;
            throughput *= vec4(albedoAt(matIdx, lambdas.x),
                               albedoAt(matIdx, lambdas.y),
                               albedoAt(matIdx, lambdas.z),
                               albedoAt(matIdx, lambdas.w));
            firstBounce = false;
            continue;
        }

        // Glass: cauchyB > 0 means per-channel chromatic dispersion -
        // each hero wavelength sees its own IOR, refracts at its own
        // angle, and traces an independent sub-path. After this
        // surface the four channels physically separate, producing
        // the visible rainbow in caustics.
        //
        // cauchyB == 0 (the default if a material doesn't opt into
        // dispersion) means all four channels share the same IOR
        // and refract identically. Short-circuit to a single shared
        // sub-path with per-channel albedo, like the mirror branch -
        // matches RGB-mode glass cost without changing render output
        // in expectation.
        if (materials[matIdx].transparent != 0) {
            float baseIor = materials[matIdx].ior;
            float cb = materials[matIdx].cauchyB;

            if (cb == 0.0) {
                // Single shared sub-path. Mirrors the cauchyB == 0
                // branch on the CPU side.
                DielectricOut b = dielectricBounce(rd, N, hit, entering, baseIor, rand(seed));
                rd = b.dir;
                ro = b.origin;
                throughput *= vec4(albedoAt(matIdx, lambdas.x),
                                   albedoAt(matIdx, lambdas.y),
                                   albedoAt(matIdx, lambdas.z),
                                   albedoAt(matIdx, lambdas.w));
                firstBounce = false;
                continue;
            }

            int remainingDepth = uDepth - bounce - 1;
            vec4 splitRad = vec4(0.0);
            for (int k = 0; k < 4; k++) {
                float lam = lambdas[k];
                float ior = cauchyIor(baseIor, cb, lam);
                DielectricOut b = dielectricBounce(rd, N, hit, entering, ior, rand(seed));
                float subRad = tracePathSpectralSingle(b.origin, b.dir, lam,
                                                       remainingDepth, bounce + 1,
                                                       seed);
                splitRad[k] = subRad * albedoAt(matIdx, lam);
            }
            radiance += throughput * splitRad;
            return radiance; // sub-paths handled the rest
        }

        // Diffuse path. Cache albedo at all 4 lambdas once per hit.
        vec4 albedoLam = vec4(albedoAt(matIdx, lambdas.x),
                              albedoAt(matIdx, lambdas.y),
                              albedoAt(matIdx, lambdas.z),
                              albedoAt(matIdx, lambdas.w));

        // Direct lighting.
        vec4 directLo = vec4(0.0);
        if (uTotalLightArea > 0.0) {
            for (int s = 0; s < uShadowSamples; s++) {
                vec3 sampleP, sampleN, sampleEmissiveRGB;
                int sampleMatIdx;
                sampleAreaLight(seed, sampleP, sampleN, sampleEmissiveRGB, sampleMatIdx);

                vec3 Li = sampleP - hit;
                vec3 wi = normalize(Li);
                float cosTheta = max(0.0, dot(wi, N));
                float lightDist2 = dot(Li, Li);
                vec3 shadowOrigin = (cosTheta <= 0.0) ? hit - N * 1e-3 : hit + N * 1e-3;

                vec3 sh, sN;
                int sMat;
                bool occluded = false;
                int litMatIdx = -1;
                if (sceneIntersect(shadowOrigin, wi, sh, sN, sMat)) {
                    vec3 d = sh - shadowOrigin;
                    float occluderDist2 = dot(d, d);
                    if (occluderDist2 < lightDist2 - 1e-3) {
                        if (!any(greaterThan(materials[sMat].emissive.rgb, vec3(0.0))))
                            occluded = true;
                        else
                            litMatIdx = sMat;
                    }
                }

                if (!occluded) {
                    float cosLight = max(0.0, dot(sampleN, -wi));
                    float G = (cosTheta * cosLight) / lightDist2;
                    float misWeight = 1.0;
                    if (uUseMIS != 0 && cosLight > 1e-6) {
                        float pdfLight = lightDist2 / (cosLight * uTotalLightArea);
                        float pdfBrdf  = cosTheta / PI;
                        misWeight = (pdfLight * pdfLight) /
                                    (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                    }
                    // Per-channel direct contribution. The picked
                    // light's matIdx threads through sampleAreaLight
                    // so the emissive spectrum lookup is exact, not
                    // an RGB-luminance approximation. Falls through
                    // to the RGB emissive (luminance-weighted) only
                    // if matIdx is unavailable, which currently
                    // shouldn't happen but the fallback keeps the
                    // path tracer robust.
                    vec4 emitLam;
                    if (sampleMatIdx >= 0) {
                        emitLam = vec4(emissiveAt(sampleMatIdx, lambdas.x),
                                       emissiveAt(sampleMatIdx, lambdas.y),
                                       emissiveAt(sampleMatIdx, lambdas.z),
                                       emissiveAt(sampleMatIdx, lambdas.w));
                    } else {
                        float emitL0 = sampleEmissiveRGB.x * 0.30
                                     + sampleEmissiveRGB.y * 0.59
                                     + sampleEmissiveRGB.z * 0.11;
                        emitLam = vec4(emitL0);
                    }
                    vec4 contrib = (albedoLam / PI) * emitLam * G * uTotalLightArea * misWeight;
                    directLo += contrib;
                }
            }
            directLo /= float(uShadowSamples);
        }
        radiance += throughput * directLo;

        // Russian roulette uses the hero (.x) channel reflectance so
        // all 4 channels share the termination decision; non-hero
        // channels carry their scaled throughput along.
        if (uUseRussian != 0 && bounce >= 1) {
            float p = clamp(albedoLam.x, 0.05, 0.95);
            if (rand(seed) > p) break;
            throughput /= p;
        }

        // Indirect bounce.
        vec3 newDir;
        if (uUseStratified != 0 && firstBounce) {
            newDir = sampleHemisphereFrom(N, pr1, pr2);
        } else {
            newDir = sampleHemisphere(N, seed);
        }
        firstBounce = false;

        ro = hit + N * 1e-3;
        rd = newDir;
        throughput *= albedoLam;
    }
    return radiance;
}

// Tone mapping moved to CPU-side post-readback so the GPU can emit HDR
// linear radiance into uOutput, which is what OIDN's HDR mode needs as
// input. See ToneMap.h for the curves; both CPU and GPU paths now go
// through the same code.

void main() {
    ivec2 pix = ivec2(int(gl_GlobalInvocationID.x) + uXOffset,
                      int(gl_GlobalInvocationID.y) + uYOffset);
    // 2D tile dispatch: bounds-check against the tile rectangle AND the
    // image extent (the tile may end mid-image with extra invocations).
    if (pix.x >= uXEnd || pix.x >= uWidth ||
        pix.y >= uYEnd || pix.y >= uHeight) return;

    // Aux capture for OIDN. One deterministic primary ray through the
    // pixel center, sceneIntersect, write albedo + shading normal. Done
    // before the noisy AA loop so OIDN gets clean per-pixel features.
    if (uWriteAux != 0) {
        float aspect = float(uWidth) / float(uHeight);
        float scale = tan(PI / 180.0 * 0.5 * uFov);
        float ax = ((2.0 * (float(pix.x) + 0.5) / float(uWidth)) - 1.0) * scale * aspect;
        float ay = -((2.0 * (float(pix.y) + 0.5) / float(uHeight)) - 1.0) * scale;
        vec3 ard = normalize(vec3(ax, ay, -1.0));
        vec3 aro = uOrigin;
        vec3 ahit, aN;
        int aMatIdx;
        if (sceneIntersect(aro, ard, ahit, aN, aMatIdx)) {
            if (dot(ard, aN) > 0.0) aN = -aN;
            imageStore(uAlbedoOut, pix, vec4(materials[aMatIdx].albedo.rgb, 1.0));
            imageStore(uNormalOut, pix, vec4(aN, 0.0));
        } else {
            // No intersect: zero both. Matches CPU's background-hit
            // behavior so OIDN sees the same sentinel for sky pixels.
            imageStore(uAlbedoOut, pix, vec4(0.0));
            imageStore(uNormalOut, pix, vec4(0.0));
        }
    }

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
            if (uUseSpectral != 0) {
                if (uHeroSamples <= 1) {
                    // Single-wavelength legacy path. Pick one lambda
                    // uniformly in the visible, route to the existing
                    // single-channel kernel (already used internally
                    // by the hero kernel for glass dispersion path
                    // splits, so it's well-exercised). Strictly worse
                    // noise than hero at the same sample count, but
                    // exposed for benchmarking and visual A/B against
                    // the hero default.
                    //
                    // Primary ray construction matches tracePathSpectral
                    // (same pix-to-direction transform); we inline it
                    // here because tracePathSpectralSingle takes the
                    // ray as origin+direction rather than a pixel
                    // coordinate.
                    float kSpan = kLambdaMax - kLambdaMin;
                    float lambda = kLambdaMin + rand(seed) * kSpan;
                    float aspect = float(uWidth) / float(uHeight);
                    float scale  = tan(uFov * 0.5);
                    vec2 nd = (jpix * 2.0 - vec2(uWidth, uHeight)) /
                              vec2(uWidth, uHeight);
                    vec3 dir = normalize(vec3(nd.x * scale * aspect,
                                              -nd.y * scale,
                                              -1.0));
                    float rad = tracePathSpectralSingle(uOrigin, dir, lambda,
                                                        uDepth, 0, seed);
                    accum += singleLambdaXYZ(lambda, rad);
                } else {
                    // Hero wavelength sampling. Pick a hero lambda
                    // uniformly in the visible range; the other 3
                    // are stratified offsets wrapped around.
                    // tracePathSpectral returns 4 scalar radiances;
                    // convert each to a CIE XYZ contribution via the
                    // observer at its lambda and average across the
                    // 4 channels (1/N is the sampling weight). The
                    // accumulator runs in XYZ for the duration of
                    // the AA loop; we convert mean to linear sRGB
                    // once after the loop.
                    float kSpan = kLambdaMax - kLambdaMin;
                    float kStride = kSpan / 4.0;
                    vec4 lambdas;
                    lambdas.x = kLambdaMin + rand(seed) * kSpan;
                    lambdas.y = lambdas.x + kStride; if (lambdas.y > kLambdaMax) lambdas.y -= kSpan;
                    lambdas.z = lambdas.x + kStride * 2.0; if (lambdas.z > kLambdaMax) lambdas.z -= kSpan;
                    lambdas.w = lambdas.x + kStride * 3.0; if (lambdas.w > kLambdaMax) lambdas.w -= kSpan;
                    vec4 rad = tracePathSpectral(jpix, r1, r2, lambdas, seed);
                    vec3 xyz = singleLambdaXYZ(lambdas.x, rad.x)
                             + singleLambdaXYZ(lambdas.y, rad.y)
                             + singleLambdaXYZ(lambdas.z, rad.z)
                             + singleLambdaXYZ(lambdas.w, rad.w);
                    accum += xyz * 0.25;
                }
            } else {
                accum += tracePath(jpix, r1, r2, seed);
            }
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
    // Spectral mode: mean is in CIE XYZ. Convert to linear sRGB so
    // OIDN, the readback path, the CPU tone-map, and the PNG encode
    // see the same color space the RGB path produces.
    if (uUseSpectral != 0)
        mean = xyzToLinearSRGB(mean);
    // HDR linear radiance. Tone mapping happens on CPU after readback
    // so OIDN gets the full pre-tone-map signal as input.
    imageStore(uOutput, pix, vec4(mean, 1.0));
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
        float cauchyB;
        // 61-sample tabulated spectra mirroring the GLSL struct above.
        // Indices 0..60 cover 400-700 nm at 5 nm spacing; 61..63 are
        // unused padding to keep each array's footprint at 256 bytes
        // (a multiple of 16) so the total struct stays std430-clean.
        // Total: 16+16+16 + 256+256 = 560 bytes, which is 16 * 35.
        // Static-assert below guards against accidental layout drift.
        float albedoSpectrum[64];
        float emissiveSpectrum[64];
    };
    static_assert(sizeof(GpuMaterial) == 560,
                  "GpuMaterial size must be 560 bytes (3 vec4 + 2 float[64])");
    static_assert(sizeof(GpuMaterial) % 16 == 0,
                  "GpuMaterial size must be a multiple of 16 bytes for std430 array stride");
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
        int   matIdx;         // materials[] index for spectral emission
    };
    struct GpuLightTriangle
    {
        float v0[4];
        float v1[4];
        float v2[4];
        float flatN[4];   // xyz=N, w=cumulative area within parent light
        float emissive[4];
        int   matIdx;     // materials[] index for spectral emission
        int   _pad[3];
    };

    // Mirrors Renderer.cpp's compressZone. strftime("%Z") on Windows
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

// ---------- OpenglRenderer ---------------------------------------------------

OpenglRenderer::OpenglRenderer(int width, int height,
                               int depth, int samples, int shadowSamples,
                               GLFWwindow *sharedContext)
    : _width{width}, _height{height},
      _maxDepth{depth}, _samples{samples}, _shadowSamples{shadowSamples},
      _sharedContext{sharedContext}
{
}

OpenglRenderer::~OpenglRenderer()
{
    // We don't tear down GL state in the destructor because the GL context
    // probably isn't current on the calling thread. The OS reclaims it on
    // process exit; the cost is negligible for a single-render lifetime.
}

bool OpenglRenderer::initGL()
{
    if (_initialized) return true;

    if (!loadGlFunctions())
    {
        std::cerr << "OpenglRenderer: failed to load GL functions" << std::endl;
        return false;
    }

    _program = compileComputeShader(kComputeShaderSrc);
    if (!_program) return false;

    // Three RGBA16F textures. uOutput holds HDR linear radiance (pre-tone-
    // map, so OIDN HDR mode gets the right input). uAlbedo/uNormal are
    // populated only when --oidn is on (writes gated by uWriteAux), but
    // allocated unconditionally so we don't reallocate per render.
    auto allocFloatTex = [&](unsigned &tex) {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _width, _height, 0,
                     GL_RGBA, GL_FLOAT, nullptr);
    };
    allocFloatTex(_outputTex);
    allocFloatTex(_albedoTex);
    allocFloatTex(_normalTex);
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

void OpenglRenderer::uploadScene(const Scenes::SceneData &scene, float &outTotalLightArea)
{
    // Material upload: one-to-one with scene.materials. The CPU side
    // already dedupes (every primitive carries a matIdx into the
    // shared registry), so we just translate Material -> GpuMaterial
    // in order. SSBO indices match scene.materials indices.
    std::vector<GpuMaterial> mats;
    mats.reserve(scene.materials.size());
    for (const auto &m : scene.materials)
    {
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
        gm.cauchyB = m.cauchyB;
        // Fill the 61-sample tabulated spectra. Source depends on whether
        // the material has a measured SPD (preferred) or only an RGB
        // albedo / emissive (fall back to evaluating the Jakob fit at
        // each wavelength). Both paths produce the same shape on disk,
        // so the GPU shader has only one lookup function and parity
        // with the CPU side is true: same per-wavelength values feed
        // both backends.
        //
        // Albedo samples are clamped to [0, 1] here so the GLSL hot
        // path doesn't need a per-bounce min() call (CPU clamps inside
        // Material::albedoAt for the same reason). Emissive samples are
        // not clamped because radiance is HDR.
        for (int i = 0; i < Spectrum::kSamples; i++)
        {
            float lambda = Spectrum::lambdaAt(i);
            float a = m.useTabulatedAlbedo
                      ? m.tabulatedAlbedo[i]
                      : RGBToSpectrum::evalSigmoidFit(m.albedoFit, lambda);
            gm.albedoSpectrum[i] = std::min(std::max(0.f, a), 1.f);

            float e = m.useTabulatedEmissive
                      ? m.tabulatedEmissive[i]
                      : RGBToSpectrum::evalSigmoidFit(m.emissiveFit, lambda);
            gm.emissiveSpectrum[i] = std::max(0.f, e);
        }
        // Tail padding (61..63) stays zero-initialized via the {} brace
        // construction above.
        mats.push_back(gm);
    }

    std::vector<GpuSphere> gpuSpheres;
    for (const auto &s : scene.spheres)
    {
        GpuSphere gs{};
        gs.center[0] = s.center[0];
        gs.center[1] = s.center[1];
        gs.center[2] = s.center[2];
        gs.center[3] = s.radius();
        gs.matIdx = s.matIdx;
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

    // Plane lights front-loaded so coplanar ties favor the light.
    std::vector<int> areaLightMatIdx(scene.areaLights.size(), -1);
    for (size_t li = 0; li < scene.areaLights.size(); li++)
    {
        const auto &L = scene.areaLights[li];
        if (L.kind != Scenes::AreaLightKind::Plane) continue;
        areaLightMatIdx[li] = L.plane.matIdx;
        addPlane(L.plane, L.plane.matIdx);
    }
    for (const auto &w : scene.walls)
        addPlane(w, w.matIdx);

    std::vector<GpuTriangle> gpuTris;
    for (const auto &t : scene.triangles)
    {
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
        gt.matIdx = t.matIdx;
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
    for (size_t li = 0; li < scene.areaLights.size(); li++)
    {
        const auto &L = scene.areaLights[li];
        GpuLight gl{};
        gl.normal_area[3] = L.totalArea;
        totalLightArea += L.totalArea;
        if (L.kind == Scenes::AreaLightKind::Plane)
        {
            gl.kind = 0;
            const Vec3f &u = L.plane.getU();
            const Vec3f &v = L.plane.getV();
            const Material &lightMat = scene.materials[L.plane.matIdx];
            for (int i = 0; i < 3; i++)
            {
                gl.origin[i] = L.plane.origin[i];
                gl.u[i] = u[i];
                gl.v[i] = v[i];
                gl.normal_area[i] = L.plane.N[i];
                gl.emissive[i] = lightMat.emissive[i];
            }
            gl.firstTri = 0;
            gl.count = 0;
            gl.matIdx = L.plane.matIdx;
        }
        else
        {
            gl.kind = 1;
            gl.firstTri = (int)gpuLightTris.size();
            gl.count = (int)L.triangles.size();
            if (!L.triangles.empty())
            {
                const Material &lightMat = scene.materials[L.triangles.front().matIdx];
                for (int i = 0; i < 3; i++)
                    gl.emissive[i] = lightMat.emissive[i];
            }
            gl.matIdx = -1; // per-triangle matIdx populated below

            for (size_t ti = 0; ti < L.triangles.size(); ti++)
            {
                const Triangle &t = L.triangles[ti];
                const Material &triMat = scene.materials[t.matIdx];
                GpuLightTriangle glt{};
                for (int i = 0; i < 3; i++)
                {
                    glt.v0[i] = t.v0[i];
                    glt.v1[i] = t.v1[i];
                    glt.v2[i] = t.v2[i];
                    glt.flatN[i] = t.flatN[i];
                    glt.emissive[i] = triMat.emissive[i];
                }
                glt.flatN[3] = L.cumulativeArea[ti];
                glt.matIdx = t.matIdx;
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

void OpenglRenderer::render(const Scenes::SceneData &scene,
                         std::chrono::steady_clock::time_point start,
                         const std::string &outputDir)
{
    lastOutputPath.clear();

    if (!_sharedContext)
    {
        std::cerr << "OpenglRenderer: no shared OpenGL context" << std::endl;
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

    // Bind output + aux images. uOutput at binding 0 always; uAlbedoOut /
    // uNormalOut at 1/2 always too (the shader gates writes via uWriteAux,
    // but the binding has to be live or the write would be a GL error).
    glBindImageTexture(0, _outputTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glBindImageTexture(1, _albedoTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glBindImageTexture(2, _normalTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

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
    // Note: tone-map (ACES vs Reinhard) used to be a GLSL uniform but is
    // now applied CPU-side after OIDN. useACES still drives the choice;
    // the GPU just emits HDR linear regardless.
    // Aux capture only when OIDN is on. The aux pass is one extra primary
    // ray per pixel (~free compared to the main loop), but skip it when
    // not needed so the GPU isn't doing useless writes.
    setI("uWriteAux",       useOIDN ? 1 : 0);
    setI("uUseSpectral",    useSpectral ? 1 : 0);
    // heroSamples: 4 = stratified hero (default), 1 = single-wavelength
    // legacy (routes the primary ray through tracePathSpectralSingle).
    // Other values clamp to 4 in the shader's branch test (uHeroSamples
    // <= 1 triggers the single-wavelength path; everything else hero).
    setI("uHeroSamples",    std::clamp(heroSamples, 1, 4));
    setI("uAaSamples",      std::max(1, aaSamples));
    setI("uUseAdaptive",    useAdaptive ? 1 : 0);
    setI("uStrata",         useStratified
                            ? std::max(1, (int)std::round(std::sqrt((float)_samples)))
                            : 0);

    // Dispatch in 2D tiles so we stay well under any GPU TDR (Timeout
    // Detection and Recovery) window. Windows defaults to a 2-second TDR;
    // a single dispatch that exceeds that is killed by the driver, taking
    // the whole process with it (no exception, no GL error. the kernel
    // just resets the GPU).
    //
    // 1D row-strip dispatch (the previous approach) hits a floor of 1 row
    // per dispatch on heavy presets, and at high resolutions one row can
    // still be too much work in a single submit. 2D tiles let us shrink
    // both dimensions, so a Picture-class render on a 70k-tri mesh at 1080
    // dispatches small enough chunks regardless of resolution.
    //
    // Tile size is computed from per-pixel work x BVH overhead. The
    // target below sets the per-dispatch budget. Lower = shorter tile
    // times = more cold-start margin against the 2-sec Windows TDR
    // cliff. The previous 0.5-sec target hit TDR in practice because
    // (a) GPU clock varies under thermal load (the same shader runs
    // ~2x slower a few seconds in than at peak), (b) first-dispatch
    // driver overhead inflates the early tiles, and (c) workload
    // variance across the image means some tiles do more work than
    // the average.
    //
    // Calibration source: cornell-spheres at 1080^2 x 393216 work units
    // took 245 sec ~ 1.87e9 work-units-per-second.
    //
    // aaSamples multiplies primary-ray count; folds into per-pixel work
    // for tile sizing so AA-enabled renders shrink tiles to compensate.
    int aaMult = std::max(1, aaSamples);
    int workPerPixel = std::max(1, _samples * _maxDepth * _shadowSamples * aaMult);

    // Per-ray cost depends on how many primitives sceneIntersect tests.
    // Spheres + planes are linear in count; triangles go through the BVH
    // at log(N/leafSize) cost. Without this multiplier, tile sizing
    // assumes a baseline-cornell scene (~7 primitives) and over-sizes
    // tiles for plane-heavy scenes. cornell-spec with 18 primitives
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

    // Adaptive sampling adds Welford running-variance bookkeeping (sqrt,
    // div, branch on threshold) inside the per-sample inner loop. The
    // average case is faster (early-exit on converged pixels), but the
    // worst case (deep-shadow corners that take all AA samples) is
    // slightly slower than the non-adaptive baseline. The 1.15x penalty
    // budgets for the worst case so tile sizing doesn't over-commit on
    // adaptive renders.
    if (useAdaptive) effectivePerPixel *= 1.15;

    // OIDN aux capture is one extra primary-ray sceneIntersect at the
    // top of every pixel. Constant cost per pixel (not multiplied by
    // samples), so the absolute add is tiny relative to the main loop.
    // Folded in for completeness and so the formula is honest about
    // what the GPU is doing.
    if (useOIDN) effectivePerPixel += (primMult + bvhMult);

    // Spectral mode does ~4 spectrum lookups per bounce per channel
    // (albedo + emission at 4 hero lambdas) plus per-channel scalar
    // multiplies. Empirically the per-bounce shader cost goes up
    // ~2.5x vs the RGB path. Bump effective work proportionally so
    // tile sizing keeps the per-tile time near the budget.
    if (useSpectral) effectivePerPixel *= 2.5;

    // 1.5e8 work units per dispatch ~= 0.08 sec on the calibration GPU.
    // 25x safety margin against the 2-sec TDR cliff. Cold-start
    // overhead (first-dispatch shader compile, GPU clock ramp, driver
    // resource validation) can inflate the first tile by 5-10x, so we
    // budget against an inflated first dispatch rather than the
    // steady-state average. Was 2.8e8 (0.15 sec / 13x margin), which
    // crashed at d=4 s=1024 S=32 + AA + adaptive + OIDN on tile 1.
    constexpr double kTargetWorkPerDispatch = 1.5e8;
    double maxPixelsD = kTargetWorkPerDispatch / effectivePerPixel;
    int maxPixelsPerDispatch = std::max(64, (int)maxPixelsD);
    int tileSide = (int)std::sqrt((double)maxPixelsPerDispatch);
    // Tile floor: heavy per-pixel work needs aggressive small tiles even at
    // the cost of more dispatch overhead. Light work uses bigger tiles to
    // avoid drowning in glFinish overhead. Three tiers:
    //
    //   < 5e5      : light work (RGB cornell at modest samples) -> floor 16
    //   < 2e6      : heavy     (RGB at high samples, spectral simple) -> floor 8
    //   >= 2e6     : very heavy (spectral + aa + adaptive + OIDN on a
    //                complex scene at production samples) -> floor 4
    //
    // The 2e6 tier was added after spectral cornell-spec at d=4 s=2048
    // S=16 aa=4 hit TDR at a tile size of 8: the formula wanted 6 but
    // the floor pinned 8, putting each dispatch ~75% over the 1.5e8
    // work-per-dispatch target. Floor of 4 gives that case 16-pixel
    // tiles which sit comfortably under budget.
    int minTileSide;
    if (effectivePerPixel >= 2e6)      minTileSide = 4;
    else if (effectivePerPixel > 5e5)  minTileSide = 8;
    else                               minTileSide = 16;
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

    // Cold-start warmup dispatch. The first compute dispatch on this
    // shader pays one-time costs the steady-state work formula doesn't
    // account for: JIT to GPU machine code, resource state validation,
    // GPU clock ramp from idle. On nvlddmkm + heavy shaders (all
    // techniques on) those costs add enough latency that a normal-
    // sized first tile blows past the 2-sec TDR cliff even with the
    // 1.5e8 work budget. Run one pixel of full work up front; the
    // shader gets fully exercised, the driver finishes its lazy JIT,
    // and the real tile loop starts on a warm pipeline.
    //
    // Why one pixel and not a no-op dispatch: drivers can lazy-compile
    // only the code paths actually executed. A one-pixel dispatch with
    // real uniforms exercises the full shader body so JIT covers
    // everything before the real loop hits it.
    setI("uXOffset", 0);
    setI("uXEnd",    1);
    setI("uYOffset", 0);
    setI("uYEnd",    1);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glFinish();

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
                // Technique list, duplicated short form so the popup
                // shows what was on without us having to plumb settings
                // back from the GUI. If the next line of output is a
                // forced process death, this is what the user sees.
                std::string techs;
                auto addT = [&](bool on, const char *label) {
                    if (!on) return;
                    if (!techs.empty()) techs += ",";
                    techs += label;
                };
                addT(useDenoise,    "denoise");
                addT(useMIS,        "mis");
                addT(useRussian,    "russian");
                addT(useStratified, "strat");
                addT(useACES,       "aces");
                if (aaSamples > 1)
                    addT(true, ("aa" + std::to_string(aaSamples)).c_str());
                addT(useAdaptive,   "adaptive");
                addT(useOIDN,       "oidn");
                if (techs.empty()) techs = "(none)";

                char act[768];
                std::snprintf(act, sizeof(act),
                    "Rendering '%s' v%s at d=%d s=%d S=%d w=%d h=%d (GPU)\n"
                    "Techniques: %s\n"
                    "Triangles: %d (BVH nodes: %d). Lights: %d.\n"
                    "Tile size: %d. Tile %d/%d, "
                    "pixels x=[%d..%d) y=[%d..%d).",
                    scene.name.c_str(), scene.version.c_str(),
                    _maxDepth, _samples, _shadowSamples, _width, _height,
                    techs.c_str(),
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

    // Readback. GPU now stores HDR linear radiance + (optionally) aux as
    // RGBA16F, so we read GL_FLOAT into 4-component float buffers and then
    // strip the alpha into Vec3f arrays for downstream processing.
    auto readbackToVec3 = [&](unsigned tex) {
        std::vector<float> tmp((size_t)_width * _height * 4);
        glBindTexture(GL_TEXTURE_2D, tex);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, tmp.data());
        std::vector<Vec3f> out((size_t)_width * _height);
        for (size_t i = 0; i < out.size(); i++)
            out[i] = Vec3f(tmp[i * 4 + 0], tmp[i * 4 + 1], tmp[i * 4 + 2]);
        return out;
    };
    std::vector<Vec3f> hdr = readbackToVec3(_outputTex);
    if (!checkGl("readback color")) {
        glfwMakeContextCurrent(nullptr);
        return;
    }
    std::vector<Vec3f> albedoBuf, normalBuf;
    if (useOIDN)
    {
        albedoBuf = readbackToVec3(_albedoTex);
        normalBuf = readbackToVec3(_normalTex);
        if (!checkGl("readback aux")) {
            glfwMakeContextCurrent(nullptr);
            return;
        }
    }

    glfwMakeContextCurrent(nullptr);

    auto end = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "GPU render took " << elapsedMs << " ms" << std::endl;

    // OIDN on the GPU path now mirrors the CPU path: HDR linear input + aux
    // (albedo + shading normal at first hit) -> OIDN HDR-mode denoise -> CPU
    // tone map. Earlier the GPU did in-shader tone-mapping and ran OIDN in
    // LDR mode on the post-tone-map 8-bit signal, which threw away most of
    // the input dynamic range and skipped the aux features that prevent
    // OIDN from blurring out edges. Cost: one extra primary ray for aux,
    // 3x the readback bandwidth (RGBA16F x 3 textures), and the OIDN
    // invocation is on the CPU regardless.
    if (useOIDN)
    {
        if (!OidnDenoise::isAvailable())
        {
            std::cerr << "warning: --oidn requested but binary was not "
                         "built with PCR_USE_OIDN=ON; skipping.\n";
        }
        else
        {
            OidnDenoise::denoise(hdr, albedoBuf, normalBuf, _width, _height);
        }
    }

    // Tone map on CPU using the same shared curves the CPU renderer uses.
    // Done after OIDN so the denoiser sees full HDR; done before bilateral
    // (in the legacy useDenoise branch) because the bilateral filter
    // operates on 8-bit RGB.
    for (auto &c : hdr)
    {
        if (useACES) ToneMap::aces(c);
        else         ToneMap::reinhard(c);
    }

    std::vector<unsigned char> rgb((size_t)_width * _height * 3);
    auto cl = [](float v) {
        if (v < 0.f) v = 0.f;
        if (v > 1.f) v = 1.f;
        return (unsigned char)(v * 255.f + 0.5f);
    };
    for (size_t i = 0; i < (size_t)_width * _height; i++)
    {
        rgb[i * 3 + 0] = cl(hdr[i][0]);
        rgb[i * 3 + 1] = cl(hdr[i][1]);
        rgb[i * 3 + 2] = cl(hdr[i][2]);
    }

    if (!useOIDN && useDenoise)
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
    // ACES is the only technique that ends up in the filename, since it
    // changes the look of the image meaningfully (different tone curve)
    // and naming it makes side-by-side comparison easier. AA / adaptive
    // / OIDN all affect quality but produce the same "color" of image,
    // so they live in the PNG metadata only.
    if (useACES)
        filename += "-aces";
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

void OpenglRenderer::destroyGL()
{
    if (_program) { glDeleteProgram(_program); _program = 0; }
    if (_outputTex) { glDeleteTextures(1, &_outputTex); _outputTex = 0; }
    if (_albedoTex) { glDeleteTextures(1, &_albedoTex); _albedoTex = 0; }
    if (_normalTex) { glDeleteTextures(1, &_normalTex); _normalTex = 0; }
    if (_sphereSSBO)   { glDeleteBuffers(1, &_sphereSSBO);   _sphereSSBO = 0; }
    if (_planeSSBO)    { glDeleteBuffers(1, &_planeSSBO);    _planeSSBO = 0; }
    if (_materialSSBO) { glDeleteBuffers(1, &_materialSSBO); _materialSSBO = 0; }
    _initialized = false;
}
