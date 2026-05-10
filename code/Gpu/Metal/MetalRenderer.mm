// MetalRenderer.mm. Apple-only path tracer using a Metal compute shader.
//
// Public interface in MetalRenderer.h matches OpenglRenderer so
// Gui/main.cpp just sees `class GpuRenderer` (resolved by the typedef
// shim in Gpu/GpuRenderer.h) and doesn't have to care which backend it
// holds.
//
// Architecture is a one-to-one port of the OpenGL backend:
//   - Scene -> seven MTLBuffers (spheres, planes, materials, triangles,
//     BVH nodes, lights, light-triangles), one MTLTexture per output
//     channel (color + OIDN aux: albedo, normal). All RGBA16F-equivalent
//     so the readback path keeps the full HDR linear signal for OIDN.
//   - One compute pipeline runs the per-pixel path tracer; same RGB +
//     spectral hero-wavelength branches as the GLSL kernel.
//   - Dispatch in 2D tiles. Apple Silicon doesn't have a Windows-style
//     TDR cliff but tiles still help cancel + progress responsiveness,
//     and they keep dispatch sizes within Metal's per-encoder limits on
//     extra-heavy renders.
//   - Tone-map and OIDN happen on the CPU after readback (matches
//     OpenGL backend).
//
// What's NOT here yet:
//   - MetalRT (hardware ray tracing) - M3+ only. Compute path works on
//     M1+ which is the v1.3.1 target hardware.
//   - Heterogeneous CPU+GPU work splitting. Pure GPU; CPU does OIDN +
//     tone-map + PNG only.
//
// Why the .mm extension: this file is Objective-C++. The MetalRenderer
// class itself is pure C++ but it stores Foundation/Metal id<...>
// objects in its pImpl, which requires Obj-C compiler support. CMake
// turns -fobjc-arc on for this single source file so the id<...>
// members are refcounted automatically; no manual retain/release.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Gpu/Metal/MetalRenderer.h"
#include "Includes/lodepng.h"
#include "Includes/Denoise.h"
#include "Includes/OidnDenoise.h"
#include "Includes/ToneMap.h"
#include "Includes/Spectrum.h"
#include "Includes/RGBToSpectrum.h"

namespace fs = std::filesystem;

// ---------- Embedded MSL compute shader -----------------------------------
//
// Direct line-by-line port of the GLSL kernel in
// Opengl/OpenglRenderer.cpp. Same data layouts, same trace functions,
// same RGB + spectral hero-wavelength branches. Translation rules
// applied uniformly:
//   - vec2/3/4   -> float2/3/4
//   - ivec2      -> int2
//   - inout T    -> thread T &
//   - out T      -> thread T &
//   - imageStore -> texture.write
//   - SSBOs and uniforms are bundled into a single Scene struct that's
//     passed by reference to every helper instead of GLSL globals
//   - kernel entry takes Scene buffers + uniforms + textures explicitly
//
// The MSL struct layouts are byte-equivalent to the GLSL std430 layouts
// so the host-side POD copies upload verbatim. Static-asserts below
// guard against drift.
static NSString *const kMSLSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846;
constant float kLambdaMin = 400.0;
constant float kLambdaMax = 700.0;
constant int   kSpecSamples = 61;
constant float kSpecStep = 5.0;

struct GpuSphere {
    float4 center_radius;
    int    matIdx;
    int    _pad0, _pad1, _pad2;
};

struct GpuPlane {
    float4 origin;
    float4 u;
    float4 v;
    float4 normal_area;
    int    matIdx;
    int    _pad0, _pad1, _pad2;
};

struct GpuMaterial {
    float4 albedo;
    float4 emissive;
    int    metallic;
    int    transparent;
    float  ior;
    float  cauchyB;
    // 61-sample tabulated reflectance / emission, 400-700 nm at 5 nm
    // spacing. Values 61..63 are unused padding to keep the struct's
    // total size a multiple of 16 bytes (matches std430 GLSL layout).
    float  albedoSpectrum[64];
    float  emissiveSpectrum[64];
};

struct GpuTriangle {
    float4 v0;
    float4 v1;
    float4 v2;
    float4 n0;
    float4 n1;
    float4 n2;
    float4 flatN;
    int    matIdx;
    int    smooth_;
    int    _pad0, _pad1;
};

struct GpuBvhNode {
    float4 boxMin;
    float4 boxMax;
    int    leftOrFirst;
    int    rightChild;
    int    count;
    int    _pad;
};

struct GpuLight {
    float4 origin;
    float4 u;
    float4 v;
    float4 normal_area;
    float4 emissive;
    int    kind;
    int    firstTri;
    int    count;
    int    matIdx;
};

struct GpuLightTriangle {
    float4 v0;
    float4 v1;
    float4 v2;
    float4 flatN;
    float4 emissive;
    int    matIdx;
    int    _pad0, _pad1, _pad2;
};

// Uniforms total: 144 bytes (33 active fields = 132 bytes + 3 trailing
// padding ints to round up to a multiple of 16). Two reasons for the
// alignment padding:
//   - When this struct is the element type of a per-pass uniforms
//     array bound at offset i * sizeof(Uniforms), Apple Silicon's MSL
//     compiler issues vectorized 16-byte loads for the constant-
//     address-space struct read. Those loads expect the base offset
//     to be 16-aligned. Without padding to a multiple-of-16 size, only
//     a subset of array offsets would hit a clean alignment - the rest
//     read garbage uniforms and the kernel's pix.y >= u.yEnd early-out
//     kills those threads (visible as horizontal black stripes in
//     output where the compiler chose the vectorized path).
//   - Keeps the host POD layout below symmetric so memcpy-via-shared-
//     buffer is byte-equivalent on both sides.
//
// The trailing four fields (passIdx, aaIdx, sampleStart, sampleCount)
// are used only by path_trace_pass; the legacy path_trace ignores them.
struct Uniforms {
    int   width;
    int   height;
    float fov;
    float originX;
    float originY;
    float originZ;
    int   depth;
    int   samples;
    int   shadowSamples;
    int   sphereCount;
    int   planeCount;
    int   triangleCount;
    int   bvhNodeCount;
    int   lightCount;
    float totalLightArea;
    int   xOffset;
    int   xEnd;
    int   yOffset;
    int   yEnd;
    int   frameSeed;
    int   useMIS;
    int   useRussian;
    int   useStratified;
    int   strata;
    int   aaSamples;
    int   useAdaptive;
    int   writeAux;
    int   useSpectral;
    int   heroSamples;
    int   passIdx;
    int   aaIdx;
    int   sampleStart;
    int   sampleCount;
    int   _pad0, _pad1, _pad2;
};

// Bundle of all device buffers + uniforms threaded through every helper
// so the GLSL "global" SSBOs translate to a single explicit parameter.
struct Scene {
    constant Uniforms                &u;
    device const GpuSphere           *spheres;
    device const GpuPlane            *planes;
    device const GpuMaterial         *materials;
    device const GpuTriangle         *triangles;
    device const GpuBvhNode          *bvhNodes;
    device const GpuLight            *lights;
    device const GpuLightTriangle    *lightTriangles;
};

// ---- Spectrum lookup --------------------------------------------------

float lookupTabulated(thread const Scene &S, int matIdx, int src, float lambda) {
    if (lambda < kLambdaMin || lambda > kLambdaMax) return 0.0f;
    float t = (lambda - kLambdaMin) / kSpecStep;
    int   i = int(t);
    if (i >= kSpecSamples - 1) {
        return src == 0 ? S.materials[matIdx].albedoSpectrum[kSpecSamples - 1]
                        : S.materials[matIdx].emissiveSpectrum[kSpecSamples - 1];
    }
    float f = t - float(i);
    if (src == 0) {
        float a = S.materials[matIdx].albedoSpectrum[i];
        float b = S.materials[matIdx].albedoSpectrum[i + 1];
        return mix(a, b, f);
    } else {
        float a = S.materials[matIdx].emissiveSpectrum[i];
        float b = S.materials[matIdx].emissiveSpectrum[i + 1];
        return mix(a, b, f);
    }
}

float albedoAt(thread const Scene &S, int matIdx, float lambda) {
    return lookupTabulated(S, matIdx, 0, lambda);
}

float emissiveAt(thread const Scene &S, int matIdx, float lambda) {
    return lookupTabulated(S, matIdx, 1, lambda);
}

// ---- CIE / sRGB -------------------------------------------------------

float wymanG(float lambda, float mu, float s1, float s2) {
    float t = (lambda < mu) ? (lambda - mu) * s1 : (lambda - mu) * s2;
    return exp(-0.5f * t * t);
}

float3 cieObserverAt(float lambda) {
    float xb =  0.362f * wymanG(lambda, 442.0f, 0.0624f, 0.0374f)
              + 1.056f * wymanG(lambda, 599.8f, 0.0264f, 0.0323f)
              - 0.065f * wymanG(lambda, 501.1f, 0.0490f, 0.0382f);
    float yb =  0.821f * wymanG(lambda, 568.8f, 0.0213f, 0.0247f)
              + 0.286f * wymanG(lambda, 530.9f, 0.0613f, 0.0322f);
    float zb =  1.217f * wymanG(lambda, 437.0f, 0.0845f, 0.0278f)
              + 0.681f * wymanG(lambda, 459.0f, 0.0385f, 0.0725f);
    return float3(xb, yb, zb);
}

float3 singleLambdaXYZ(float lambda, float radiance) {
    const float kYBarIntegral = 106.895210f;
    return cieObserverAt(lambda) * radiance
         * (kLambdaMax - kLambdaMin) / kYBarIntegral;
}

float3 xyzToLinearSRGB(float3 xyz) {
    return float3(
         3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
        -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
         0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z
    );
}

// ---- PCG random -------------------------------------------------------

uint pcg(thread uint &state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rand(thread uint &seed) {
    return float(pcg(seed)) / 4294967295.0f;
}

float3 sampleHemisphereFrom(float3 N, float r1, float r2) {
    float r = sqrt(r1);
    float phi = 2.0f * PI * r2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0f, 1.0f - x*x - y*y));
    float3 helper = (abs(N.x) <= abs(N.y)) ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 T = normalize(cross(N, helper));
    float3 B = cross(N, T);
    return T * x + B * y + N * z;
}

float3 sampleHemisphere(float3 N, thread uint &seed) {
    return sampleHemisphereFrom(N, rand(seed), rand(seed));
}

// ---- Geometry intersection -------------------------------------------

bool intersectSphere(float3 ro, float3 rd, float3 center, float radius,
                     thread float &t) {
    float3 cp = center - ro;
    float rayLen = dot(rd, cp);
    float tSq = dot(cp, cp) - rayLen * rayLen;
    if (tSq > radius * radius) return false;
    float tDist = sqrt(radius * radius - tSq);
    float t0 = rayLen - tDist;
    float t1 = rayLen + tDist;
    if (t0 < 0.0f) t0 = t1;
    if (t0 < 0.0f) return false;
    t = t0;
    return true;
}

bool intersectTriangle(float3 ro, float3 rd, GpuTriangle t,
                       thread float &tt, thread float3 &hitOut, thread float3 &nOut) {
    const float EPS = 1e-6f;
    float3 e1 = t.v1.xyz - t.v0.xyz;
    float3 e2 = t.v2.xyz - t.v0.xyz;
    float3 pvec = cross(rd, e2);
    float det = dot(e1, pvec);
    if (abs(det) < EPS) return false;
    float invDet = 1.0f / det;
    float3 tvec = ro - t.v0.xyz;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    float3 qvec = cross(tvec, e1);
    float v = dot(rd, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    float thit = dot(e2, qvec) * invDet;
    if (thit <= EPS) return false;
    tt = thit;
    hitOut = ro + rd * thit;
    if (t.smooth_ != 0) {
        float w = 1.0f - u - v;
        nOut = normalize(t.n0.xyz * w + t.n1.xyz * u + t.n2.xyz * v);
    } else {
        nOut = t.flatN.xyz;
    }
    return true;
}

bool intersectPlane(float3 ro, float3 rd, GpuPlane p,
                    thread float &t, thread float3 &hitOut) {
    const float EPS = 1e-6f;
    float3 N = p.normal_area.xyz;
    float denom = dot(rd, N);
    if (abs(denom) <= EPS) return false;
    float tt = dot(p.origin.xyz - ro, N) / denom;
    if (tt <= EPS) return false;
    float3 tempHit = ro + rd * tt;
    float3 localHit = tempHit - p.origin.xyz;
    float s = dot(localHit, p.u.xyz) / dot(p.u.xyz, p.u.xyz);
    float q = dot(localHit, p.v.xyz) / dot(p.v.xyz, p.v.xyz);
    if (s < 0.0f || s > 1.0f || q < 0.0f || q > 1.0f) return false;
    t = tt;
    hitOut = tempHit;
    return true;
}

// ---- Dielectric optics -----------------------------------------------

float schlickFresnel(float cosTheta, float n1, float n2) {
    float F0 = (n1 - n2) / (n1 + n2);
    F0 = F0 * F0;
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

float cauchyIor(float baseIor, float cb, float lambdaNm) {
    return baseIor + cb * 1e4f / (lambdaNm * lambdaNm);
}

struct DielectricOut { float3 dir; float3 origin; };

DielectricOut dielectricBounce(float3 rayDir, float3 N, float3 hit,
                               bool entering, float ior, float fresnelRand) {
    float cosI = -dot(rayDir, N);
    float n1 = entering ? 1.0f : ior;
    float n2 = entering ? ior : 1.0f;
    float eta = n1 / n2;
    float sinT2 = eta * eta * (1.0f - cosI * cosI);
    DielectricOut o;
    if (sinT2 >= 1.0f) {
        o.dir = reflect(rayDir, N);
        o.origin = hit + N * 1e-3f;
        return o;
    }
    float F = schlickFresnel(cosI, n1, n2);
    if (fresnelRand < F) {
        o.dir = reflect(rayDir, N);
        o.origin = hit + N * 1e-3f;
    } else {
        float cosT = sqrt(1.0f - sinT2);
        o.dir = rayDir * eta + N * (eta * cosI - cosT);
        o.origin = hit - N * 1e-3f;
    }
    return o;
}

// ---- BVH traversal ---------------------------------------------------

bool intersectAabb(float3 ro, float3 rd, float3 mn, float3 mx, float segMax,
                   thread float &tNear) {
    float tmin = 0.0f;
    float tmax = segMax;
    for (int i = 0; i < 3; i++) {
        float invD = 1.0f / rd[i];
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

bool intersectBvh(thread const Scene &S, float3 ro, float3 rd, float closest_t,
                  thread float3 &hit, thread float3 &N, thread int &matIdx,
                  thread float &t_out) {
    if (S.u.bvhNodeCount == 0) return false;
    int   stackIdx[32];
    float stackTNear[32];
    int top = 0;
    {
        float tRoot;
        if (!intersectAabb(ro, rd, S.bvhNodes[0].boxMin.xyz, S.bvhNodes[0].boxMax.xyz, closest_t, tRoot))
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
        GpuBvhNode n = S.bvhNodes[stackIdx[top]];

        if (n.count > 0) {
            for (int i = 0; i < n.count; i++) {
                int triIdx = n.leftOrFirst + i;
                float tt;
                float3 ph, pn;
                if (intersectTriangle(ro, rd, S.triangles[triIdx], tt, ph, pn) && tt < closest) {
                    closest = tt;
                    hit = ph;
                    N = pn;
                    matIdx = S.triangles[triIdx].matIdx;
                    anyHit = true;
                }
            }
            continue;
        }

        GpuBvhNode cl = S.bvhNodes[n.leftOrFirst];
        GpuBvhNode cr = S.bvhNodes[n.rightChild];
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

bool sceneIntersect(thread const Scene &S, float3 ro, float3 rd,
                    thread float3 &hit, thread float3 &N, thread int &matIdx) {
    float closest = 1e30f;
    bool found = false;

    for (int i = 0; i < S.u.sphereCount; i++) {
        float t;
        if (intersectSphere(ro, rd, S.spheres[i].center_radius.xyz, S.spheres[i].center_radius.w, t)) {
            if (t < closest) {
                closest = t;
                hit = ro + rd * t;
                N = normalize(hit - S.spheres[i].center_radius.xyz);
                matIdx = S.spheres[i].matIdx;
                found = true;
            }
        }
    }
    for (int i = 0; i < S.u.planeCount; i++) {
        float t;
        float3 ph;
        if (intersectPlane(ro, rd, S.planes[i], t, ph)) {
            if (t < closest) {
                closest = t;
                hit = ph;
                N = S.planes[i].normal_area.xyz;
                matIdx = S.planes[i].matIdx;
                found = true;
            }
        }
    }

    if (S.u.triangleCount > 0) {
        float3 triHit, triN;
        int triMat;
        float triT;
        if (intersectBvh(S, ro, rd, closest, triHit, triN, triMat, triT) && triT < closest) {
            closest = triT;
            hit = triHit;
            N = triN;
            matIdx = triMat;
            found = true;
        }
    }
    return found;
}

void sampleAreaLight(thread const Scene &S, thread uint &seed,
                     thread float3 &sampleP, thread float3 &sampleN,
                     thread float3 &sampleEmissive, thread int &sampleMatIdx) {
    int lightIdx = 0;
    if (S.u.lightCount > 1) {
        float pickTarget = rand(seed) * S.u.totalLightArea;
        float cumul = 0.0f;
        for (int i = 0; i < S.u.lightCount; i++) {
            cumul += S.lights[i].normal_area.w;
            if (pickTarget <= cumul) { lightIdx = i; break; }
        }
    }

    GpuLight L = S.lights[lightIdx];
    if (L.kind == 0) {
        float ru = rand(seed);
        float rv = rand(seed);
        sampleP = L.origin.xyz + L.u.xyz * ru + L.v.xyz * rv;
        sampleN = L.normal_area.xyz;
        sampleEmissive = L.emissive.rgb;
        sampleMatIdx = L.matIdx;
    } else {
        float rtri = rand(seed) * L.normal_area.w;
        int lo = L.firstTri;
        int hi = L.firstTri + L.count - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (S.lightTriangles[mid].flatN.w < rtri) lo = mid + 1;
            else hi = mid;
        }
        int triIdx = lo;
        float r1 = rand(seed);
        float r2 = rand(seed);
        if (r1 + r2 > 1.0f) { r1 = 1.0f - r1; r2 = 1.0f - r2; }
        float3 v0 = S.lightTriangles[triIdx].v0.xyz;
        float3 v1 = S.lightTriangles[triIdx].v1.xyz;
        float3 v2 = S.lightTriangles[triIdx].v2.xyz;
        sampleP = v0 + (v1 - v0) * r1 + (v2 - v0) * r2;
        sampleN = S.lightTriangles[triIdx].flatN.xyz;
        sampleEmissive = S.lightTriangles[triIdx].emissive.rgb;
        sampleMatIdx = S.lightTriangles[triIdx].matIdx;
    }
}
)MSL"
@R"MSL(
// ---- RGB path tracer ------------------------------------------------

float3 tracePath(thread const Scene &S, float2 pix, float pr1, float pr2,
                 thread uint &seed) {
    float aspect = float(S.u.width) / float(S.u.height);
    float scale = tan(PI / 180.0f * 0.5f * S.u.fov);
    float x = ((2.0f * (pix.x + 0.5f) / float(S.u.width)) - 1.0f) * scale * aspect;
    float y = -((2.0f * (pix.y + 0.5f) / float(S.u.height)) - 1.0f) * scale;
    float3 rd = normalize(float3(x, y, -1.0f));
    float3 ro = float3(S.u.originX, S.u.originY, S.u.originZ);

    float3 throughput = float3(1.0f);
    float3 radiance = float3(0.0f);
    bool firstBounce = true;

    for (int bounce = 0; bounce < S.u.depth; bounce++) {
        float3 hit, N;
        int matIdx;
        if (!sceneIntersect(S, ro, rd, hit, N, matIdx)) break;

        bool entering = dot(rd, N) < 0.0f;
        if (!entering) N = -N;

        GpuMaterial mat = S.materials[matIdx];

        if (any(mat.emissive.rgb > float3(0.0f))) {
            radiance += throughput * mat.emissive.rgb;
            break;
        }

        if (mat.metallic != 0) {
            rd = reflect(rd, N);
            ro = hit + N * 1e-3f;
            throughput *= mat.albedo.rgb;
            firstBounce = false;
            continue;
        }

        if (mat.transparent != 0) {
            DielectricOut b = dielectricBounce(rd, N, hit, entering, mat.ior, rand(seed));
            rd = b.dir;
            ro = b.origin;
            throughput *= mat.albedo.rgb;
            firstBounce = false;
            continue;
        }

        float3 directLo = float3(0.0f);
        if (S.u.totalLightArea > 0.0f) {
            for (int s = 0; s < S.u.shadowSamples; s++) {
                float3 sampleP, sampleN, sampleEmissive;
                int sampleMatIdx;
                sampleAreaLight(S, seed, sampleP, sampleN, sampleEmissive, sampleMatIdx);

                float3 Li = sampleP - hit;
                float3 wi = normalize(Li);
                float cosTheta = max(0.0f, dot(wi, N));
                float lightDist2 = dot(Li, Li);
                float3 shadowOrigin = (cosTheta <= 0.0f) ? hit - N * 1e-3f : hit + N * 1e-3f;

                float3 sh, sN;
                int sMat;
                bool occluded = false;
                if (sceneIntersect(S, shadowOrigin, wi, sh, sN, sMat)) {
                    float3 d = sh - shadowOrigin;
                    float occluderDist2 = dot(d, d);
                    if (occluderDist2 < lightDist2 - 1e-3f) {
                        GpuMaterial om = S.materials[sMat];
                        if (!any(om.emissive.rgb > float3(0.0f))) occluded = true;
                    }
                }

                if (!occluded) {
                    float cosLight = max(0.0f, dot(sampleN, -wi));
                    float G = (cosTheta * cosLight) / lightDist2;
                    float3 directContrib = (mat.albedo.rgb / PI) * sampleEmissive * G * S.u.totalLightArea;

                    if (S.u.useMIS != 0 && cosLight > 1e-6f) {
                        float pdfLight = lightDist2 / (cosLight * S.u.totalLightArea);
                        float pdfBrdf  = cosTheta / PI;
                        float w = (pdfLight * pdfLight) /
                                  (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                        directContrib *= w;
                    }
                    directLo += directContrib;
                }
            }
            directLo /= float(S.u.shadowSamples);
        }
        radiance += throughput * directLo;

        if (S.u.useRussian != 0 && bounce >= 1) {
            float p = clamp(max(max(mat.albedo.r, mat.albedo.g), mat.albedo.b), 0.05f, 0.95f);
            if (rand(seed) > p) break;
            throughput /= p;
        }

        float3 newDir;
        if (S.u.useStratified != 0 && firstBounce) {
            newDir = sampleHemisphereFrom(N, pr1, pr2);
        } else {
            newDir = sampleHemisphere(N, seed);
        }
        firstBounce = false;

        ro = hit + N * 1e-3f;
        rd = newDir;
        throughput *= mat.albedo.rgb;
    }
    return radiance;
}

// ---- Single-wavelength continuation (used by tracePathSpectral on glass) -

float tracePathSpectralSingle(thread const Scene &S, float3 ro, float3 rd, float lambda,
                              int remainingDepth, int parentBounce, thread uint &seed) {
    float throughput = 1.0f;
    float radiance = 0.0f;
    bool firstBounce = true;

    for (int bounce = 0; bounce < remainingDepth; bounce++) {
        float3 hit, N;
        int matIdx;
        if (!sceneIntersect(S, ro, rd, hit, N, matIdx)) break;

        bool entering = dot(rd, N) < 0.0f;
        if (!entering) N = -N;

        if (any(S.materials[matIdx].emissive.rgb > float3(0.0f))) {
            radiance += throughput * emissiveAt(S, matIdx, lambda);
            break;
        }

        if (S.materials[matIdx].metallic != 0) {
            rd = reflect(rd, N);
            ro = hit + N * 1e-3f;
            throughput *= albedoAt(S, matIdx, lambda);
            firstBounce = false;
            continue;
        }

        if (S.materials[matIdx].transparent != 0) {
            float ior = cauchyIor(S.materials[matIdx].ior, S.materials[matIdx].cauchyB, lambda);
            DielectricOut b = dielectricBounce(rd, N, hit, entering, ior, rand(seed));
            rd = b.dir;
            ro = b.origin;
            throughput *= albedoAt(S, matIdx, lambda);
            firstBounce = false;
            continue;
        }

        float albedoLam = albedoAt(S, matIdx, lambda);

        float directLo = 0.0f;
        if (S.u.totalLightArea > 0.0f) {
            for (int s = 0; s < S.u.shadowSamples; s++) {
                float3 sampleP, sampleN, sampleEmissiveRGB;
                int sampleMatIdx;
                sampleAreaLight(S, seed, sampleP, sampleN, sampleEmissiveRGB, sampleMatIdx);

                float3 Li = sampleP - hit;
                float3 wi = normalize(Li);
                float cosTheta = max(0.0f, dot(wi, N));
                float lightDist2 = dot(Li, Li);
                float3 shadowOrigin = (cosTheta <= 0.0f) ? hit - N * 1e-3f : hit + N * 1e-3f;

                float3 sh, sN;
                int sMat;
                bool occluded = false;
                if (sceneIntersect(S, shadowOrigin, wi, sh, sN, sMat)) {
                    float3 d = sh - shadowOrigin;
                    float occluderDist2 = dot(d, d);
                    if (occluderDist2 < lightDist2 - 1e-3f) {
                        if (!any(S.materials[sMat].emissive.rgb > float3(0.0f)))
                            occluded = true;
                    }
                }

                if (!occluded) {
                    float cosLight = max(0.0f, dot(sampleN, -wi));
                    float G = (cosTheta * cosLight) / lightDist2;
                    float misWeight = 1.0f;
                    if (S.u.useMIS != 0 && cosLight > 1e-6f) {
                        float pdfLight = lightDist2 / (cosLight * S.u.totalLightArea);
                        float pdfBrdf  = cosTheta / PI;
                        misWeight = (pdfLight * pdfLight) /
                                    (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                    }
                    float emitL;
                    if (sampleMatIdx >= 0) {
                        emitL = emissiveAt(S, sampleMatIdx, lambda);
                    } else {
                        emitL = sampleEmissiveRGB.x * 0.30f
                              + sampleEmissiveRGB.y * 0.59f
                              + sampleEmissiveRGB.z * 0.11f;
                    }
                    directLo += (albedoLam / PI) * emitL * G * S.u.totalLightArea * misWeight;
                }
            }
            directLo /= float(S.u.shadowSamples);
        }
        radiance += throughput * directLo;

        if (S.u.useRussian != 0 && (bounce + parentBounce) >= 1) {
            float p = clamp(albedoLam, 0.05f, 0.95f);
            if (rand(seed) > p) break;
            throughput /= p;
        }

        float3 newDir = sampleHemisphere(N, seed);
        firstBounce = false;
        ro = hit + N * 1e-3f;
        rd = newDir;
        throughput *= albedoLam;
    }
    return radiance;
}
)MSL"
@R"MSL(
// ---- Hero-wavelength spectral path tracer ----------------------------

float4 tracePathSpectral(thread const Scene &S, float2 pix, float pr1, float pr2,
                         float4 lambdas, thread uint &seed) {
    float aspect = float(S.u.width) / float(S.u.height);
    float scale = tan(PI / 180.0f * 0.5f * S.u.fov);
    float x = ((2.0f * (pix.x + 0.5f) / float(S.u.width)) - 1.0f) * scale * aspect;
    float y = -((2.0f * (pix.y + 0.5f) / float(S.u.height)) - 1.0f) * scale;
    float3 rd = normalize(float3(x, y, -1.0f));
    float3 ro = float3(S.u.originX, S.u.originY, S.u.originZ);

    float4 throughput = float4(1.0f);
    float4 radiance = float4(0.0f);
    bool firstBounce = true;

    for (int bounce = 0; bounce < S.u.depth; bounce++) {
        float3 hit, N;
        int matIdx;
        if (!sceneIntersect(S, ro, rd, hit, N, matIdx)) break;

        bool entering = dot(rd, N) < 0.0f;
        if (!entering) N = -N;

        if (any(S.materials[matIdx].emissive.rgb > float3(0.0f))) {
            float4 emit = float4(emissiveAt(S, matIdx, lambdas.x),
                                 emissiveAt(S, matIdx, lambdas.y),
                                 emissiveAt(S, matIdx, lambdas.z),
                                 emissiveAt(S, matIdx, lambdas.w));
            radiance += throughput * emit;
            break;
        }

        if (S.materials[matIdx].metallic != 0) {
            rd = reflect(rd, N);
            ro = hit + N * 1e-3f;
            throughput *= float4(albedoAt(S, matIdx, lambdas.x),
                                 albedoAt(S, matIdx, lambdas.y),
                                 albedoAt(S, matIdx, lambdas.z),
                                 albedoAt(S, matIdx, lambdas.w));
            firstBounce = false;
            continue;
        }

        if (S.materials[matIdx].transparent != 0) {
            float baseIor = S.materials[matIdx].ior;
            float cb = S.materials[matIdx].cauchyB;

            if (cb == 0.0f) {
                DielectricOut b = dielectricBounce(rd, N, hit, entering, baseIor, rand(seed));
                rd = b.dir;
                ro = b.origin;
                throughput *= float4(albedoAt(S, matIdx, lambdas.x),
                                     albedoAt(S, matIdx, lambdas.y),
                                     albedoAt(S, matIdx, lambdas.z),
                                     albedoAt(S, matIdx, lambdas.w));
                firstBounce = false;
                continue;
            }

            int remainingDepth = S.u.depth - bounce - 1;
            float4 splitRad = float4(0.0f);
            for (int k = 0; k < 4; k++) {
                float lam = lambdas[k];
                float ior = cauchyIor(baseIor, cb, lam);
                DielectricOut b = dielectricBounce(rd, N, hit, entering, ior, rand(seed));
                float subRad = tracePathSpectralSingle(S, b.origin, b.dir, lam,
                                                       remainingDepth, bounce + 1,
                                                       seed);
                splitRad[k] = subRad * albedoAt(S, matIdx, lam);
            }
            radiance += throughput * splitRad;
            return radiance;
        }

        float4 albedoLam = float4(albedoAt(S, matIdx, lambdas.x),
                                  albedoAt(S, matIdx, lambdas.y),
                                  albedoAt(S, matIdx, lambdas.z),
                                  albedoAt(S, matIdx, lambdas.w));

        float4 directLo = float4(0.0f);
        if (S.u.totalLightArea > 0.0f) {
            for (int s = 0; s < S.u.shadowSamples; s++) {
                float3 sampleP, sampleN, sampleEmissiveRGB;
                int sampleMatIdx;
                sampleAreaLight(S, seed, sampleP, sampleN, sampleEmissiveRGB, sampleMatIdx);

                float3 Li = sampleP - hit;
                float3 wi = normalize(Li);
                float cosTheta = max(0.0f, dot(wi, N));
                float lightDist2 = dot(Li, Li);
                float3 shadowOrigin = (cosTheta <= 0.0f) ? hit - N * 1e-3f : hit + N * 1e-3f;

                float3 sh, sN;
                int sMat;
                bool occluded = false;
                if (sceneIntersect(S, shadowOrigin, wi, sh, sN, sMat)) {
                    float3 d = sh - shadowOrigin;
                    float occluderDist2 = dot(d, d);
                    if (occluderDist2 < lightDist2 - 1e-3f) {
                        if (!any(S.materials[sMat].emissive.rgb > float3(0.0f)))
                            occluded = true;
                    }
                }

                if (!occluded) {
                    float cosLight = max(0.0f, dot(sampleN, -wi));
                    float G = (cosTheta * cosLight) / lightDist2;
                    float misWeight = 1.0f;
                    if (S.u.useMIS != 0 && cosLight > 1e-6f) {
                        float pdfLight = lightDist2 / (cosLight * S.u.totalLightArea);
                        float pdfBrdf  = cosTheta / PI;
                        misWeight = (pdfLight * pdfLight) /
                                    (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                    }
                    float4 emitLam;
                    if (sampleMatIdx >= 0) {
                        emitLam = float4(emissiveAt(S, sampleMatIdx, lambdas.x),
                                         emissiveAt(S, sampleMatIdx, lambdas.y),
                                         emissiveAt(S, sampleMatIdx, lambdas.z),
                                         emissiveAt(S, sampleMatIdx, lambdas.w));
                    } else {
                        float emitL0 = sampleEmissiveRGB.x * 0.30f
                                     + sampleEmissiveRGB.y * 0.59f
                                     + sampleEmissiveRGB.z * 0.11f;
                        emitLam = float4(emitL0);
                    }
                    float4 contrib = (albedoLam / PI) * emitLam * G * S.u.totalLightArea * misWeight;
                    directLo += contrib;
                }
            }
            directLo /= float(S.u.shadowSamples);
        }
        radiance += throughput * directLo;

        if (S.u.useRussian != 0 && bounce >= 1) {
            float p = clamp(albedoLam.x, 0.05f, 0.95f);
            if (rand(seed) > p) break;
            throughput /= p;
        }

        float3 newDir;
        if (S.u.useStratified != 0 && firstBounce) {
            newDir = sampleHemisphereFrom(N, pr1, pr2);
        } else {
            newDir = sampleHemisphere(N, seed);
        }
        firstBounce = false;

        ro = hit + N * 1e-3f;
        rd = newDir;
        throughput *= albedoLam;
    }
    return radiance;
}

// ---- Kernel entry --------------------------------------------------

kernel void path_trace(
    constant Uniforms                          &u            [[buffer(0)]],
    device const GpuSphere                     *spheres      [[buffer(1)]],
    device const GpuPlane                      *planes       [[buffer(2)]],
    device const GpuMaterial                   *materials    [[buffer(3)]],
    device const GpuTriangle                   *triangles    [[buffer(4)]],
    device const GpuBvhNode                    *bvhNodes     [[buffer(5)]],
    device const GpuLight                      *lights       [[buffer(6)]],
    device const GpuLightTriangle              *lightTris    [[buffer(7)]],
    texture2d<float, access::write>             output       [[texture(0)]],
    texture2d<float, access::write>             albedoOut    [[texture(1)]],
    texture2d<float, access::write>             normalOut    [[texture(2)]],
    uint2                                       gid          [[thread_position_in_grid]])
{
    Scene S = { u, spheres, planes, materials, triangles, bvhNodes, lights, lightTris };

    int2 pix = int2(int(gid.x) + u.xOffset, int(gid.y) + u.yOffset);
    if (pix.x >= u.xEnd || pix.x >= u.width ||
        pix.y >= u.yEnd || pix.y >= u.height) return;

    if (u.writeAux != 0) {
        float aspect = float(u.width) / float(u.height);
        float scale = tan(PI / 180.0f * 0.5f * u.fov);
        float ax = ((2.0f * (float(pix.x) + 0.5f) / float(u.width)) - 1.0f) * scale * aspect;
        float ay = -((2.0f * (float(pix.y) + 0.5f) / float(u.height)) - 1.0f) * scale;
        float3 ard = normalize(float3(ax, ay, -1.0f));
        float3 aro = float3(u.originX, u.originY, u.originZ);
        float3 ahit, aN;
        int aMatIdx;
        if (sceneIntersect(S, aro, ard, ahit, aN, aMatIdx)) {
            if (dot(ard, aN) > 0.0f) aN = -aN;
            albedoOut.write(float4(materials[aMatIdx].albedo.rgb, 1.0f), uint2(pix.x, pix.y));
            normalOut.write(float4(aN, 0.0f), uint2(pix.x, pix.y));
        } else {
            albedoOut.write(float4(0.0f), uint2(pix.x, pix.y));
            normalOut.write(float4(0.0f), uint2(pix.x, pix.y));
        }
    }

    uint seed = uint(pix.x) * 1973u + uint(pix.y) * 9277u + uint(u.frameSeed) * 26699u;

    int aaN = max(1, u.aaSamples);
    float3 mean = float3(0.0f);
    float3 M2 = float3(0.0f);
    int taken = 0;
    for (int aa = 0; aa < aaN; aa++) {
        float2 jpix = float2(pix);
        if (aaN > 1) {
            jpix.x += rand(seed) - 0.5f;
            jpix.y += rand(seed) - 0.5f;
        }
        float3 accum = float3(0.0f);
        for (int s = 0; s < u.samples; s++) {
            float r1, r2;
            if (u.useStratified != 0 && u.strata > 0) {
                int sx = s % u.strata;
                int sy = (s / u.strata) % u.strata;
                r1 = (float(sx) + rand(seed)) / float(u.strata);
                r2 = (float(sy) + rand(seed)) / float(u.strata);
            } else {
                r1 = rand(seed);
                r2 = rand(seed);
            }
            if (u.useSpectral != 0) {
                if (u.heroSamples <= 1) {
                    float kSpan = kLambdaMax - kLambdaMin;
                    float lambda = kLambdaMin + rand(seed) * kSpan;
                    float aspect = float(u.width) / float(u.height);
                    float scale  = tan(u.fov * 0.5f);
                    float2 nd = (jpix * 2.0f - float2(u.width, u.height)) /
                                float2(u.width, u.height);
                    float3 dir = normalize(float3(nd.x * scale * aspect,
                                                  -nd.y * scale,
                                                  -1.0f));
                    float3 origin = float3(u.originX, u.originY, u.originZ);
                    float rad = tracePathSpectralSingle(S, origin, dir, lambda,
                                                        u.depth, 0, seed);
                    accum += singleLambdaXYZ(lambda, rad);
                } else {
                    float kSpan = kLambdaMax - kLambdaMin;
                    float kStride = kSpan / 4.0f;
                    float4 lambdas;
                    lambdas.x = kLambdaMin + rand(seed) * kSpan;
                    lambdas.y = lambdas.x + kStride; if (lambdas.y > kLambdaMax) lambdas.y -= kSpan;
                    lambdas.z = lambdas.x + kStride * 2.0f; if (lambdas.z > kLambdaMax) lambdas.z -= kSpan;
                    lambdas.w = lambdas.x + kStride * 3.0f; if (lambdas.w > kLambdaMax) lambdas.w -= kSpan;
                    float4 rad = tracePathSpectral(S, jpix, r1, r2, lambdas, seed);
                    float3 xyz = singleLambdaXYZ(lambdas.x, rad.x)
                               + singleLambdaXYZ(lambdas.y, rad.y)
                               + singleLambdaXYZ(lambdas.z, rad.z)
                               + singleLambdaXYZ(lambdas.w, rad.w);
                    accum += xyz * 0.25f;
                }
            } else {
                accum += tracePath(S, jpix, r1, r2, seed);
            }
        }
        accum /= float(u.samples);

        taken += 1;
        float3 delta = accum - mean;
        mean += delta / float(taken);
        float3 delta2 = accum - mean;
        M2 += delta * delta2;

        if (u.useAdaptive != 0 && taken >= 4) {
            float3 variance = M2 / float(taken - 1);
            float3 rel = variance / (mean * mean + float3(0.01f));
            if (max(max(rel.r, rel.g), rel.b) < 0.05f) break;
        }
    }
    if (u.useSpectral != 0)
        mean = xyzToLinearSRGB(mean);
    output.write(float4(mean, 1.0f), uint2(pix.x, pix.y));
}

// ---- Multi-pass kernel ---------------------------------------------
//
// Used by the saturation-friendly dispatch path (when useAdaptive == 0).
// Each invocation of this kernel covers the full image (so the GPU runs
// at full saturation) but only computes one AA index's contribution for
// the sampleStart..sampleStart+sampleCount slice. The CPU dispatches
// this many times - one per (aaIdx, sample-batch) pair - to bound each
// command buffer's wallclock under Apple's compute watchdog.
//
// Output texture acts as an HDR sum accumulator, NOT a mean: each pass
// adds its sum-of-contributions to whatever's already there, and the
// CPU divides by total contribution count (aaSamples * samples) at
// readback time. Spectral mode keeps the accumulator in CIE XYZ; the
// XYZ -> linear sRGB conversion also moves to the CPU. Both transforms
// are linear so they commute with averaging (no precision loss).
//
// AA jitter is computed deterministically from (pix, frameSeed, aaIdx)
// so the same aaIdx produces the same sub-pixel offset across every
// sample-batch pass for that AA. Sample seeds further include
// sampleStart so different batches use different RNG sequences. The
// output is statistically equivalent to (but not bit-identical with)
// the legacy single-pass kernel for the same configuration.
//
// On passIdx == 0 the kernel writes its sum directly (clobbering the
// texture's prior contents). On later passes it reads + accumulates.
// This means the texture doesn't need to be zero-cleared before the
// run as long as pass 0 covers every pixel - which it does, since
// each pass covers the full image.

kernel void path_trace_pass(
    constant Uniforms                          &u            [[buffer(0)]],
    device const GpuSphere                     *spheres      [[buffer(1)]],
    device const GpuPlane                      *planes       [[buffer(2)]],
    device const GpuMaterial                   *materials    [[buffer(3)]],
    device const GpuTriangle                   *triangles    [[buffer(4)]],
    device const GpuBvhNode                    *bvhNodes     [[buffer(5)]],
    device const GpuLight                      *lights       [[buffer(6)]],
    device const GpuLightTriangle              *lightTris    [[buffer(7)]],
    texture2d<float, access::read_write>        output       [[texture(0)]],
    texture2d<float, access::write>             albedoOut    [[texture(1)]],
    texture2d<float, access::write>             normalOut    [[texture(2)]],
    uint2                                       gid          [[thread_position_in_grid]])
{
    Scene S = { u, spheres, planes, materials, triangles, bvhNodes, lights, lightTris };

    int2 pix = int2(int(gid.x) + u.xOffset, int(gid.y) + u.yOffset);
    if (pix.x >= u.xEnd || pix.x >= u.width ||
        pix.y >= u.yEnd || pix.y >= u.height) return;

    // OIDN aux capture: only when uniforms ask for it (caller sets
    // writeAux = 1 only on passIdx == 0 + aaIdx == 0).
    if (u.writeAux != 0) {
        float aspect = float(u.width) / float(u.height);
        float scale = tan(PI / 180.0f * 0.5f * u.fov);
        float ax = ((2.0f * (float(pix.x) + 0.5f) / float(u.width)) - 1.0f) * scale * aspect;
        float ay = -((2.0f * (float(pix.y) + 0.5f) / float(u.height)) - 1.0f) * scale;
        float3 ard = normalize(float3(ax, ay, -1.0f));
        float3 aro = float3(u.originX, u.originY, u.originZ);
        float3 ahit, aN;
        int aMatIdx;
        if (sceneIntersect(S, aro, ard, ahit, aN, aMatIdx)) {
            if (dot(ard, aN) > 0.0f) aN = -aN;
            albedoOut.write(float4(materials[aMatIdx].albedo.rgb, 1.0f), uint2(pix.x, pix.y));
            normalOut.write(float4(aN, 0.0f), uint2(pix.x, pix.y));
        } else {
            albedoOut.write(float4(0.0f), uint2(pix.x, pix.y));
            normalOut.write(float4(0.0f), uint2(pix.x, pix.y));
        }
    }

    int aaN = max(1, u.aaSamples);

    // Jitter seed: depends on aaIdx but NOT on sampleStart, so every
    // sample-batch within an aaIdx uses the same sub-pixel offset.
    uint jitterSeed = uint(pix.x) * 1973u + uint(pix.y) * 9277u
                    + uint(u.frameSeed) * 26699u
                    + uint(u.aaIdx) * 16127u;
    float2 jpix = float2(pix);
    if (aaN > 1) {
        jpix.x += rand(jitterSeed) - 0.5f;
        jpix.y += rand(jitterSeed) - 0.5f;
    }

    // Sample seed: includes sampleStart so different batches of the
    // same aaIdx use different RNG sequences. The 0x9e3779b9 mix-in
    // is the golden-ratio hash constant - any reasonably-decorrelated
    // value works.
    uint seed = jitterSeed
              + uint(u.sampleStart) * 7919u
              + 0x9e3779b9u;

    int sampleEnd = u.sampleStart + u.sampleCount;
    float3 accum = float3(0.0f);

    for (int s = u.sampleStart; s < sampleEnd; s++) {
        float r1, r2;
        if (u.useStratified != 0 && u.strata > 0) {
            int sx = s % u.strata;
            int sy = (s / u.strata) % u.strata;
            r1 = (float(sx) + rand(seed)) / float(u.strata);
            r2 = (float(sy) + rand(seed)) / float(u.strata);
        } else {
            r1 = rand(seed);
            r2 = rand(seed);
        }
        if (u.useSpectral != 0) {
            if (u.heroSamples <= 1) {
                float kSpan = kLambdaMax - kLambdaMin;
                float lambda = kLambdaMin + rand(seed) * kSpan;
                float aspect = float(u.width) / float(u.height);
                float scale  = tan(u.fov * 0.5f);
                float2 nd = (jpix * 2.0f - float2(u.width, u.height)) /
                            float2(u.width, u.height);
                float3 dir = normalize(float3(nd.x * scale * aspect,
                                              -nd.y * scale,
                                              -1.0f));
                float3 origin = float3(u.originX, u.originY, u.originZ);
                float rad = tracePathSpectralSingle(S, origin, dir, lambda,
                                                    u.depth, 0, seed);
                accum += singleLambdaXYZ(lambda, rad);
            } else {
                float kSpan = kLambdaMax - kLambdaMin;
                float kStride = kSpan / 4.0f;
                float4 lambdas;
                lambdas.x = kLambdaMin + rand(seed) * kSpan;
                lambdas.y = lambdas.x + kStride; if (lambdas.y > kLambdaMax) lambdas.y -= kSpan;
                lambdas.z = lambdas.x + kStride * 2.0f; if (lambdas.z > kLambdaMax) lambdas.z -= kSpan;
                lambdas.w = lambdas.x + kStride * 3.0f; if (lambdas.w > kLambdaMax) lambdas.w -= kSpan;
                float4 rad = tracePathSpectral(S, jpix, r1, r2, lambdas, seed);
                float3 xyz = singleLambdaXYZ(lambdas.x, rad.x)
                           + singleLambdaXYZ(lambdas.y, rad.y)
                           + singleLambdaXYZ(lambdas.z, rad.z)
                           + singleLambdaXYZ(lambdas.w, rad.w);
                accum += xyz * 0.25f;
            }
        } else {
            accum += tracePath(S, jpix, r1, r2, seed);
        }
    }

    if (u.passIdx == 0) {
        // First pass clobbers the texture's prior contents.
        output.write(float4(accum, 0.0f), uint2(pix.x, pix.y));
    } else {
        float4 prev = output.read(uint2(pix.x, pix.y));
        output.write(prev + float4(accum, 0.0f), uint2(pix.x, pix.y));
    }
}
)MSL";

// ---------- Host-side POD layouts mirroring the MSL structs ---------------
//
// These structs are byte-for-byte-equivalent to the MSL definitions
// above. We copy them straight into MTLBuffers via newBufferWithBytes,
// and the GPU reinterprets the bytes through the MSL struct layout.
//
// Same layout the OpenGL backend uses (std430 packing). Static-asserts
// guard against drift; if either side changes, both static_asserts
// must be updated together.

namespace
{
    struct GpuSphere
    {
        float center[4];
        int   matIdx;
        int   _pad[3];
    };
    static_assert(sizeof(GpuSphere) == 32, "GpuSphere host layout must match MSL");

    struct GpuPlane
    {
        float origin[4];
        float u[4];
        float v[4];
        float normal_area[4];
        int   matIdx;
        int   _pad[3];
    };
    static_assert(sizeof(GpuPlane) == 80, "GpuPlane host layout must match MSL");

    struct GpuMaterial
    {
        float albedo[4];
        float emissive[4];
        int   metallic;
        int   transparent;
        float ior;
        float cauchyB;
        float albedoSpectrum[64];
        float emissiveSpectrum[64];
    };
    static_assert(sizeof(GpuMaterial) == 560,
                  "GpuMaterial size must be 560 bytes (matches GLSL std430 + MSL)");
    static_assert(sizeof(GpuMaterial) % 16 == 0,
                  "GpuMaterial must be a multiple of 16 bytes for array stride");

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
    static_assert(sizeof(GpuTriangle) == 128, "GpuTriangle host layout must match MSL");

    struct GpuBvhNode
    {
        float boxMin[4];
        float boxMax[4];
        int   leftOrFirst;
        int   rightChild;
        int   count;
        int   _pad;
    };
    static_assert(sizeof(GpuBvhNode) == 48, "GpuBvhNode host layout must match MSL");

    struct GpuLight
    {
        float origin[4];
        float u[4];
        float v[4];
        float normal_area[4];
        float emissive[4];
        int   kind;
        int   firstTri;
        int   count;
        int   matIdx;
    };
    static_assert(sizeof(GpuLight) == 96, "GpuLight host layout must match MSL");

    struct GpuLightTriangle
    {
        float v0[4];
        float v1[4];
        float v2[4];
        float flatN[4];
        float emissive[4];
        int   matIdx;
        int   _pad[3];
    };
    static_assert(sizeof(GpuLightTriangle) == 96,
                  "GpuLightTriangle host layout must match MSL");

    struct Uniforms
    {
        int   width;
        int   height;
        float fov;
        float originX;
        float originY;
        float originZ;
        int   depth;
        int   samples;
        int   shadowSamples;
        int   sphereCount;
        int   planeCount;
        int   triangleCount;
        int   bvhNodeCount;
        int   lightCount;
        float totalLightArea;
        int   xOffset;
        int   xEnd;
        int   yOffset;
        int   yEnd;
        int   frameSeed;
        int   useMIS;
        int   useRussian;
        int   useStratified;
        int   strata;
        int   aaSamples;
        int   useAdaptive;
        int   writeAux;
        int   useSpectral;
        int   heroSamples;
        // Multi-pass kernel uses these; legacy path_trace ignores them.
        int   passIdx;
        int   aaIdx;
        int   sampleStart;
        int   sampleCount;
        int   _pad0, _pad1, _pad2;
    };
    static_assert(sizeof(Uniforms) == 144,
                  "Uniforms must be 144 bytes (multiple of 16) so per-pass "
                  "buffer offsets are 16-aligned for MSL vectorized loads");

    // Mirrors the OpenGL backend's compressZone helper (Windows long
    // zone names like "Eastern Daylight Time" -> "EDT"). On macOS the
    // POSIX abbreviations come back already short, so the map is
    // mostly a no-op here, but we keep the same code path to produce
    // identical filename slugs across all three OSes.
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
        if (utc) gmtime_r(&now, &tm);
        else     localtime_r(&now, &tm);
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

// ---------- pImpl: Obj-C state owned by MetalRenderer ---------------------

struct MetalRenderer::Impl
{
    id<MTLDevice>               device         = nil;
    id<MTLCommandQueue>         queue          = nil;
    // Two compute pipelines from the same MSL library:
    //   pipeline      = legacy single-pass kernel (path_trace), used by
    //                   the strip dispatch path. Honors useAdaptive's
    //                   per-pixel Welford early-exit.
    //   pipelinePass  = multi-pass accumulator kernel (path_trace_pass),
    //                   used by the saturation-friendly dispatch path
    //                   when useAdaptive is off. Each invocation covers
    //                   the full image and contributes one (aaIdx,
    //                   sample-batch) slice to the running sum.
    id<MTLComputePipelineState> pipeline       = nil;
    id<MTLComputePipelineState> pipelinePass   = nil;

    // Output + OIDN aux. RGBA32Float so the readback path keeps full HDR
    // (matches OpenGL backend's RGBA16F, but Metal's RGBA16Float
    // texture readback to host CPU bytes is more awkward than RGBA32F;
    // this trades a bit of GPU memory for a clean memcpy on Apple
    // Silicon's unified memory).
    id<MTLTexture>              outputTex      = nil;
    id<MTLTexture>              albedoTex      = nil;
    id<MTLTexture>              normalTex      = nil;

    id<MTLBuffer>               sphereBuf      = nil;
    id<MTLBuffer>               planeBuf       = nil;
    id<MTLBuffer>               materialBuf    = nil;
    id<MTLBuffer>               triangleBuf    = nil;
    id<MTLBuffer>               bvhBuf         = nil;
    id<MTLBuffer>               lightBuf       = nil;
    id<MTLBuffer>               lightTriBuf    = nil;
    // Uniforms aren't kept here: render() allocates a fresh per-render
    // MTLBuffer with one Uniforms entry per strip, since per-strip data
    // has to be addressable independently while pipelined command buffers
    // are still in flight on the queue.

    bool                        initialized    = false;
};

// ---------- Helpers (file-local, namespaced to avoid mac-only build noise)

namespace
{
    // Allocate a non-empty MTLBuffer of `bytes`, copying `data` if non-null.
    // Empty inputs get a 16-byte dummy so the kernel can still be encoded
    // with valid bindings (shader gates access via the count uniforms).
    id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device,
                                   const void *data, size_t bytes,
                                   size_t dummyBytes = 16)
    {
        if (bytes == 0)
        {
            std::vector<unsigned char> dummy(dummyBytes, 0);
            return [device newBufferWithBytes:dummy.data()
                                       length:dummy.size()
                                      options:MTLResourceStorageModeShared];
        }
        return [device newBufferWithBytes:data
                                   length:bytes
                                  options:MTLResourceStorageModeShared];
    }

    bool initMetal(MetalRenderer::Impl &im, int width, int height)
    {
        if (im.initialized) return true;

        im.device = MTLCreateSystemDefaultDevice();
        if (!im.device)
        {
            std::cerr << "MetalRenderer: no Metal device available" << std::endl;
            return false;
        }
        im.queue = [im.device newCommandQueue];

        NSError *err = nil;
        id<MTLLibrary> lib = [im.device newLibraryWithSource:kMSLSource
                                                     options:nil
                                                       error:&err];
        if (!lib)
        {
            std::cerr << "MetalRenderer: MSL compile failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "(no error info)")
                      << std::endl;
            return false;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"path_trace"];
        if (!fn)
        {
            std::cerr << "MetalRenderer: MSL kernel 'path_trace' not found"
                      << std::endl;
            return false;
        }
        im.pipeline = [im.device newComputePipelineStateWithFunction:fn error:&err];
        if (!im.pipeline)
        {
            std::cerr << "MetalRenderer: compute pipeline build failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "(no error info)")
                      << std::endl;
            return false;
        }

        id<MTLFunction> fnPass = [lib newFunctionWithName:@"path_trace_pass"];
        if (!fnPass)
        {
            std::cerr << "MetalRenderer: MSL kernel 'path_trace_pass' not found"
                      << std::endl;
            return false;
        }
        im.pipelinePass = [im.device newComputePipelineStateWithFunction:fnPass error:&err];
        if (!im.pipelinePass)
        {
            std::cerr << "MetalRenderer: multi-pass compute pipeline build failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "(no error info)")
                      << std::endl;
            return false;
        }

        // Return-by-value here, not a reference-out parameter: ARC
        // reference parameters default to __autoreleasing ownership,
        // which clashes with the __strong id<MTLTexture> struct fields.
        // Returning by value sidesteps the ownership-annotation
        // gymnastics; ARC inserts the right retain/release on assignment.
        auto makeFloatTex = [&]() -> id<MTLTexture> {
            MTLTextureDescriptor *desc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                             width:(NSUInteger)width
                                            height:(NSUInteger)height
                                         mipmapped:NO];
            desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModeShared;
            return [im.device newTextureWithDescriptor:desc];
        };
        im.outputTex = makeFloatTex();
        im.albedoTex = makeFloatTex();
        im.normalTex = makeFloatTex();

        im.initialized = true;
        return true;
    }

    // Build the seven scene buffers + return the total area-light area
    // (used by the kernel for light-pick PDFs). Mirrors OpenglRenderer's
    // uploadScene line-for-line.
    void uploadScene(MetalRenderer::Impl &im,
                     const Scenes::SceneData &scene,
                     float &outTotalLightArea)
    {
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

        // Plane lights front-loaded so coplanar ties favor the light
        // (matches OpenGL backend).
        for (const auto &L : scene.areaLights)
        {
            if (L.kind != Scenes::AreaLightKind::Plane) continue;
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
                gl.matIdx = -1;

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

        im.sphereBuf   = makeSharedBuffer(im.device, gpuSpheres.data(),
                                          gpuSpheres.size() * sizeof(GpuSphere),
                                          sizeof(GpuSphere));
        im.planeBuf    = makeSharedBuffer(im.device, gpuPlanes.data(),
                                          gpuPlanes.size() * sizeof(GpuPlane),
                                          sizeof(GpuPlane));
        im.materialBuf = makeSharedBuffer(im.device, mats.data(),
                                          mats.size() * sizeof(GpuMaterial),
                                          sizeof(GpuMaterial));
        im.triangleBuf = makeSharedBuffer(im.device, gpuTris.data(),
                                          gpuTris.size() * sizeof(GpuTriangle),
                                          sizeof(GpuTriangle));
        im.bvhBuf      = makeSharedBuffer(im.device, gpuBvh.data(),
                                          gpuBvh.size() * sizeof(GpuBvhNode),
                                          sizeof(GpuBvhNode));
        im.lightBuf    = makeSharedBuffer(im.device, gpuLights.data(),
                                          gpuLights.size() * sizeof(GpuLight),
                                          sizeof(GpuLight));
        im.lightTriBuf = makeSharedBuffer(im.device, gpuLightTris.data(),
                                          gpuLightTris.size() * sizeof(GpuLightTriangle),
                                          sizeof(GpuLightTriangle));
    }
}

// ---------- MetalRenderer -------------------------------------------------

MetalRenderer::MetalRenderer(int width, int height,
                             int depth, int samples, int shadowSamples,
                             GLFWwindow * /*sharedContext*/)
    : _width{width}, _height{height},
      _maxDepth{depth}, _samples{samples}, _shadowSamples{shadowSamples}
{
    _impl = new Impl();
}

MetalRenderer::~MetalRenderer()
{
    // ARC releases id<...> members when Impl is destroyed.
    delete _impl;
    _impl = nullptr;
}

void MetalRenderer::render(const Scenes::SceneData &scene,
                           std::chrono::steady_clock::time_point start,
                           const std::string &outputDir)
{
    lastOutputPath.clear();
    @autoreleasepool {

    if (scene.areaLights.empty())
    {
        std::cerr << "physically-cringe-rendering: scene has no area lights"
                  << std::endl;
        return;
    }

    if (!initMetal(*_impl, _width, _height)) return;

    float totalLightArea = 0.f;
    uploadScene(*_impl, scene, totalLightArea);

    // Plane count = walls + plane-kind area lights (matches OpenGL).
    int planeLightCount = 0;
    for (const auto &L : scene.areaLights)
        if (L.kind == Scenes::AreaLightKind::Plane) planeLightCount++;

    // Two dispatch paths, selected at runtime by useAdaptive:
    //
    //   useAdaptive ON  -> strip path (legacy path_trace kernel).
    //                      Each strip is a small spatial region; the
    //                      kernel keeps Welford state per pixel and
    //                      can early-exit converged AA samples. Doesn't
    //                      saturate the GPU on heavy renders (small
    //                      strips => few threadgroups => idle cores)
    //                      but adaptive sampling is preserved.
    //
    //   useAdaptive OFF -> multi-pass path (path_trace_pass kernel).
    //                      Each dispatch covers the FULL image (at full
    //                      GPU saturation: 1080^2 / (32*32) = 1156
    //                      threadgroups vs. 48 cores) but only computes
    //                      one (aaIdx, sample-batch) slice per pass.
    //                      The output texture acts as a running-sum
    //                      accumulator; CPU divides by total contribution
    //                      count at readback. ~2x faster than strip path
    //                      on Picture-class workloads.
    //
    // Post-dispatch (wait + audit + readback + tone-map + PNG) is shared.
    // Multi-pass normalization (divide accumulator by total contributions
    // and convert XYZ -> sRGB if spectral) happens after readback below.

    // Common uniform fields (everything except per-pass / per-strip
    // bounds). Both paths fill in their own y / pass / sample uniforms.
    Uniforms uBase{};
    uBase.width            = _width;
    uBase.height           = _height;
    uBase.fov              = scene.camera.fov;
    uBase.originX          = scene.camera.position[0];
    uBase.originY          = scene.camera.position[1];
    uBase.originZ          = scene.camera.position[2];
    uBase.depth            = _maxDepth;
    uBase.samples          = _samples;
    uBase.shadowSamples    = _shadowSamples;
    uBase.sphereCount      = (int)scene.spheres.size();
    uBase.planeCount       = (int)scene.walls.size() + planeLightCount;
    uBase.triangleCount    = (int)scene.triangles.size();
    uBase.bvhNodeCount     = (int)scene.triangleBvh.size();
    uBase.lightCount       = (int)scene.areaLights.size();
    uBase.totalLightArea   = totalLightArea;
    uBase.frameSeed        = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count() & 0x7FFFFFFF);
    uBase.useMIS           = useMIS ? 1 : 0;
    uBase.useRussian       = useRussian ? 1 : 0;
    uBase.useStratified    = useStratified ? 1 : 0;
    uBase.strata           = useStratified
                             ? std::max(1, (int)std::round(std::sqrt((float)_samples)))
                             : 0;
    uBase.aaSamples        = std::max(1, aaSamples);
    uBase.useAdaptive      = useAdaptive ? 1 : 0;
    uBase.writeAux         = useOIDN ? 1 : 0;
    uBase.useSpectral      = useSpectral ? 1 : 0;
    uBase.heroSamples      = std::clamp(heroSamples, 1, 4);
    uBase.xOffset          = 0;
    uBase.xEnd             = _width;

    // Atomic counter bumped from each command buffer's completion
    // handler. The handlers fire on a Metal-internal thread; the
    // counter lives on this stack frame and stays alive until the
    // final waitUntilCompleted below returns.
    std::atomic<int> doneCmdBuffers{0};
    std::atomic<int> *progressPtr = progressRows;
    int height = _height;

    NSMutableArray<id<MTLCommandBuffer>> *pending =
        [[NSMutableArray alloc] init];
    bool multiPassUsed = false;
    int totalContributions = 0;

    if (useAdaptive)
    {
        // ---- Strip path ------------------------------------------
        //
        // Adaptive sampling keeps Welford state across AA samples
        // inside a single kernel invocation, which is incompatible
        // with the multi-pass accumulator pattern (per-pass kernel
        // doesn't see the running variance from other passes). Fall
        // back to the strip dispatch: small spatial tiles, each one
        // running the legacy path_trace kernel with the full AA +
        // adaptive loop intact. Doesn't saturate the GPU on heavy
        // workloads but produces correct adaptive output.
        //
        // Strip-height heuristic: ~3 sec per strip on the calibrated
        // M1 Ultra throughput (~1.1B ops/sec for this kernel),
        // floored at 32 strips for progress UX, ceilinged at 1 row
        // per strip for the pathological heavy case.
        int aaMult = std::max(1, aaSamples);
        long long workPerPixel =
            (long long)_samples * _maxDepth * _shadowSamples * aaMult;
        if (useAdaptive) workPerPixel = (long long)((double)workPerPixel * 1.15);
        if (useSpectral) workPerPixel = (long long)((double)workPerPixel * 2.5);

        constexpr long long kTargetWorkPerStrip = 3'300'000'000LL;
        long long pixelsPerStrip = std::max(
            (long long)_width,
            kTargetWorkPerStrip / std::max(1LL, workPerPixel));
        int stripHeight = std::clamp(
            (int)(pixelsPerStrip / std::max(1, _width)),
            1, _height);

        constexpr int kMinStrips = 32;
        int provisionalNumStrips = (_height + stripHeight - 1) / stripHeight;
        if (provisionalNumStrips < kMinStrips && _height >= kMinStrips)
        {
            stripHeight = std::max(1, _height / kMinStrips);
        }
        int numStrips = (_height + stripHeight - 1) / stripHeight;

        int tgX = 32, tgY = 32;
        if (_impl->pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(tgX * tgY))
        {
            tgX = 16;
            tgY = 16;
        }
        MTLSize threadsPerGroup = MTLSizeMake(tgX, tgY, 1);

        size_t uniformsStride = sizeof(Uniforms);
        id<MTLBuffer> uniformsBuf =
            [_impl->device newBufferWithLength:uniformsStride * (size_t)numStrips
                                       options:MTLResourceStorageModeShared];
        Uniforms *uniformsPtr = (Uniforms *)[uniformsBuf contents];
        for (int i = 0; i < numStrips; i++)
        {
            int yStart = i * stripHeight;
            int yEnd = std::min(yStart + stripHeight, _height);
            Uniforms u = uBase;
            u.yOffset = yStart;
            u.yEnd    = yEnd;
            uniformsPtr[i] = u;
        }

        for (int i = 0; i < numStrips; i++)
        {
            if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
                break;

            int yStart = i * stripHeight;
            int yEnd = std::min(yStart + stripHeight, _height);
            int stripH = yEnd - yStart;

            id<MTLCommandBuffer> cmdbuf = [_impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:_impl->pipeline];
            [enc setBuffer:uniformsBuf      offset:(NSUInteger)(i * uniformsStride) atIndex:0];
            [enc setBuffer:_impl->sphereBuf   offset:0 atIndex:1];
            [enc setBuffer:_impl->planeBuf    offset:0 atIndex:2];
            [enc setBuffer:_impl->materialBuf offset:0 atIndex:3];
            [enc setBuffer:_impl->triangleBuf offset:0 atIndex:4];
            [enc setBuffer:_impl->bvhBuf      offset:0 atIndex:5];
            [enc setBuffer:_impl->lightBuf    offset:0 atIndex:6];
            [enc setBuffer:_impl->lightTriBuf offset:0 atIndex:7];
            [enc setTexture:_impl->outputTex atIndex:0];
            [enc setTexture:_impl->albedoTex atIndex:1];
            [enc setTexture:_impl->normalTex atIndex:2];

            MTLSize threadgroups = MTLSizeMake((NSUInteger)((_width + tgX - 1) / tgX),
                                               (NSUInteger)((stripH + tgY - 1) / tgY),
                                               1);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
            [enc endEncoding];

            std::atomic<int> *donePtr = &doneCmdBuffers;
            int totalStrips = numStrips;
            [cmdbuf addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
                int done = donePtr->fetch_add(1, std::memory_order_relaxed) + 1;
                if (progressPtr)
                {
                    int approxRows = (int)((double)done / totalStrips * height);
                    progressPtr->store(approxRows, std::memory_order_relaxed);
                }
            }];

            [cmdbuf commit];
            [pending addObject:cmdbuf];
        }
    }
    else
    {
        // ---- Multi-pass path -------------------------------------
        //
        // Each command buffer dispatches the FULL image with the
        // path_trace_pass kernel, which accumulates one (aaIdx,
        // sample-batch) slice into the output texture as a running
        // sum. Total work per pixel = aaSamples * samples * shadow *
        // depth ops; we split that across enough passes to keep each
        // dispatch under the watchdog budget.
        //
        // Per-pass per-pixel budget: ~2900 ops on the calibrated
        // M1 Ultra throughput (~1.1B ops/sec) gives ~3 sec per
        // dispatch on a 1080^2 image. Picture preset ends up at
        // ~548 passes; Production at ~12 passes; Quick at 4.
        multiPassUsed = true;

        long long opsPerPixelPerSample = (long long)_shadowSamples * _maxDepth;
        if (useSpectral) opsPerPixelPerSample =
            (long long)((double)opsPerPixelPerSample * 2.5);

        constexpr long long kTargetWorkPerPixelPerPass = 2900;
        int samplesPerPass = std::max(1,
            (int)(kTargetWorkPerPixelPerPass / std::max(1LL, opsPerPixelPerSample)));
        samplesPerPass = std::min(samplesPerPass, _samples);

        int passesPerAa = (_samples + samplesPerPass - 1) / samplesPerPass;
        int aaN = std::max(1, aaSamples);
        int totalPasses = aaN * passesPerAa;
        totalContributions = aaN * _samples;

        int tgX = 32, tgY = 32;
        if (_impl->pipelinePass.maxTotalThreadsPerThreadgroup < (NSUInteger)(tgX * tgY))
        {
            tgX = 16;
            tgY = 16;
        }
        MTLSize threadsPerGroup = MTLSizeMake(tgX, tgY, 1);

        size_t uniformsStride = sizeof(Uniforms);
        id<MTLBuffer> uniformsBuf =
            [_impl->device newBufferWithLength:uniformsStride * (size_t)totalPasses
                                       options:MTLResourceStorageModeShared];
        Uniforms *uniformsPtr = (Uniforms *)[uniformsBuf contents];
        for (int p = 0; p < totalPasses; p++)
        {
            int aaIdx       = p / passesPerAa;
            int batchIdx    = p % passesPerAa;
            int sampleStart = batchIdx * samplesPerPass;
            int sampleEnd   = std::min(sampleStart + samplesPerPass, _samples);

            Uniforms u = uBase;
            u.passIdx     = p;
            u.aaIdx       = aaIdx;
            u.sampleStart = sampleStart;
            u.sampleCount = sampleEnd - sampleStart;
            u.yOffset     = 0;
            u.yEnd        = _height;
            // OIDN aux capture happens once, on the first pass only;
            // every other pass writeAux=0 so it doesn't repeatedly
            // overwrite the same aux pixels with identical data.
            u.writeAux    = (useOIDN && p == 0) ? 1 : 0;
            uniformsPtr[p] = u;
        }

        MTLSize threadgroups = MTLSizeMake((NSUInteger)((_width + tgX - 1) / tgX),
                                           (NSUInteger)((_height + tgY - 1) / tgY),
                                           1);
        for (int p = 0; p < totalPasses; p++)
        {
            if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
                break;

            id<MTLCommandBuffer> cmdbuf = [_impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:_impl->pipelinePass];
            [enc setBuffer:uniformsBuf      offset:(NSUInteger)(p * uniformsStride) atIndex:0];
            [enc setBuffer:_impl->sphereBuf   offset:0 atIndex:1];
            [enc setBuffer:_impl->planeBuf    offset:0 atIndex:2];
            [enc setBuffer:_impl->materialBuf offset:0 atIndex:3];
            [enc setBuffer:_impl->triangleBuf offset:0 atIndex:4];
            [enc setBuffer:_impl->bvhBuf      offset:0 atIndex:5];
            [enc setBuffer:_impl->lightBuf    offset:0 atIndex:6];
            [enc setBuffer:_impl->lightTriBuf offset:0 atIndex:7];
            [enc setTexture:_impl->outputTex atIndex:0];
            [enc setTexture:_impl->albedoTex atIndex:1];
            [enc setTexture:_impl->normalTex atIndex:2];

            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
            [enc endEncoding];

            std::atomic<int> *donePtr = &doneCmdBuffers;
            int totalP = totalPasses;
            [cmdbuf addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
                int done = donePtr->fetch_add(1, std::memory_order_relaxed) + 1;
                if (progressPtr)
                {
                    int approxRows = (int)((double)done / totalP * height);
                    progressPtr->store(approxRows, std::memory_order_relaxed);
                }
            }];

            [cmdbuf commit];
            [pending addObject:cmdbuf];
        }
    }

    // One wait, at the end. All command buffers run pipelined on the
    // GPU; the CPU only blocks here for whatever's still in flight
    // after the encoding loop completes.
    if ([pending count] > 0)
        [[pending lastObject] waitUntilCompleted];

    // Audit cmd buffer outcomes. Apple's compute watchdog (the
    // looser-than-Windows-TDR equivalent) shows up here as
    // status=MTLCommandBufferStatusError; the kernel ran out of
    // wallclock and was killed before its writes landed. Visible
    // in the output as horizontal black bands matching the killed
    // strips. Surface those failures explicitly rather than letting
    // them masquerade as "the kernel ran fine, the image just looks
    // weird."
    int erroredCmdBuffers = 0;
    for (id<MTLCommandBuffer> cb in pending)
    {
        if (cb.status == MTLCommandBufferStatusError)
        {
            erroredCmdBuffers++;
            if (cb.error)
            {
                std::cerr << "Metal cmd buffer error: "
                          << [[cb.error localizedDescription] UTF8String]
                          << std::endl;
            }
        }
    }
    if (erroredCmdBuffers > 0)
    {
        const char *kind = multiPassUsed ? "passes" : "strips";
        std::cerr << "Metal render: " << erroredCmdBuffers << " of "
                  << [pending count] << " " << kind << " failed (likely "
                  << "GPU watchdog timeout). Reduce samples / depth / "
                  << "shadow / AA, or render at a smaller resolution."
                  << std::endl;
    }

    if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
        std::cout << "Metal render cancelled." << std::endl;

    // Readback. RGBA32Float textures: getBytes is a memcpy on Apple
    // Silicon's unified memory. Strip into Vec3f for downstream
    // (matches OpenGL backend's readbackToVec3 helper).
    auto readbackToVec3 = [&](id<MTLTexture> tex) {
        std::vector<float> tmp((size_t)_width * _height * 4);
        [tex getBytes:tmp.data()
                  bytesPerRow:(NSUInteger)(_width * 4 * sizeof(float))
                   fromRegion:MTLRegionMake2D(0, 0, (NSUInteger)_width, (NSUInteger)_height)
                  mipmapLevel:0];
        std::vector<Vec3f> out((size_t)_width * _height);
        for (size_t i = 0; i < out.size(); i++)
            out[i] = Vec3f(tmp[i * 4 + 0], tmp[i * 4 + 1], tmp[i * 4 + 2]);
        return out;
    };
    std::vector<Vec3f> hdr = readbackToVec3(_impl->outputTex);
    std::vector<Vec3f> albedoBuf, normalBuf;
    if (useOIDN)
    {
        albedoBuf = readbackToVec3(_impl->albedoTex);
        normalBuf = readbackToVec3(_impl->normalTex);
    }

    // Multi-pass normalization. The kernel accumulates a sum of per-
    // sample contributions in the output texture; CPU divides by the
    // total contribution count (aaSamples * samples) to get the mean.
    // Spectral mode also moves XYZ -> linear sRGB to the CPU here so
    // the kernel can stay in the more numerically clean XYZ space
    // across passes (the conversion is linear so this is mathematically
    // identical to per-pass conversion).
    if (multiPassUsed && totalContributions > 0)
    {
        float invTotal = 1.0f / (float)totalContributions;
        for (auto &c : hdr) c *= invTotal;
        if (useSpectral)
        {
            for (auto &c : hdr)
            {
                Vec3f xyz = c;
                c[0] =  3.2404542f * xyz[0] - 1.5371385f * xyz[1] - 0.4985314f * xyz[2];
                c[1] = -0.9692660f * xyz[0] + 1.8760108f * xyz[1] + 0.0415560f * xyz[2];
                c[2] =  0.0556434f * xyz[0] - 0.2040259f * xyz[1] + 1.0572252f * xyz[2];
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Metal render took " << elapsedMs << " ms" << std::endl;

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
    addText("Backend",       "GPU Metal compute");
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

    } // @autoreleasepool
}
