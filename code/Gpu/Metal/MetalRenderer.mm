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
#include "Includes/PngText.h"
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

// Uniforms total: 144 bytes (34 active fields = 136 bytes + 2 trailing
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
// The trailing five fields (passIdx, aaIdx, sampleStart, sampleCount,
// batchEndOfAa) are used only by path_trace_pass / path_trace_pass_adaptive;
// the legacy path_trace ignores them.
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
    int   batchEndOfAa;
    int   _pad1, _pad2;
};

// Per-pixel Welford state for the adaptive multi-pass kernel.
// Layout matches the host PCRWelfordState POD below.
struct Welford {
    float stagingR;   // sum of contributions within the current aaIdx
    float stagingG;
    float stagingB;
    float meanR;      // running Welford mean of completed-aaIdx means
    float meanG;
    float meanB;
    float m2R;        // running Welford M2 of completed-aaIdx means
    float m2G;
    float m2B;
    int   taken;      // number of completed aaIdx data points
    int   done;       // 1 = converged, skip remaining work
    int   _pad0;
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

    // First-touch detection: when spatial chopping is enabled different
    // tiles have different `passIdx` for their first touch, so we use
    // the (aaIdx, sampleStart) tuple instead. For the non-chopped case
    // this is true exactly when passIdx == 0, so behavior is unchanged
    // when there's no spatial chop.
    if (u.aaIdx == 0 && u.sampleStart == 0) {
        output.write(float4(accum, 0.0f), uint2(pix.x, pix.y));
    } else {
        float4 prev = output.read(uint2(pix.x, pix.y));
        output.write(prev + float4(accum, 0.0f), uint2(pix.x, pix.y));
    }
}

// ---- Multi-pass adaptive kernel ------------------------------------
//
// Used by the saturation-friendly dispatch path when useAdaptive=ON.
// Same per-pass full-image dispatch geometry as path_trace_pass (so
// the GPU stays saturated) but augmented with per-pixel Welford state
// living in a device-side buffer. State per pixel:
//   - stagingRGB: running sum of contributions within the CURRENT aaIdx
//   - meanRGB / m2RGB / taken: Welford accumulator over COMPLETED aaIdx
//     means, matching the strip path's Welford granularity exactly
//     (each data point = one full AA iteration's mean)
//   - done: 1 if relative variance < threshold after >=4 aaIdx taken
//
// Per pass:
//   - On the first dispatch that touches a given pixel (aaIdx == 0 and
//     sampleStart == 0), the kernel clobbers the per-pixel state to
//     zero. Spatial chopping is detected via (aaIdx, sampleStart) rather
//     than passIdx so each tile's first dispatch initializes its own
//     pixels; without spatial chopping that's exactly passIdx == 0.
//   - If done==1, the kernel returns immediately and the pixel's
//     prior output-texture write (the running Welford mean) stands.
//   - Otherwise: same sampling loop as path_trace_pass adds to staging.
//   - If batchEndOfAa==1, the staging is divided by samples to get the
//     AA iteration mean, that mean is folded into Welford, staging is
//     reset, and the running mean is written to the output texture.
//     The convergence check fires after >=4 aaIdx taken using the same
//     rel-variance threshold (0.05) the strip path uses.
//
// Output texture writes contain the MEAN, not a sum: done pixels keep
// their last-written mean intact across subsequent passes. CPU readback
// for this path therefore SKIPS the divide-by-total-contributions step
// that non-adaptive multi-pass uses.
//
// Memory cost: 48 bytes per pixel for the Welford buffer (private
// storage mode, GPU-only). 1080^2 = 56 MB, 4K = 800 MB, 8K = 3.2 GB.
// Well within M1 Ultra's 64 GB unified memory budget.

kernel void path_trace_pass_adaptive(
    constant Uniforms                          &u            [[buffer(0)]],
    device const GpuSphere                     *spheres      [[buffer(1)]],
    device const GpuPlane                      *planes       [[buffer(2)]],
    device const GpuMaterial                   *materials    [[buffer(3)]],
    device const GpuTriangle                   *triangles    [[buffer(4)]],
    device const GpuBvhNode                    *bvhNodes     [[buffer(5)]],
    device const GpuLight                      *lights       [[buffer(6)]],
    device const GpuLightTriangle              *lightTris    [[buffer(7)]],
    device Welford                             *welford      [[buffer(8)]],
    texture2d<float, access::read_write>        output       [[texture(0)]],
    texture2d<float, access::write>             albedoOut    [[texture(1)]],
    texture2d<float, access::write>             normalOut    [[texture(2)]],
    uint2                                       gid          [[thread_position_in_grid]])
{
    Scene S = { u, spheres, planes, materials, triangles, bvhNodes, lights, lightTris };

    int2 pix = int2(int(gid.x) + u.xOffset, int(gid.y) + u.yOffset);
    if (pix.x >= u.xEnd || pix.x >= u.width ||
        pix.y >= u.yEnd || pix.y >= u.height) return;

    int wfIdx = pix.y * u.width + pix.x;

    // First-touch detection for this pixel. Spatial chopping makes
    // different tiles' first passes have different passIdx values, so
    // we key off (aaIdx, sampleStart) which is (0, 0) for every tile's
    // first dispatch. For the non-chopped case this matches passIdx==0.
    if (u.aaIdx == 0 && u.sampleStart == 0) {
        welford[wfIdx].stagingR = 0.0f;
        welford[wfIdx].stagingG = 0.0f;
        welford[wfIdx].stagingB = 0.0f;
        welford[wfIdx].meanR    = 0.0f;
        welford[wfIdx].meanG    = 0.0f;
        welford[wfIdx].meanB    = 0.0f;
        welford[wfIdx].m2R      = 0.0f;
        welford[wfIdx].m2G      = 0.0f;
        welford[wfIdx].m2B      = 0.0f;
        welford[wfIdx].taken    = 0;
        welford[wfIdx].done     = 0;
    }

    // Done pixels: skip the work, leave output texture write intact.
    if (welford[wfIdx].done != 0) return;

    // OIDN aux capture: only when uniforms ask for it. Done pixels
    // are excluded by the early return above, but converged pixels
    // already had aux captured on pass 0 so they don't miss anything.
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

    uint jitterSeed = uint(pix.x) * 1973u + uint(pix.y) * 9277u
                    + uint(u.frameSeed) * 26699u
                    + uint(u.aaIdx) * 16127u;
    float2 jpix = float2(pix);
    if (aaN > 1) {
        jpix.x += rand(jitterSeed) - 0.5f;
        jpix.y += rand(jitterSeed) - 0.5f;
    }

    uint seed = jitterSeed
              + uint(u.sampleStart) * 7919u
              + 0x9e3779b9u;

    int sampleEnd = u.sampleStart + u.sampleCount;
    float3 batchSum = float3(0.0f);

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
                batchSum += singleLambdaXYZ(lambda, rad);
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
                batchSum += xyz * 0.25f;
            }
        } else {
            batchSum += tracePath(S, jpix, r1, r2, seed);
        }
    }

    // Fold the batch sum into the per-pixel staging buffer.
    welford[wfIdx].stagingR += batchSum.x;
    welford[wfIdx].stagingG += batchSum.y;
    welford[wfIdx].stagingB += batchSum.z;

    if (u.batchEndOfAa == 0) return;

    // Last batch of this aaIdx: finalize the AA iteration mean and
    // feed it into the Welford accumulator.
    float invSamples = 1.0f / float(max(1, u.samples));
    float3 aaMean = float3(welford[wfIdx].stagingR,
                           welford[wfIdx].stagingG,
                           welford[wfIdx].stagingB) * invSamples;

    welford[wfIdx].stagingR = 0.0f;
    welford[wfIdx].stagingG = 0.0f;
    welford[wfIdx].stagingB = 0.0f;

    int takenNew = welford[wfIdx].taken + 1;
    float3 mean = float3(welford[wfIdx].meanR,
                         welford[wfIdx].meanG,
                         welford[wfIdx].meanB);
    float3 m2   = float3(welford[wfIdx].m2R,
                         welford[wfIdx].m2G,
                         welford[wfIdx].m2B);

    float3 delta  = aaMean - mean;
    mean += delta / float(takenNew);
    float3 delta2 = aaMean - mean;
    m2 += delta * delta2;

    welford[wfIdx].meanR = mean.x;
    welford[wfIdx].meanG = mean.y;
    welford[wfIdx].meanB = mean.z;
    welford[wfIdx].m2R   = m2.x;
    welford[wfIdx].m2G   = m2.y;
    welford[wfIdx].m2B   = m2.z;
    welford[wfIdx].taken = takenNew;

    // Convergence check: same threshold as the strip-path kernel
    // (0.05 on rel variance) since data points have identical
    // granularity (mean of `samples` paths per aaIdx).
    if (takenNew >= 4) {
        float3 variance = m2 / float(takenNew - 1);
        float3 rel = variance / (mean * mean + float3(0.01f));
        if (max(max(rel.r, rel.g), rel.b) < 0.05f) {
            welford[wfIdx].done = 1;
        }
    }

    // Output texture holds the running mean. Done pixels' last
    // write persists across the remaining passes since they
    // early-return above.
    output.write(float4(mean, 1.0f), uint2(pix.x, pix.y));
}

// ============================================================
// ==== Wavefront path tracer kernels =========================
// ============================================================
//
// Architecturally distinct from the megakernel path_trace_* family
// above. The megakernel runs the WHOLE path-tracing loop for a pixel
// in one kernel invocation (one bounce, two bounces, ..., output);
// divergence within a SIMD group when different pixels' rays take
// different code paths is the dominant ALU stall source.
//
// Wavefront splits the loop across multiple specialized kernels:
//
//   1. wf_generate_primary_rays  -- camera unprojection, one ray per
//      pixel, write to RayState SoA buffers (commit #2, this one).
//   2. wf_intersect              -- BVH traversal, one ray per thread,
//      write hit info to HitInfo SoA buffers (commit #3).
//   3. wf_compact_by_material    -- SIMD-group-batched scatter of
//      hit records into per-material queues using
//      simd_prefix_exclusive_sum() and one atomic per SIMD group
//      (commit #4).
//   4. wf_shade_diffuse / _mirror / _glass / _emissive -- one shading
//      kernel per material type; each is divergence-free internally
//      since all rays in its queue share the same material (commit
//      #5). New bounce rays land back in the ray queue.
//   5. Host driver loops 2 -> 3 -> 4 for `depth` bounces, then writes
//      accumulated colors to the output texture (commit #6).
//
// Output should be statistically equivalent to megakernel (same rays,
// same bounces, same math, just rebatched per material for ALU
// coherence) but not bit-identical because the RNG seeding scheme
// has to be redone (rays interleave through phases instead of
// running sequentially per thread).
//
// Buffer-binding convention: kernel uniforms at buffer(0); RayState
// SoA at buffers(1..16) in the order originX, originY, originZ, dirX,
// dirY, dirZ, throughputR, throughputG, throughputB, colorR, colorG,
// colorB, pixelIdx, rngState, bounceDepth, alive. HitInfo SoA at
// buffers(17..23) in order matIdx, hitX, hitY, hitZ, normalX, normalY,
// normalZ. Scene data starts at buffer(24) when kernels need it.
// Wavefront kernels never use textures - output writeback happens
// in the post-pipeline driver kernel.

kernel void wf_generate_primary_rays(
    constant Uniforms          &u                  [[buffer(0)]],
    device packed_float3       *origin             [[buffer(1)]],
    device packed_float3       *dir                [[buffer(2)]],
    device packed_float3       *throughput         [[buffer(3)]],
    device packed_float3       *color              [[buffer(4)]],
    device uint                *pixelIdx           [[buffer(5)]],
    device uint                *rngState           [[buffer(6)]],
    device uint                *bounceDepth        [[buffer(7)]],
    device uint                *alive              [[buffer(8)]],
    device const Welford       *pixelWelford       [[buffer(26)]],
    device float4              *lambdas            [[buffer(27)]],
    device float4              *spectralThroughput [[buffer(28)]],
    uint                        gid                [[thread_position_in_grid]])
{
    // 1D dispatch over rayCount = pixelCount * samplesPerPass.
    // rayIdx layout: rays for sample-slot s live at indices
    // [s*pixelCount, (s+1)*pixelCount). Matches what wf_output_writeback
    // expects when it sums per-pixel across samples.
    uint pixelCount = uint(u.width) * uint(u.height);
    uint samples    = max(1u, uint(u.sampleCount));
    uint rayCount   = pixelCount * samples;
    if (gid >= rayCount) return;

    uint pixelIdxLocal = gid % pixelCount;
    uint sampleSlot    = gid / pixelCount;
    int2 pix = int2(int(pixelIdxLocal % uint(u.width)),
                    int(pixelIdxLocal / uint(u.width)));

    // Adaptive convergence early-exit: pixelWelford[pixelIdxLocal].done
    // is set by the writeback kernel once a pixel's running mean has
    // stabilized (taken >= 4 aaIdx with rel variance < 0.05). Done
    // pixels skip primary-ray gen entirely - we just zero alive and
    // the downstream intersect / compaction / shading kernels treat
    // the ray as terminated.
    if (u.useAdaptive != 0 && pixelWelford[pixelIdxLocal].done != 0) {
        alive[gid] = 0u;
        bounceDepth[gid] = 0u;
        pixelIdx[gid] = pixelIdxLocal;
        return;
    }

    // RNG seed: includes sampleSlot so the N rays for a pixel within
    // one pipeline run get decorrelated random sequences. The
    // megakernel hands different `s` values into the same per-pixel
    // RNG by stepping the seed within the kernel loop; wavefront
    // mints a fresh seed per (pix, aaIdx, sampleStart+sampleSlot).
    uint jitterSeed = uint(pix.x) * 1973u + uint(pix.y) * 9277u
                    + uint(u.frameSeed) * 26699u
                    + uint(u.aaIdx) * 16127u;

    // AA jitter: same sub-pixel offset across every sample-batch
    // within a single aaIdx (depends on aaIdx + pix, not on
    // sampleStart). Matches megakernel.
    int aaN = max(1, u.aaSamples);
    float2 jpix = float2(pix);
    if (aaN > 1) {
        jpix.x += rand(jitterSeed) - 0.5f;
        jpix.y += rand(jitterSeed) - 0.5f;
    }

    // Camera unprojection: same math as the megakernel's primary-ray
    // setup. NDC -> view space, focal-length scale baked into the FOV
    // tangent, image axes are right/up but Metal output texture is
    // top-down so the y component is flipped.
    // FOV is in DEGREES on the host side. Megakernel's tracePath wraps
    // it with `tan(PI / 180.0 * 0.5 * fov)`; an earlier copy of this
    // ray-gen forgot the degree-to-radian conversion and was producing
    // a wildly-wide-FOV camera (the "zoomed in to a corner" wavefront
    // output you reported). Match megakernel exactly.
    float aspect = float(u.width) / float(u.height);
    float scale  = tan(PI / 180.0f * 0.5f * u.fov);
    float2 nd = (jpix * 2.0f - float2(u.width, u.height)) /
                float2(u.width, u.height);
    float3 rayDir = normalize(float3(nd.x * scale * aspect,
                                     -nd.y * scale,
                                     -1.0f));
    float3 rayOrigin = float3(u.originX, u.originY, u.originZ);

    // Write the RayState SoA at this ray's slot (gid).
    origin[gid]      = rayOrigin;
    dir[gid]         = rayDir;
    throughput[gid]  = float3(1.0f);
    color[gid]       = float3(0.0f);
    pixelIdx[gid]    = pixelIdxLocal;

    // Per-ray RNG seed: includes sampleStart (per-pass offset) AND
    // sampleSlot (within-pass offset) so each unique (aaIdx, sample)
    // tuple gets its own decorrelated sequence. 12379 is a random
    // prime, distinct from the 7919 used for sampleStart, to keep
    // the two contributions from collapsing into the same bit pattern.
    uint seed = jitterSeed
              + uint(u.sampleStart) * 7919u
              + sampleSlot * 12379u
              + 0x9e3779b9u;

    // Spectral mode: assign the 4 hero wavelengths via Wilkie 2014
    // stratified sampling, one random offset + 3 strides through the
    // 400..700 nm range with wraparound. Matches the megakernel hero=4
    // recipe in path_trace_pass exactly so wavefront and megakernel
    // sample the same spectral distribution. Initial throughput is
    // float4(1) since no surface has been touched yet.
    if (u.useSpectral != 0) {
        float kSpan   = kLambdaMax - kLambdaMin;
        float kStride = kSpan / 4.0f;
        float4 lams;
        lams.x = kLambdaMin + rand(seed) * kSpan;
        lams.y = lams.x + kStride;          if (lams.y > kLambdaMax) lams.y -= kSpan;
        lams.z = lams.x + kStride * 2.0f;   if (lams.z > kLambdaMax) lams.z -= kSpan;
        lams.w = lams.x + kStride * 3.0f;   if (lams.w > kLambdaMax) lams.w -= kSpan;
        lambdas[gid]            = lams;
        spectralThroughput[gid] = float4(1.0f);
    }

    rngState[gid]    = seed;
    bounceDepth[gid] = 0;
    alive[gid]       = 1;
}

// Wavefront intersect kernel. Reads each ray's origin/direction from the
// RayState SoA, runs the existing sceneIntersect helper (which checks
// spheres, planes, and the triangle BVH in a fixed priority order), and
// writes the resulting matIdx + hit position + surface normal into the
// HitInfo SoA. Dead rays (alive == 0) skip the work and get matIdx = -1
// written as sentinel so subsequent compaction sees them as "no hit /
// terminated"; rays that intersected nothing also get matIdx = -1 and
// will terminate at compaction time.
//
// Buffer-binding layout (packed_float3 SoA, per the convention
// established by ray-gen):
//   buffer(0)   = uniforms
//   buffer(1)   = origin (packed_float3, read)
//   buffer(2)   = dir    (packed_float3, read)
//   buffer(8)   = alive  (uint, read)
//   buffer(9)   = matIdx (int, write)
//   buffer(10)  = hit    (packed_float3, write)
//   buffer(11)  = normal (packed_float3, write)
//   buffer(12..18) = scene buffers (spheres, planes, materials,
//                    triangles, bvh, lights, lightTris)
// The other RayState fields (throughput, color, pixelIdx, rngState,
// bounceDepth) aren't accessed here and intentionally aren't bound.
//
// Dispatch geometry: 1D over the linear ray-buffer length
// (width * height). Threadgroup size is the same threadgroup knob from
// commit-184a7cf (configurable via GUI debug panel / CLI --threadgroup-x
// --threadgroup-y, default 8x8 = 64 threads). The 1D dispatch uses x =
// threadgroupX * threadgroupY, y = 1, z = 1; the kernel reads
// thread_position_in_grid as a uint and bounds-checks against the
// ray count.

kernel void wf_intersect(
    constant Uniforms                          &u            [[buffer(0)]],
    device const packed_float3                 *origin       [[buffer(1)]],
    device const packed_float3                 *dir          [[buffer(2)]],
    device const uint                          *alive        [[buffer(8)]],
    device int                                 *matIdxOut    [[buffer(9)]],
    device packed_float3                       *hit          [[buffer(10)]],
    device packed_float3                       *normalOut    [[buffer(11)]],
    device const GpuSphere                     *spheres      [[buffer(12)]],
    device const GpuPlane                      *planes       [[buffer(13)]],
    device const GpuMaterial                   *materials    [[buffer(14)]],
    device const GpuTriangle                   *triangles    [[buffer(15)]],
    device const GpuBvhNode                    *bvhNodes     [[buffer(16)]],
    device const GpuLight                      *lights       [[buffer(17)]],
    device const GpuLightTriangle              *lightTris    [[buffer(18)]],
    uint                                        gid          [[thread_position_in_grid]])
{
    // rayCount = pixelCount * samplesPerPass for multi-spp wavefront;
    // megakernel sets sampleCount=1 effectively because it dispatches
    // one ray per pixel per pass (wavefront 1-spp also has sampleCount=1).
    uint rayCount = uint(u.width) * uint(u.height) * max(1u, uint(u.sampleCount));
    if (gid >= rayCount) return;

    if (alive[gid] == 0u) {
        matIdxOut[gid] = -1;
        return;
    }

    float3 ro = origin[gid];
    float3 rd = dir[gid];

    Scene S = { u, spheres, planes, materials, triangles, bvhNodes,
                lights, lightTris };

    float3 hp, N;
    int mi;
    if (sceneIntersect(S, ro, rd, hp, N, mi)) {
        matIdxOut[gid] = mi;
        hit[gid]       = hp;
        normalOut[gid] = N;
    } else {
        matIdxOut[gid] = -1;
    }
}

// Wavefront compaction: scatter rays into per-material queues using
// SIMD-group-batched atomic counter increments.
//
// Apple's WWDC22 'Scale compute workloads' explicitly flags global
// atomics as a primary bottleneck on multi-core M-series GPUs, and
// recommends SIMD-group / threadgroup-level batching to reduce atomic
// pressure. AMD's GPUOpen 'Fast Compaction with mbcnt' converges on
// the same pattern from the AMD side. Per-thread atomic-append at
// pcr's 1080^2 = 1.17M rays scale would hit 1.17M atomic_inc against
// 4 hot counters; this batched pattern hits ~36K (one per SIMD group)
// for the same scatter, a 32x reduction in atomic traffic.
//
// Algorithm per SIMD group of 32 threads:
//   For each bucket b in {0,1,2,3}:
//     1. pred = (this thread's matType == b) ? 1 : 0
//     2. prefix = simd_prefix_exclusive_sum(pred)  // 0..N within group
//     3. count  = simd_sum(pred)                   // 0..32 total active
//     4. if count > 0:
//        lane-0 does atomic_fetch_add(queueCounter[b], count) -> base
//        broadcast base to all lanes
//        if pred: queue[b][base + prefix] = my_gid
//
// Material classification reads the GpuMaterial struct at matIdx:
//   emissive surface (any rgb > 0)  -> bucket 3
//   metallic (mirror)               -> bucket 1
//   transparent (glass)             -> bucket 2
//   else (diffuse)                  -> bucket 0
// Dead rays (matIdx == -1, set by wf_intersect on no-hit / dead-ray
// passthrough) get matType=-1 and skip all buckets, terminating the
// path. Their accumulated color stays as written at the last shading
// step (or initial zero for primary-ray escapees).
//
// Buffer layout:
//   buffer(0)         uniforms
//   buffer(9)         matIdx (input from HitInfo SoA)
//   buffer(14)        materials (scene buffer, read for classification)
//   buffer(19)        queueCounters (atomic_uint[4])
//   buffer(20..23)    queue[diffuse|mirror|glass|emissive]
//
// queueCounters must be zeroed by the host (blit-encoder fill) before
// each compaction dispatch. The driver in commit #6 handles that.

kernel void wf_compact_by_material(
    constant Uniforms                          &u             [[buffer(0)]],
    device const int                           *matIdx        [[buffer(9)]],
    device const GpuMaterial                   *materials     [[buffer(14)]],
    device atomic_uint                         *queueCounters [[buffer(19)]],
    device uint                                *queueDiffuse  [[buffer(20)]],
    device uint                                *queueMirror   [[buffer(21)]],
    device uint                                *queueGlass    [[buffer(22)]],
    device uint                                *queueEmissive [[buffer(23)]],
    uint                                        gid           [[thread_position_in_grid]])
{
    uint rayCount = uint(u.width) * uint(u.height) * max(1u, uint(u.sampleCount));
    bool inBounds = (gid < rayCount);

    // Classify this thread's ray. Out-of-bounds threads and dead-ray
    // threads (matIdx == -1) participate in the SIMD prefix-sum scans
    // with pred=0 - they don't contribute to any queue but they do
    // participate in the SIMD-group reductions, which is what
    // simd_prefix_exclusive_sum / simd_sum require to behave
    // correctly (the intrinsics are uniform across all 32 lanes).
    int matType = -1;
    if (inBounds)
    {
        int mIdx = matIdx[gid];
        if (mIdx >= 0)
        {
            GpuMaterial m = materials[mIdx];
            bool hasEmissive = (m.emissive.x > 0.0f ||
                                m.emissive.y > 0.0f ||
                                m.emissive.z > 0.0f);
            if (hasEmissive)        matType = 3;
            else if (m.metallic)    matType = 1;
            else if (m.transparent) matType = 2;
            else                    matType = 0;
        }
    }

    // Four buckets, four SIMD-batched atomic-appends. The loop unrolls
    // at compile time (b is a constexpr literal in each iteration).
    for (int b = 0; b < 4; b++)
    {
        uint pred  = (matType == b) ? 1u : 0u;
        uint prefix = simd_prefix_exclusive_sum(pred);
        uint count  = simd_sum(pred);

        uint base = 0u;
        if (count > 0u)
        {
            if (simd_is_first())
            {
                base = atomic_fetch_add_explicit(&queueCounters[b],
                                                 count,
                                                 memory_order_relaxed);
            }
            base = simd_broadcast(base, 0);
        }

        if (pred != 0u)
        {
            uint slot = base + prefix;
            switch (b)
            {
                case 0: queueDiffuse[slot]  = gid; break;
                case 1: queueMirror[slot]   = gid; break;
                case 2: queueGlass[slot]    = gid; break;
                case 3: queueEmissive[slot] = gid; break;
            }
        }
    }
}

// ============================================================
// ==== Wavefront shading kernels (one per material type) ====
// ============================================================
//
// Each kernel reads its material's compacted queue (produced by
// wf_compact_by_material) and processes ONLY rays of that material
// type. No material branching inside a kernel = no divergence inside
// a SIMD group = the whole point of wavefront.
//
// Common pattern per kernel:
//   1. Read queue[matType][thread] to get the ray's global index gid.
//   2. Load ray state (origin, direction, throughput, color, rngState,
//      bounceDepth) and hit info (hitPos, normal, matIdx) from the
//      SoA buffers at index gid.
//   3. Read materials[matIdx]. Flip the surface normal if back-facing
//      (entering = dot(rd, N) < 0; if not entering, N = -N) so all
//      downstream math sees an outward-facing normal.
//   4. Apply the material's bounce math (different per kernel).
//   5. For the diffuse kernel: also do next-event-estimation against
//      area lights (shadow ray cast, MIS-weighted contribution added
//      to the color accumulator). Other kernels skip NEE because their
//      BSDFs are delta functions (no diffuse component to direct-light).
//   6. Russian roulette + max-depth termination: bounceDepth + 1, RR
//      check using max(throughput), set alive=0 if terminated.
//   7. Write updated state back to RayState SoA.
//
// Dispatch shape: 1D over the queue length. The host driver (commit
// #6) reads queueCounters[matType] after compaction to size each
// shading kernel's grid; if the queue is empty for a given material,
// the dispatch is skipped entirely.
//
// Buffer-binding layout (packed_float3 SoA, fits within Metal's 31-
// buffer-per-kernel limit). Each kernel binds only what it touches:
//   buffer(0)   uniforms (read)
//   buffer(1)   origin (packed_float3, R+W per bounce)
//   buffer(2)   dir    (packed_float3, R+W per bounce)
//   buffer(3)   throughput (packed_float3, R+W)
//   buffer(4)   color  (packed_float3, R+W, light accumulator)
//   buffer(6)   rngState (uint, R+W)
//   buffer(7)   bounceDepth (uint, R+W)
//   buffer(8)   alive (uint, W on terminate)
//   buffer(9)   matIdx (int, R only - populated by intersect)
//   buffer(10)  hit    (packed_float3, R only)
//   buffer(11)  normal (packed_float3, R only)
//   buffer(12..18) scene buffers (R, only diffuse uses all for shadow
//                  rays; mirror/glass/emissive bind just materials at 14)
//   buffer(24)  this material's input queue (read; ray indices)
//   buffer(25)  this material's queue length (queueCounters bound at
//               offset matType*4 so the kernel's `constant uint&`
//               binding loads queueCounters[matType])
// pixelIdx (buffer 5) is unused by shading kernels - the post-pass
// output writeback kernel uses it.

// ---- Emissive: terminate with light contribution -------------------
//
// Ray hit an emissive surface (area light, sun, etc.). In megakernel,
// this would be `radiance += throughput * emissive; break;`. Here we
// add to the color accumulator and set alive=0 so the ray doesn't
// continue to the next bounce.

kernel void wf_shade_emissive(
    constant Uniforms                          &u             [[buffer(0)]],
    device const packed_float3                 *throughput    [[buffer(3)]],
    device packed_float3                       *color         [[buffer(4)]],
    device uint                                *alive         [[buffer(8)]],
    device const int                           *matIdx        [[buffer(9)]],
    device const GpuMaterial                   *materials     [[buffer(14)]],
    device const uint                          *queue         [[buffer(24)]],
    constant uint                              &queueLen      [[buffer(25)]],
    uint                                        tid           [[thread_position_in_grid]])
{
    if (tid >= queueLen) return;
    uint gid = queue[tid];

    int mi = matIdx[gid];
    float3 t = throughput[gid];
    float3 emissive = materials[mi].emissive.rgb;

    color[gid] = float3(color[gid]) + t * emissive;
    alive[gid] = 0u;
}

// ---- Mirror: perfect specular reflection ---------------------------
//
// Reflect the direction off the surface normal, modulate throughput
// by albedo. No PDF division because mirrors are delta-distribution
// BSDFs (the integrand collapses to a single direction). Origin
// nudged by epsilon along the normal to avoid self-intersection.

kernel void wf_shade_mirror(
    constant Uniforms                          &u             [[buffer(0)]],
    device packed_float3                       *origin        [[buffer(1)]],
    device packed_float3                       *dir           [[buffer(2)]],
    device packed_float3                       *throughput    [[buffer(3)]],
    device uint                                *rngState      [[buffer(6)]],
    device uint                                *bounceDepth   [[buffer(7)]],
    device uint                                *alive         [[buffer(8)]],
    device const int                           *matIdx        [[buffer(9)]],
    device const packed_float3                 *hit           [[buffer(10)]],
    device const packed_float3                 *normalIn      [[buffer(11)]],
    device const GpuMaterial                   *materials     [[buffer(14)]],
    device const uint                          *queue         [[buffer(24)]],
    constant uint                              &queueLen      [[buffer(25)]],
    uint                                        tid           [[thread_position_in_grid]])
{
    if (tid >= queueLen) return;
    uint gid = queue[tid];

    float3 rd  = dir[gid];
    float3 N   = normalIn[gid];
    float3 h   = hit[gid];

    bool entering = dot(rd, N) < 0.0f;
    if (!entering) N = -N;

    int mi = matIdx[gid];
    float3 albedo = materials[mi].albedo.rgb;
    float3 t = throughput[gid];

    float3 newDir = reflect(rd, N);
    float3 newOrigin = h + N * 1e-3f;
    t *= albedo;

    uint depth = bounceDepth[gid] + 1u;
    uint seed = rngState[gid];
    bool alive_after = true;
    if (u.useRussian != 0 && depth >= 2u) {
        float p = clamp(max(max(albedo.r, albedo.g), albedo.b), 0.05f, 0.95f);
        if (rand(seed) > p) alive_after = false;
        else t /= p;
    }
    if (depth >= uint(u.depth)) alive_after = false;

    origin[gid]      = newOrigin;
    dir[gid]         = newDir;
    throughput[gid]  = t;
    rngState[gid]    = seed;
    bounceDepth[gid] = depth;
    alive[gid]       = alive_after ? 1u : 0u;
}

// ---- Glass: dielectric refraction via Schlick Fresnel --------------
//
// Calls dielectricBounce (existing helper from megakernel) which
// handles the entering / total-internal-reflection cases and returns
// a new direction + origin. Same math as megakernel's transparent-
// material branch in tracePath.

kernel void wf_shade_glass(
    constant Uniforms                          &u             [[buffer(0)]],
    device packed_float3                       *origin        [[buffer(1)]],
    device packed_float3                       *dir           [[buffer(2)]],
    device packed_float3                       *throughput    [[buffer(3)]],
    device uint                                *rngState      [[buffer(6)]],
    device uint                                *bounceDepth   [[buffer(7)]],
    device uint                                *alive         [[buffer(8)]],
    device const int                           *matIdx        [[buffer(9)]],
    device const packed_float3                 *hit           [[buffer(10)]],
    device const packed_float3                 *normalIn      [[buffer(11)]],
    device const GpuMaterial                   *materials     [[buffer(14)]],
    device const uint                          *queue         [[buffer(24)]],
    constant uint                              &queueLen      [[buffer(25)]],
    uint                                        tid           [[thread_position_in_grid]])
{
    if (tid >= queueLen) return;
    uint gid = queue[tid];

    float3 rd = dir[gid];
    float3 N  = normalIn[gid];
    float3 h  = hit[gid];

    bool entering = dot(rd, N) < 0.0f;
    if (!entering) N = -N;

    int mi = matIdx[gid];
    GpuMaterial mat = materials[mi];
    float3 albedo = mat.albedo.rgb;
    float3 t = throughput[gid];
    uint seed = rngState[gid];

    DielectricOut b = dielectricBounce(rd, N, h, entering, mat.ior, rand(seed));
    t *= albedo;

    uint depth = bounceDepth[gid] + 1u;
    bool alive_after = true;
    if (u.useRussian != 0 && depth >= 2u) {
        float p = clamp(max(max(albedo.r, albedo.g), albedo.b), 0.05f, 0.95f);
        if (rand(seed) > p) alive_after = false;
        else t /= p;
    }
    if (depth >= uint(u.depth)) alive_after = false;

    origin[gid]      = b.origin;
    dir[gid]         = b.dir;
    throughput[gid]  = t;
    rngState[gid]    = seed;
    bounceDepth[gid] = depth;
    alive[gid]       = alive_after ? 1u : 0u;
}

// ---- Diffuse: cosine-weighted hemisphere + NEE direct lighting -----
//
// The most involved kernel. Two contributions:
//   1. Direct lighting via next-event estimation: pick a point on an
//      area light, cast a shadow ray, accumulate the (visibility *
//      G-term * albedo/PI * emissive * totalLightArea) integrand.
//      MIS weighting when useMIS is on (matches megakernel exactly).
//   2. Indirect: cosine-weighted hemisphere sample for the next-bounce
//      direction, throughput *= albedo. (PI cancellation: BRDF =
//      albedo/PI, PDF = cosTheta/PI, BRDF*cosTheta/PDF = albedo. The
//      cosTheta * PI cancellation is why we don't see an explicit
//      cosTheta in the throughput update.)

kernel void wf_shade_diffuse(
    constant Uniforms                          &u             [[buffer(0)]],
    device packed_float3                       *origin        [[buffer(1)]],
    device packed_float3                       *dir           [[buffer(2)]],
    device packed_float3                       *throughput    [[buffer(3)]],
    device packed_float3                       *color         [[buffer(4)]],
    device uint                                *rngState      [[buffer(6)]],
    device uint                                *bounceDepth   [[buffer(7)]],
    device uint                                *alive         [[buffer(8)]],
    device const int                           *matIdx        [[buffer(9)]],
    device const packed_float3                 *hit           [[buffer(10)]],
    device const packed_float3                 *normalIn      [[buffer(11)]],
    device const GpuSphere                     *spheres       [[buffer(12)]],
    device const GpuPlane                      *planes        [[buffer(13)]],
    device const GpuMaterial                   *materials     [[buffer(14)]],
    device const GpuTriangle                   *triangles     [[buffer(15)]],
    device const GpuBvhNode                    *bvhNodes      [[buffer(16)]],
    device const GpuLight                      *lights        [[buffer(17)]],
    device const GpuLightTriangle              *lightTris     [[buffer(18)]],
    device const uint                          *queue         [[buffer(24)]],
    constant uint                              &queueLen      [[buffer(25)]],
    uint                                        tid           [[thread_position_in_grid]])
{
    if (tid >= queueLen) return;
    uint gid = queue[tid];

    float3 rd = dir[gid];
    float3 N  = normalIn[gid];
    float3 h  = hit[gid];

    bool entering = dot(rd, N) < 0.0f;
    if (!entering) N = -N;

    int mi = matIdx[gid];
    float3 albedo = materials[mi].albedo.rgb;
    float3 t = throughput[gid];
    uint seed = rngState[gid];

    Scene S = { u, spheres, planes, materials, triangles, bvhNodes,
                lights, lightTris };

    // Direct lighting via NEE: average u.shadowSamples shadow rays.
    float3 directLo = float3(0.0f);
    if (u.totalLightArea > 0.0f) {
        for (int s = 0; s < u.shadowSamples; s++) {
            float3 sampleP, sampleN, sampleEmissive;
            int sampleMatIdx;
            sampleAreaLight(S, seed, sampleP, sampleN, sampleEmissive, sampleMatIdx);

            float3 Li = sampleP - h;
            float3 wi = normalize(Li);
            float cosTheta = max(0.0f, dot(wi, N));
            float lightDist2 = dot(Li, Li);
            float3 shadowOrigin = (cosTheta <= 0.0f) ? h - N * 1e-3f : h + N * 1e-3f;

            bool occluded = false;
            float3 sh, sN;
            int sMat;
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
                float3 directContrib = (albedo / PI) * sampleEmissive * G * u.totalLightArea;
                if (u.useMIS != 0 && cosLight > 1e-6f) {
                    float pdfLight = lightDist2 / (cosLight * u.totalLightArea);
                    float pdfBrdf  = cosTheta / PI;
                    float w = (pdfLight * pdfLight) /
                              (pdfLight * pdfLight + pdfBrdf * pdfBrdf);
                    directContrib *= w;
                }
                directLo += directContrib;
            }
        }
        directLo /= float(u.shadowSamples);
    }
    // Accumulate direct lighting into the color buffer.
    color[gid] = float3(color[gid]) + t * directLo;

    // Russian roulette: same conditions as megakernel (after bounce 0).
    uint depth = bounceDepth[gid] + 1u;
    bool alive_after = true;
    if (u.useRussian != 0 && depth >= 2u) {
        float p = clamp(max(max(albedo.r, albedo.g), albedo.b), 0.05f, 0.95f);
        if (rand(seed) > p) alive_after = false;
        else t /= p;
    }
    if (depth >= uint(u.depth)) alive_after = false;

    // Indirect: cosine-weighted hemisphere sample. Stratified not
    // supported in wavefront v1.
    float3 newDir = sampleHemisphere(N, seed);
    float3 newOrigin = h + N * 1e-3f;
    t *= albedo;

    origin[gid]      = newOrigin;
    dir[gid]         = newDir;
    throughput[gid]  = t;
    rngState[gid]    = seed;
    bounceDepth[gid] = depth;
    alive[gid]       = alive_after ? 1u : 0u;
}

// ============================================================
// ==== Wavefront output writeback ============================
// ============================================================
//
// Reduce kernel that runs once per pixel at the end of a wavefront
// pipeline pass. Sums the `sampleCount` accumulated colors that
// belong to each pixel (one ray per sample per pixel were processed
// in parallel during the pass) and writes the sum into the output
// texture - clobbering on the first touch, accumulating thereafter
// to handle multi-pass aaSamples * passesPerAa accumulation.
//
// Ray layout in the SoA buffers during a wavefront pass:
//   rayIdx in [0, pixelCount * sampleCount)
//   pixelIdx   = rayIdx % pixelCount
//   sampleSlot = rayIdx / pixelCount
// One thread here per pixel, iterates the sampleCount entries that
// belong to it, sums their colors, writes the result.
//
// First-touch detection mirrors megakernel's path_trace_pass:
// (aaIdx == 0 AND sampleStart == 0) means this is the very first
// contribution to this pixel; later passes accumulate. CPU readback
// divides by total contributions (aaSamples * samples) the same way
// non-adaptive megakernel multi-pass does.

kernel void wf_output_writeback(
    constant Uniforms                          &u             [[buffer(0)]],
    device const packed_float3                 *color         [[buffer(4)]],
    device Welford                             *pixelWelford  [[buffer(26)]],
    texture2d<float, access::read_write>        output        [[texture(0)]],
    uint2                                       gid           [[thread_position_in_grid]])
{
    if (gid.x >= uint(u.width) || gid.y >= uint(u.height)) return;
    uint pixelIdx   = gid.y * uint(u.width) + gid.x;
    uint pixelCount = uint(u.width) * uint(u.height);
    uint samples    = max(1u, uint(u.sampleCount));

    // Adaptive convergence early-exit. Done pixels keep their
    // last-written running-mean in the output texture; we skip the
    // sum-and-accumulate work entirely.
    if (u.useAdaptive != 0 && pixelWelford[pixelIdx].done != 0)
        return;

    float3 sumThisPass = float3(0.0f);
    for (uint s = 0u; s < samples; s++) {
        uint rayIdx = s * pixelCount + pixelIdx;
        sumThisPass += float3(color[rayIdx]);
    }

    if (u.useAdaptive != 0) {
        // Adaptive path: accumulate this pass's contribution into the
        // per-pixel Welford staging buffer. On the last batch of an
        // aaIdx (batchEndOfAa=1), finalize the AA-iteration mean,
        // fold it into the Welford accumulator, check for convergence,
        // and write the running mean to the output texture.
        //
        // Mid-aaIdx passes: don't touch the output texture. It keeps
        // showing the last-written mean from the previous aaIdx
        // boundary. CPU readback for the adaptive path therefore
        // skips the divide-by-total step (texture is already in mean
        // space, not sum space).
        pixelWelford[pixelIdx].stagingR += sumThisPass.x;
        pixelWelford[pixelIdx].stagingG += sumThisPass.y;
        pixelWelford[pixelIdx].stagingB += sumThisPass.z;

        if (u.batchEndOfAa != 0)
        {
            // Sum-of-samples-this-aaIdx -> per-aaIdx mean.
            float invSamples = 1.0f / float(max(1, u.samples));
            float3 aaMean = float3(pixelWelford[pixelIdx].stagingR,
                                    pixelWelford[pixelIdx].stagingG,
                                    pixelWelford[pixelIdx].stagingB) * invSamples;

            // Reset staging for the next aaIdx.
            pixelWelford[pixelIdx].stagingR = 0.0f;
            pixelWelford[pixelIdx].stagingG = 0.0f;
            pixelWelford[pixelIdx].stagingB = 0.0f;

            // Welford update over completed-aaIdx means. Same
            // formulation as megakernel adaptive's path_trace_pass_adaptive
            // so the convergence behavior matches: each data point is
            // one AA iteration's mean, threshold 0.05 on relative
            // variance after >= 4 taken.
            int takenNew = pixelWelford[pixelIdx].taken + 1;
            float3 mean = float3(pixelWelford[pixelIdx].meanR,
                                 pixelWelford[pixelIdx].meanG,
                                 pixelWelford[pixelIdx].meanB);
            float3 m2 = float3(pixelWelford[pixelIdx].m2R,
                               pixelWelford[pixelIdx].m2G,
                               pixelWelford[pixelIdx].m2B);
            float3 delta = aaMean - mean;
            mean += delta / float(takenNew);
            float3 delta2 = aaMean - mean;
            m2 += delta * delta2;

            pixelWelford[pixelIdx].meanR = mean.x;
            pixelWelford[pixelIdx].meanG = mean.y;
            pixelWelford[pixelIdx].meanB = mean.z;
            pixelWelford[pixelIdx].m2R   = m2.x;
            pixelWelford[pixelIdx].m2G   = m2.y;
            pixelWelford[pixelIdx].m2B   = m2.z;
            pixelWelford[pixelIdx].taken = takenNew;

            if (takenNew >= 4) {
                float3 variance = m2 / float(takenNew - 1);
                float3 rel = variance / (mean * mean + float3(0.01f));
                if (max(max(rel.r, rel.g), rel.b) < 0.05f) {
                    pixelWelford[pixelIdx].done = 1;
                }
            }

            output.write(float4(mean, 1.0f), gid);
        }
        return;
    }

    // Non-adaptive path. First-touch (first dispatch covering this
    // pixel within the render) clobbers the texture; subsequent passes
    // accumulate. Matches the megakernel non-adaptive multi-pass
    // first-touch convention so the texture-accumulation semantics
    // are identical between the two architectures.
    if (u.aaIdx == 0 && u.sampleStart == 0) {
        output.write(float4(sumThisPass, 0.0f), gid);
    } else {
        float4 prev = output.read(gid);
        output.write(prev + float4(sumThisPass, 0.0f), gid);
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
        // 1 if this pass is the last batch of its aaIdx (so the
        // adaptive multi-pass kernel finalizes the AA mean + folds
        // it into per-pixel Welford). 0 otherwise. Non-adaptive
        // multi-pass and legacy path_trace ignore this field.
        int   batchEndOfAa;
        int   _pad1, _pad2;
    };
    static_assert(sizeof(Uniforms) == 144,
                  "Uniforms must be 144 bytes (multiple of 16) so per-pass "
                  "buffer offsets are 16-aligned for MSL vectorized loads");

    // Per-pixel state for the adaptive multi-pass kernel. Layout matches
    // the MSL Welford struct above. GPU-only (private storage); host
    // doesn't read or write the contents - sizeof here is purely so we
    // can newBufferWithLength(numPixels * sizeof(PCRWelfordState)).
    struct PCRWelfordState
    {
        float stagingR, stagingG, stagingB;
        float meanR,    meanG,    meanB;
        float m2R,      m2G,      m2B;
        int   taken;
        int   done;
        int   _pad0;
    };
    static_assert(sizeof(PCRWelfordState) == 48,
                  "PCRWelfordState must be 48 bytes to match the MSL Welford struct");

    // ---------- Wavefront ray-state SoA buffers --------------------------
    //
    // Per-render allocation for the wavefront path tracer. Each "field" of
    // a ray gets its own MTLBuffer (Struct of Arrays layout) so that
    // adjacent threads in a SIMD group read adjacent memory addresses,
    // matching Apple Silicon's preferred coalesced-load access pattern.
    //
    // The alternative (AoS, one big struct per ray, all fields packed
    // contiguously) is simpler to think about but loses coalescing because
    // a SIMD group of 32 threads each reading "their" ray's origin_x ends
    // up scattering 32 reads across 32 different cache lines instead of
    // one fat coalesced load. SoA flips the layout so all 32 threads'
    // origin_x values are contiguous, one cache line, one transaction.
    // Production wavefront renderers (PBRT-v4, OptiX-style, Hyperion)
    // converge on SoA for this reason.
    //
    // Two struct families:
    //   - RayState (persistent across bounces): origin, direction, the
    //     running color accumulator and throughput, pixel index, RNG
    //     state, bounce counter, alive flag. Updated by the shading
    //     kernels after each bounce.
    //   - HitInfo (transient per bounce): material index, hit position,
    //     surface normal. Written by the intersect kernel, read by the
    //     per-material shading kernels, overwritten on the next bounce.
    //
    // Storage: MTLResourceStorageModePrivate (GPU-only, no CPU-side
    // mapping needed). Allocated at render start sized by width*height
    // since wavefront processes one (aaIdx, sample-batch) pass at a time
    // (full-AA-times-samples in-flight at once is impractical: at 1080^2
    // with aa=4 samples=2048 that's 9.6B rays = ~960 GB of state, doesn't
    // fit). The pass-loop runs the wavefront pipeline aaSamples *
    // passesPerAa times, same multi-pass structure as megakernel.
    //
    // Sizing per ray: 16 float fields + 4 uint fields = 80 bytes ray state
    // + 7 float fields + 1 int field = 32 bytes hit info = 112 bytes/ray.
    // Spectral mode adds 8 more floats per ray (lambdas + spectralThroughput,
    // two float4 buffers) for a total of 144 bytes/ray. 1080^2 = 132 MB
    // RGB, 168 MB spectral; 4K = 1.87 GB / 2.4 GB; 8K = 7.5 GB / 9.6 GB.
    // M1 Ultra's 64 GB unified memory comfortable up through 8K either way.
    //
    // These structs are populated by allocateWavefrontBuffers() and live
    // for the duration of a single render() call. Buffers self-release
    // via ARC when the holder goes out of scope.
    struct WavefrontRayBuffers
    {
        // Persistent ray state. Vec3 fields packed into MSL packed_float3
        // (12 bytes per element, alignment 4) to fit within Metal's
        // 31-buffer-per-kernel limit. Separate XYZ buffers would cost 3
        // binding slots per field and put the diffuse shading kernel
        // over the limit; packed_float3 is 1 slot per field at the same
        // memory bandwidth as 3 separate float buffers (32-thread SIMD
        // reads 32 * 12 = 384 bytes contiguous = 6 cache lines, matching
        // the separate-buffer case). This is the "hybrid SoA" layout
        // PBRT-v4 uses for the same reason.
        id<MTLBuffer> origin;       // packed_float3
        id<MTLBuffer> dir;          // packed_float3
        id<MTLBuffer> throughput;   // packed_float3
        id<MTLBuffer> color;        // packed_float3
        id<MTLBuffer> pixelIdx;     // uint
        id<MTLBuffer> rngState;     // uint
        id<MTLBuffer> bounceDepth;  // uint
        id<MTLBuffer> alive;        // uint

        // Transient hit info (overwritten by intersect each bounce).
        id<MTLBuffer> matIdx;       // int
        id<MTLBuffer> hit;          // packed_float3
        id<MTLBuffer> normal;       // packed_float3

        // Per-material queues populated by wf_compact_by_material.
        // Each holds up to rayCount uint ray-indices (worst case: every
        // ray hits the same material type). Shading kernels for material
        // type T read queueT[0..queueCounters[T]-1] as their input set.
        id<MTLBuffer> queueDiffuse;
        id<MTLBuffer> queueMirror;
        id<MTLBuffer> queueGlass;
        id<MTLBuffer> queueEmissive;

        // Per-pixel Welford state (PCRWelfordState struct, 48 bytes per
        // pixel) for adaptive wavefront. Always allocated when wavefront
        // is active even if useAdaptive=false; the kernels guard their
        // reads with `u.useAdaptive` so the storage is dead-code in
        // non-adaptive mode. The buffer is pixelCount-sized (not
        // rayCount-sized), since adaptive convergence is decided per
        // pixel, not per sample slot.
        id<MTLBuffer> pixelWelford;

        // 4 atomic_uint counters, one per material queue. Zeroed by the
        // host (blit-encoder fill) before each compaction dispatch.
        // After compaction, queueCounters[t] = number of valid entries
        // in queue<t>; the shading-kernel dispatch reads this to size
        // its grid.
        id<MTLBuffer> queueCounters;

        // Spectral-mode-only ray state. The wavefront shading kernels
        // use Wilkie 2014 hero wavelength sampling with N=4 stratified
        // wavelengths per ray (same scheme as megakernel hero=4). On a
        // dispersive glass hit, the kernel terminates secondaries by
        // zeroing spectralThroughput.yzw and lets lambdas.x continue
        // alone (PBRT-v4 / Mitsuba 3 convention). Both buffers are
        // allocated unconditionally so the kernels can declare them as
        // bound parameters; in RGB mode they're dead memory.
        //   lambdas:            float4 per ray (16 B). The four hero
        //                       wavelengths assigned at ray-gen.
        //   spectralThroughput: float4 per ray (16 B). Per-wavelength
        //                       scalar throughput, replaces the
        //                       packed_float3 throughput buffer in
        //                       spectral mode. RGB mode keeps using
        //                       the packed_float3 throughput; the two
        //                       are mutually exclusive at any given
        //                       render. Cost ~35 MB at 1080^2, ~537 MB
        //                       at 4K. Acceptable.
        id<MTLBuffer> lambdas;            // float4
        id<MTLBuffer> spectralThroughput; // float4

        NSUInteger rayCount = 0;
        bool valid = false;
    };

    // Allocate the wavefront SoA buffer set sized to hold `rayCount` rays.
    // All buffers are private storage (GPU-only). Returns a struct whose
    // .valid flag is false if any allocation failed; caller checks .valid
    // before dispatching and skips wavefront on failure.
    //
    // Currently unused at the call sites: the wavefront kernels haven't
    // shipped, so renderInternal still falls back to megakernel before
    // ever reaching this allocator. Once the kernels land, the wavefront
    // dispatch path in renderInternal calls this, runs the pipeline, and
    // lets the returned struct die at scope-exit (ARC handles release).
    [[maybe_unused]] WavefrontRayBuffers allocateWavefrontBuffers(
        id<MTLDevice> device, NSUInteger rayCount, NSUInteger pixelCount)
    {
        WavefrontRayBuffers wf;
        wf.rayCount = rayCount;
        auto alloc = [&](size_t bytes) -> id<MTLBuffer> {
            return [device newBufferWithLength:bytes
                                       options:MTLResourceStorageModePrivate];
        };
        // packed_float3 in MSL is 12 bytes, alignment 4 (no padding,
        // unlike float3 which is 16-byte-aligned vec). Host side just
        // sizes the buffer to N*12 bytes; the kernel reinterprets via
        // its `device packed_float3 *` parameter declaration.
        const size_t packedF3Bytes = rayCount * 3 * sizeof(float);
        const size_t uintBytes     = rayCount * sizeof(uint32_t);
        const size_t intBytes      = rayCount * sizeof(int32_t);

        wf.origin      = alloc(packedF3Bytes);
        wf.dir         = alloc(packedF3Bytes);
        wf.throughput  = alloc(packedF3Bytes);
        wf.color       = alloc(packedF3Bytes);
        wf.pixelIdx    = alloc(uintBytes);
        wf.rngState    = alloc(uintBytes);
        wf.bounceDepth = alloc(uintBytes);
        wf.alive       = alloc(uintBytes);
        wf.matIdx      = alloc(intBytes);
        wf.hit         = alloc(packedF3Bytes);
        wf.normal      = alloc(packedF3Bytes);

        wf.queueDiffuse  = alloc(uintBytes);
        wf.queueMirror   = alloc(uintBytes);
        wf.queueGlass    = alloc(uintBytes);
        wf.queueEmissive = alloc(uintBytes);
        wf.queueCounters = alloc(4 * sizeof(uint32_t));

        // Per-pixel Welford state for adaptive wavefront. Sized to
        // pixelCount (one entry per output pixel, NOT one per ray) since
        // adaptive convergence is decided per pixel. ~48 MB at 1080^2,
        // ~768 MB at 4K, ~3 GB at 8K. Allocated unconditionally so the
        // ray-gen + writeback kernels can declare it as a bound buffer
        // parameter; the kernels guard their reads behind u.useAdaptive
        // so it's dead memory in non-adaptive renders.
        wf.pixelWelford = alloc(pixelCount * 48);  // 48 = sizeof(PCRWelfordState)

        // Spectral-mode-only ray state (lambdas + spectralThroughput).
        // Always allocated so the kernels can declare these as bound
        // parameters and the dispatch driver can bind them
        // unconditionally; the kernel signatures branch on u.useSpectral
        // before reading. Each is float4 = 16 B per ray.
        const size_t float4Bytes = rayCount * 4 * sizeof(float);
        wf.lambdas            = alloc(float4Bytes);
        wf.spectralThroughput = alloc(float4Bytes);

        wf.valid = (wf.origin && wf.dir && wf.throughput && wf.color &&
                    wf.pixelIdx && wf.rngState && wf.bounceDepth && wf.alive &&
                    wf.matIdx && wf.hit && wf.normal &&
                    wf.queueDiffuse && wf.queueMirror &&
                    wf.queueGlass && wf.queueEmissive &&
                    wf.queueCounters && wf.pixelWelford &&
                    wf.lambdas && wf.spectralThroughput);
        return wf;
    }

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
    // Three compute pipelines from the same MSL library:
    //   pipeline            = legacy single-pass kernel (path_trace),
    //                         kept as a fallback for unusual configs.
    //   pipelinePass        = multi-pass accumulator kernel
    //                         (path_trace_pass), used when useAdaptive
    //                         is off. Each invocation covers the full
    //                         image and contributes one (aaIdx,
    //                         sample-batch) slice to the running sum.
    //   pipelinePassAdapt   = multi-pass accumulator kernel with
    //                         per-pixel Welford state and sparse skip
    //                         (path_trace_pass_adaptive), used when
    //                         useAdaptive is on. Same dispatch geometry
    //                         as pipelinePass, full GPU saturation;
    //                         converged pixels early-return so total
    //                         work shrinks for low-variance regions.
    id<MTLComputePipelineState> pipeline           = nil;
    id<MTLComputePipelineState> pipelinePass       = nil;
    id<MTLComputePipelineState> pipelinePassAdapt  = nil;

    // Wavefront pipeline states (compiled in initMetal alongside
    // megakernel pipelines, dispatched by the host driver when
    // useWavefront=true). Currently only the primary-ray-generation
    // kernel exists; intersect / shading kernels arrive in subsequent
    // commits per the wavefront commit sequence.
    id<MTLComputePipelineState> pipelineWfRayGen     = nil;
    id<MTLComputePipelineState> pipelineWfIntersect  = nil;
    id<MTLComputePipelineState> pipelineWfCompact    = nil;
    id<MTLComputePipelineState> pipelineWfShadeEmissive = nil;
    id<MTLComputePipelineState> pipelineWfShadeMirror   = nil;
    id<MTLComputePipelineState> pipelineWfShadeGlass    = nil;
    id<MTLComputePipelineState> pipelineWfShadeDiffuse  = nil;
    id<MTLComputePipelineState> pipelineWfWriteback     = nil;

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
    // MTLBuffer with one Uniforms entry per pass, since per-pass data
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

        id<MTLFunction> fnPassAdapt = [lib newFunctionWithName:@"path_trace_pass_adaptive"];
        if (!fnPassAdapt)
        {
            std::cerr << "MetalRenderer: MSL kernel 'path_trace_pass_adaptive' not found"
                      << std::endl;
            return false;
        }
        im.pipelinePassAdapt = [im.device newComputePipelineStateWithFunction:fnPassAdapt error:&err];
        if (!im.pipelinePassAdapt)
        {
            std::cerr << "MetalRenderer: adaptive multi-pass compute pipeline build failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "(no error info)")
                      << std::endl;
            return false;
        }

        // Wavefront pipelines. Currently only the primary-ray-generation
        // kernel ships; intersect / shading / driver kernels arrive in
        // commits #3-#6. Each kernel is independently compiled here so
        // the pipeline list is build-time visible and missing kernels
        // surface as a clear MSL-not-found error rather than at first
        // dispatch.
        id<MTLFunction> fnWfRayGen = [lib newFunctionWithName:@"wf_generate_primary_rays"];
        if (!fnWfRayGen)
        {
            std::cerr << "MetalRenderer: MSL kernel 'wf_generate_primary_rays' not found"
                      << std::endl;
            return false;
        }
        im.pipelineWfRayGen = [im.device newComputePipelineStateWithFunction:fnWfRayGen error:&err];
        if (!im.pipelineWfRayGen)
        {
            std::cerr << "MetalRenderer: wavefront primary-ray pipeline build failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "(no error info)")
                      << std::endl;
            return false;
        }

        id<MTLFunction> fnWfIntersect = [lib newFunctionWithName:@"wf_intersect"];
        if (!fnWfIntersect)
        {
            std::cerr << "MetalRenderer: MSL kernel 'wf_intersect' not found"
                      << std::endl;
            return false;
        }
        im.pipelineWfIntersect = [im.device newComputePipelineStateWithFunction:fnWfIntersect error:&err];
        if (!im.pipelineWfIntersect)
        {
            std::cerr << "MetalRenderer: wavefront intersect pipeline build failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "(no error info)")
                      << std::endl;
            return false;
        }

        id<MTLFunction> fnWfCompact = [lib newFunctionWithName:@"wf_compact_by_material"];
        if (!fnWfCompact)
        {
            std::cerr << "MetalRenderer: MSL kernel 'wf_compact_by_material' not found"
                      << std::endl;
            return false;
        }
        im.pipelineWfCompact = [im.device newComputePipelineStateWithFunction:fnWfCompact error:&err];
        if (!im.pipelineWfCompact)
        {
            std::cerr << "MetalRenderer: wavefront compaction pipeline build failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "(no error info)")
                      << std::endl;
            return false;
        }

        // Four shading kernels, one per material type. Each binds only
        // the SoA buffers it touches (drops unused ones to stay under
        // Metal's 31-buffer-per-kernel limit). Mirror/glass/emissive
        // bind a minimal scene (materials only); diffuse binds the full
        // scene because NEE shadow rays go through sceneIntersect.
        //
        // Returns the pipeline state by value (nil on failure) rather
        // than writing through an out-parameter pointer. ARC defaults
        // an `id<X> *` parameter to __autoreleasing ownership, but the
        // im.pipelineWf* fields are __strong (struct fields default to
        // strong), and ARC refuses to materialize the autorelease/
        // writeback temp through a non-local strong pointer. Returning
        // by value lets the caller's assignment do the ARC handoff
        // cleanly without any ownership annotation gymnastics.
        auto buildShadingPipeline = [&](const char *name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn) {
                std::cerr << "MetalRenderer: MSL kernel '" << name
                          << "' not found" << std::endl;
                return nil;
            }
            NSError *e = nil;
            id<MTLComputePipelineState> ps =
                [im.device newComputePipelineStateWithFunction:fn error:&e];
            if (!ps) {
                std::cerr << "MetalRenderer: wavefront shading pipeline '"
                          << name << "' build failed: "
                          << (e ? [[e localizedDescription] UTF8String] : "(no error info)")
                          << std::endl;
                return nil;
            }
            return ps;
        };
        im.pipelineWfShadeEmissive = buildShadingPipeline("wf_shade_emissive");
        if (!im.pipelineWfShadeEmissive) return false;
        im.pipelineWfShadeMirror   = buildShadingPipeline("wf_shade_mirror");
        if (!im.pipelineWfShadeMirror)   return false;
        im.pipelineWfShadeGlass    = buildShadingPipeline("wf_shade_glass");
        if (!im.pipelineWfShadeGlass)    return false;
        im.pipelineWfShadeDiffuse  = buildShadingPipeline("wf_shade_diffuse");
        if (!im.pipelineWfShadeDiffuse)  return false;

        id<MTLFunction> fnWfWriteback = [lib newFunctionWithName:@"wf_output_writeback"];
        if (!fnWfWriteback)
        {
            std::cerr << "MetalRenderer: MSL kernel 'wf_output_writeback' not found"
                      << std::endl;
            return false;
        }
        im.pipelineWfWriteback = [im.device newComputePipelineStateWithFunction:fnWfWriteback error:&err];
        if (!im.pipelineWfWriteback)
        {
            std::cerr << "MetalRenderer: wavefront output-writeback pipeline build failed: "
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

    // Architecture selection. Wavefront baseline ships as of this
    // commit and runs whenever useWavefront=true under one of the
    // supported configurations. Configs that wavefront doesn't (yet)
    // support fall back to megakernel with a stderr warning so the
    // user isn't silently surprised by getting one architecture when
    // they asked for another. Currently unsupported in wavefront:
    //   1. useSpectral - wavefront shading kernels are RGB-only; the
    //      spectral hero-wavelength path lives only in megakernel's
    //      tracePathSpectral. Wavefront spectral support is its own
    //      project (per-wavelength queues or per-wavelength state in
    //      the ray buffers).
    //   2. useAdaptive AND wavefrontMultiSample - adaptive works in
    //      1spp wavefront (the writeback can finalize a per-aaIdx mean
    //      cleanly since each pipeline run is exactly one sample), but
    //      multi-sample-per-pass wavefront would need a more complex
    //      reduction over the per-pass samples.
    bool effectiveWavefront = useWavefront;
    if (useWavefront)
    {
        const char *fallbackReason = nullptr;
        if (useSpectral)
            fallbackReason = "spectral mode";
        else if (useAdaptive && wavefrontMultiSample)
            fallbackReason = "adaptive + multi-sample-per-pass";
        if (fallbackReason)
        {
            std::cerr << "MetalRenderer: wavefront doesn't support "
                      << fallbackReason << " yet. Falling back to "
                      << "megakernel for this render."
                      << std::endl;
            effectiveWavefront = false;
        }
    }

    if (!initMetal(*_impl, _width, _height)) return;

    float totalLightArea = 0.f;
    uploadScene(*_impl, scene, totalLightArea);

    // Plane count = walls + plane-kind area lights (matches OpenGL).
    int planeLightCount = 0;
    for (const auto &L : scene.areaLights)
        if (L.kind == Scenes::AreaLightKind::Plane) planeLightCount++;

    // One dispatch path, two kernel variants selected at runtime by
    // useAdaptive:
    //
    //   useAdaptive OFF -> path_trace_pass (non-adaptive accumulator).
    //                      Each dispatch covers the FULL image at full
    //                      GPU saturation (e.g. 1080^2 / (32*32) = 1156
    //                      threadgroups vs. 48 M1-Ultra cores) but only
    //                      computes one (aaIdx, sample-batch) slice per
    //                      pass. The output texture acts as a running-
    //                      sum accumulator; CPU divides by total
    //                      contribution count at readback.
    //
    //   useAdaptive ON  -> path_trace_pass_adaptive (Welford in a
    //                      per-pixel device buffer). Same dispatch
    //                      geometry as above for the same saturation
    //                      win. Each pass adds its batch into a per-
    //                      pixel staging slot; at the last batch of an
    //                      aaIdx, the staging gets divided to produce
    //                      one AA-iteration mean which feeds Welford,
    //                      same granularity as the legacy strip path.
    //                      The kernel writes the running Welford mean
    //                      (not a sum) to the output texture, and done-
    //                      flag pixels early-return on subsequent
    //                      passes so total work shrinks for low-
    //                      variance regions.
    //
    // The strip-path kernel (path_trace, pipeline `pipeline`) stays in
    // the MSL library as a fallback / reference but is no longer
    // dispatched: adaptive's multi-pass equivalent now exists and is
    // ~6x faster on Picture-class workloads on M1 Ultra.
    //
    // Post-dispatch (wait + audit + readback + tone-map + PNG) is
    // shared. Non-adaptive normalization (divide accumulator by total
    // contributions) and the XYZ -> sRGB conversion for spectral mode
    // happen after readback below. Adaptive output is already a mean
    // so the divide is skipped (the kernel writes the running mean
    // directly).

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
    int totalContributions = 0;

    // ---- Multi-pass dispatch (adaptive or non-adaptive) ----------
    //
    // Each command buffer dispatches one tile of the image with one of
    // the path_trace_pass{,_adaptive} kernels. Total work per pixel is
    // aaSamples * samples * shadow * depth ops; we split that across
    // enough passes (and, when the full image at samplesPerPass=1 still
    // exceeds the dispatch budget, enough spatial tiles) to keep each
    // dispatch under Apple's compute watchdog.
    //
    // Watchdog-safe budget is on TOTAL ops per dispatch (not per pixel).
    // Calibrated against M1 Ultra's saturated multi-pass throughput
    // (~3.15 G ops/sec on Picture-class kernels). 9B ops per dispatch
    // ~= 3 sec, well under the watchdog.
    //
    // Two-axis budgeting:
    //   1. Try to fit a full-image dispatch by picking samplesPerPass <=
    //      _samples that keeps width*height*samplesPerPass*opsPerPixel
    //      <= budget.
    //   2. If samplesPerPass clamps to 1 and we still exceed budget,
    //      tile the image into row strips. tileH = budget /
    //      (width * opsPerPixel). Each tile dispatches one
    //      (aaIdx, batch, tileIdx) at samplesPerPass=1.
    //
    //   1080^2 Picture (1.17M pixels x 192 ops/pixel/sample):
    //     samplesPerPass = 9B / (1.17M * 192) = 40, ~50 passes/AA, 1 tile
    //   4K Picture (16.7M pixels):
    //     samplesPerPass = 9B / (16.7M * 192) = 2, ~1024 passes/AA, 1 tile
    //   8K Picture (67M pixels):
    //     samplesPerPass = 9B / (67M * 192) = 0.7 -> floor to 1, 1 tile.
    //     ~13B ops/dispatch ~= 4 sec, borderline vs. watchdog.
    //   16K Picture (268M pixels):
    //     samplesPerPass = 1 (clamped). Full-image dispatch is
    //     ~52B ops ~= 16 sec, would be killed. Spatial chop kicks
    //     in: tileH = 9B / (16384 * 192) = 2860 rows, 6 tiles total.
    //     Each tile = 16384 * 2860 * 192 = 9B ops, on budget.

    long long opsPerPixelPerSample = (long long)_shadowSamples * _maxDepth;
    if (useSpectral) opsPerPixelPerSample =
        (long long)((double)opsPerPixelPerSample * 2.5);

    constexpr long long kTargetOpsPerDispatch = 9'000'000'000LL;
    long long pixelCount = std::max(1LL, (long long)_width * _height);
    long long opsPerPixelBudget = std::max(
        1LL, kTargetOpsPerDispatch / pixelCount);
    int samplesPerPass = std::max(1,
        (int)(opsPerPixelBudget / std::max(1LL, opsPerPixelPerSample)));
    samplesPerPass = std::min(samplesPerPass, _samples);

    // Spatial chopping kicks in only when samplesPerPass is already
    // clamped to 1 and the full-image dispatch still exceeds budget.
    // Otherwise tileH defaults to the full image height (one tile).
    int tileH = _height;
    if (samplesPerPass == 1)
    {
        long long opsPerFullRow = (long long)_width * opsPerPixelPerSample;
        long long rowsPerDispatch = std::max(
            1LL, kTargetOpsPerDispatch / std::max(1LL, opsPerFullRow));
        if (rowsPerDispatch < _height)
            tileH = (int)rowsPerDispatch;
    }
    int tilesY = (_height + tileH - 1) / tileH;

    // Wavefront override. Two sub-modes:
    //   wavefrontMultiSample=false (default): 1 sample per pipeline run.
    //     Smallest working set (rayCount = pixelCount). Highest dispatch
    //     count - totalPasses = aaSamples * samples instead of
    //     aaSamples * passesPerAa.
    //   wavefrontMultiSample=true: keep megakernel-computed samplesPerPass.
    //     Working set scales linearly with samplesPerPass (more memory
    //     bandwidth pressure) but dispatch count drops back to megakernel-
    //     equivalent. Net effect is workload-dependent.
    // Spatial chopping (tilesY > 1) doesn't apply to wavefront: its
    // kernels are individually small relative to the watchdog budget
    // regardless of resolution.
    if (effectiveWavefront)
    {
        if (!wavefrontMultiSample)
            samplesPerPass = 1;
        tilesY = 1;
    }

    int passesPerAa = (_samples + samplesPerPass - 1) / samplesPerPass;
    int aaN = std::max(1, aaSamples);
    int totalPasses = aaN * passesPerAa * tilesY;
    totalContributions = aaN * _samples;

    id<MTLComputePipelineState> activePipeline =
        useAdaptive ? _impl->pipelinePassAdapt : _impl->pipelinePass;

    // User-configurable threadgroup shape. Defaults to 32x32 (matches
    // the v1.4.0 hardcoded value); 8x8 / 16x16 / 32x8 are common shapes
    // worth A/B-testing. Validate against the pipeline's limit and fall
    // back to 16x16 if invalid (e.g. user picked something the pipeline
    // can't actually run with). Total threads also has to be > 0; treat
    // a zero in either axis as "use the default".
    int tgX = (threadgroupX > 0) ? threadgroupX : 32;
    int tgY = (threadgroupY > 0) ? threadgroupY : 32;
    if (activePipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(tgX * tgY))
    {
        std::cerr << "MetalRenderer: requested threadgroup "
                  << tgX << "x" << tgY << " (=" << (tgX * tgY)
                  << " threads) exceeds pipeline limit of "
                  << activePipeline.maxTotalThreadsPerThreadgroup
                  << "; falling back to 16x16."
                  << std::endl;
        tgX = 16;
        tgY = 16;
    }
    MTLSize threadsPerGroup = MTLSizeMake(tgX, tgY, 1);

    // Per-pixel Welford buffer for adaptive multi-pass. Initialized
    // by the kernel on passIdx==0 (no separate clear kernel needed).
    // GPU-only private storage so writes stay on-chip.
    id<MTLBuffer> welfordBuf = nil;
    if (useAdaptive)
    {
        size_t welfordBytes =
            (size_t)_width * (size_t)_height * sizeof(PCRWelfordState);
        welfordBuf = [_impl->device newBufferWithLength:welfordBytes
                                                options:MTLResourceStorageModePrivate];
        if (!welfordBuf)
        {
            std::cerr << "MetalRenderer: failed to allocate "
                      << welfordBytes
                      << "-byte Welford buffer for adaptive multi-pass."
                      << std::endl;
            return;
        }
    }

    // Per-pass uniforms. Pass ordering is (aaIdx outer, batchIdx middle,
    // tileIdx innermost). That keeps each pixel's batch sequence in
    // (0,0), (0,1), ..., (0,passesPerAa-1), (1,0), ... order for its
    // single tile, which is what the adaptive Welford accumulator and
    // the OIDN-aux first-touch logic both expect.
    size_t uniformsStride = sizeof(Uniforms);
    id<MTLBuffer> uniformsBuf =
        [_impl->device newBufferWithLength:uniformsStride * (size_t)totalPasses
                                   options:MTLResourceStorageModeShared];
    Uniforms *uniformsPtr = (Uniforms *)[uniformsBuf contents];
    int passesPerAaTotal = passesPerAa * tilesY;
    for (int p = 0; p < totalPasses; p++)
    {
        int aaIdx       = p / passesPerAaTotal;
        int withinAa    = p % passesPerAaTotal;
        int batchIdx    = withinAa / tilesY;
        int tileIdx     = withinAa % tilesY;
        int sampleStart = batchIdx * samplesPerPass;
        int sampleEnd   = std::min(sampleStart + samplesPerPass, _samples);
        int yStart      = tileIdx * tileH;
        int yEnd        = std::min(yStart + tileH, _height);

        Uniforms u = uBase;
        u.passIdx      = p;
        u.aaIdx        = aaIdx;
        u.sampleStart  = sampleStart;
        u.sampleCount  = sampleEnd - sampleStart;
        u.batchEndOfAa = (batchIdx == passesPerAa - 1) ? 1 : 0;
        u.yOffset      = yStart;
        u.yEnd         = yEnd;
        // OIDN aux capture: once per pixel only. With spatial chop the
        // aux must run on every tile's first dispatch (aaIdx==0,
        // sampleStart==0) so each pixel gets its albedo+normal written
        // exactly once.
        u.writeAux     = (useOIDN && aaIdx == 0 && sampleStart == 0) ? 1 : 0;
        uniformsPtr[p] = u;
    }

    // Wavefront allocation: per-render, GPU-only ray-state buffers.
    // Sized for samplesPerPass rays per pixel concurrently in flight.
    //   wavefrontMultiSample=false: rayCount = pixelCount (one ray per pixel)
    //   wavefrontMultiSample=true:  rayCount = pixelCount * samplesPerPass
    //     where samplesPerPass is megakernel's budget-derived value (~40 at
    //     1080^2 Picture). Working set grows linearly; alloc may fail at
    //     very high resolutions in multi-spp mode, in which case the
    //     renderer falls back to megakernel.
    WavefrontRayBuffers wfBufs;
    if (effectiveWavefront)
    {
        NSUInteger pixelCount = NSUInteger(_width) * NSUInteger(_height);
        NSUInteger rayCount = pixelCount * NSUInteger(samplesPerPass);
        wfBufs = allocateWavefrontBuffers(_impl->device, rayCount, pixelCount);
        if (!wfBufs.valid)
        {
            std::cerr << "MetalRenderer: failed to allocate wavefront SoA "
                      << "buffers for " << rayCount << " rays "
                      << "(samplesPerPass=" << samplesPerPass
                      << "); falling back to megakernel for this render."
                      << std::endl;
            effectiveWavefront = false;
        }
    }

    if (effectiveWavefront)
    {
        // Wavefront dispatch path. One command buffer per pass. Each
        // command buffer encodes: ray-gen -> (intersect, compact, four
        // shading kernels) x depth -> output writeback. queueCounters
        // are zeroed via blit-encoder fillBuffer at the start of each
        // bounce; Metal command-buffer ordering serializes the blit
        // ahead of subsequent compute dispatches, no explicit barriers
        // needed.
        //
        // Each shading kernel is dispatched at rayCount threads (with
        // a bounds-check against queueLen inside the kernel) rather
        // than indirectly with the actual queue length. Indirect
        // dispatch via MTLDispatchThreadgroupsIndirectArguments would
        // avoid the wasted lane launches at the cost of more host
        // code; deferred until profiling shows it matters.

        NSUInteger pixelCount = NSUInteger(_width) * NSUInteger(_height);
        NSUInteger rayCount   = pixelCount * NSUInteger(samplesPerPass);
        // 2D dispatch over (width, height) for output writeback (one
        // thread per pixel).
        MTLSize threadgroups2D = MTLSizeMake(
            (NSUInteger)((_width  + tgX - 1) / tgX),
            (NSUInteger)((_height + tgY - 1) / tgY), 1);
        // 1D dispatch over rayCount for ray-gen / intersect / compact /
        // shading. rayCount in 1spp mode is pixelCount; in multi-spp
        // mode it's pixelCount * samplesPerPass.
        NSUInteger linearThreadsPerGroup = NSUInteger(tgX) * NSUInteger(tgY);
        MTLSize threadgroups1D = MTLSizeMake(
            (rayCount + linearThreadsPerGroup - 1) / linearThreadsPerGroup, 1, 1);
        MTLSize threadsPerGroup1D = MTLSizeMake(linearThreadsPerGroup, 1, 1);

        // Zero the per-pixel Welford state buffer once at render start.
        // Metal private storage is uninitialized; the writeback kernel
        // and ray-gen kernel both READ from this buffer (welford.done
        // and welford.staging*) on every pass, so garbage initial values
        // would cause pixels to be misclassified as "done" or accumulate
        // into NaN staging. A single blit-fill is essentially free
        // compared to the render and avoids the alternative of kernel-
        // side first-touch init logic.
        if (useAdaptive)
        {
            id<MTLCommandBuffer> initCmd = [_impl->queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [initCmd blitCommandEncoder];
            NSUInteger welfordBytes = pixelCount * 48;
            [blit fillBuffer:wfBufs.pixelWelford
                       range:NSMakeRange(0, welfordBytes)
                       value:0];
            [blit endEncoding];
            [initCmd commit];
            // Queue is serial; the fill completes before any subsequent
            // cmdbuf's compute work. No waitUntilCompleted needed.
        }

        for (int p = 0; p < totalPasses; p++)
        {
            if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
                break;

            id<MTLCommandBuffer> cmdbuf = [_impl->queue commandBuffer];

            // ---- Ray-gen ----
            // Buffer layout (packed_float3 SoA, MSL hard-caps buffer
            // index at 30):
            //   0 uniforms; 1..4 origin/dir/throughput/color (packed_float3);
            //   5..8 pixelIdx/rngState/bounceDepth/alive (uint);
            //   9 matIdx (int); 10..11 hit/normal (packed_float3);
            //   12..18 scene; 19 queueCounters; 20..23 four queues;
            //   24 shading queue input; 25 queueLen (counter offset);
            //   26 pixelWelford (PCRWelfordState[]);
            //   27 lambdas (float4, hero wavelengths, spectral-only);
            //   28 spectralThroughput (float4, per-wavelength scalar
            //      throughput, spectral-only).
            // Dispatched 1D over rayCount (= pixelCount * samplesPerPass)
            // so multi-sample-per-pass mode lights up one ray per pixel
            // per sample concurrently.
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:_impl->pipelineWfRayGen];
                [enc setBuffer:uniformsBuf       offset:p * uniformsStride atIndex:0];
                [enc setBuffer:wfBufs.origin      offset:0 atIndex:1];
                [enc setBuffer:wfBufs.dir         offset:0 atIndex:2];
                [enc setBuffer:wfBufs.throughput  offset:0 atIndex:3];
                [enc setBuffer:wfBufs.color       offset:0 atIndex:4];
                [enc setBuffer:wfBufs.pixelIdx    offset:0 atIndex:5];
                [enc setBuffer:wfBufs.rngState    offset:0 atIndex:6];
                [enc setBuffer:wfBufs.bounceDepth offset:0 atIndex:7];
                [enc setBuffer:wfBufs.alive       offset:0 atIndex:8];
                [enc setBuffer:wfBufs.pixelWelford       offset:0 atIndex:26];
                [enc setBuffer:wfBufs.lambdas            offset:0 atIndex:27];
                [enc setBuffer:wfBufs.spectralThroughput offset:0 atIndex:28];
                [enc dispatchThreadgroups:threadgroups1D threadsPerThreadgroup:threadsPerGroup1D];
                [enc endEncoding];
            }

            // ---- (intersect, compact, shade*4) x depth ----
            for (int bounce = 0; bounce < _maxDepth; bounce++)
            {
                // Zero the per-bounce queue counters.
                {
                    id<MTLBlitCommandEncoder> blit = [cmdbuf blitCommandEncoder];
                    [blit fillBuffer:wfBufs.queueCounters
                              range:NSMakeRange(0, 4 * sizeof(uint32_t))
                              value:0];
                    [blit endEncoding];
                }

                // Intersect.
                {
                    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                    [enc setComputePipelineState:_impl->pipelineWfIntersect];
                    [enc setBuffer:uniformsBuf       offset:p * uniformsStride atIndex:0];
                    [enc setBuffer:wfBufs.origin      offset:0 atIndex:1];
                    [enc setBuffer:wfBufs.dir         offset:0 atIndex:2];
                    [enc setBuffer:wfBufs.alive       offset:0 atIndex:8];
                    [enc setBuffer:wfBufs.matIdx      offset:0 atIndex:9];
                    [enc setBuffer:wfBufs.hit         offset:0 atIndex:10];
                    [enc setBuffer:wfBufs.normal      offset:0 atIndex:11];
                    [enc setBuffer:_impl->sphereBuf   offset:0 atIndex:12];
                    [enc setBuffer:_impl->planeBuf    offset:0 atIndex:13];
                    [enc setBuffer:_impl->materialBuf offset:0 atIndex:14];
                    [enc setBuffer:_impl->triangleBuf offset:0 atIndex:15];
                    [enc setBuffer:_impl->bvhBuf      offset:0 atIndex:16];
                    [enc setBuffer:_impl->lightBuf    offset:0 atIndex:17];
                    [enc setBuffer:_impl->lightTriBuf offset:0 atIndex:18];
                    [enc dispatchThreadgroups:threadgroups1D threadsPerThreadgroup:threadsPerGroup1D];
                    [enc endEncoding];
                }

                // Compact by material.
                {
                    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                    [enc setComputePipelineState:_impl->pipelineWfCompact];
                    [enc setBuffer:uniformsBuf            offset:p * uniformsStride atIndex:0];
                    [enc setBuffer:wfBufs.matIdx            offset:0 atIndex:9];
                    [enc setBuffer:_impl->materialBuf       offset:0 atIndex:14];
                    [enc setBuffer:wfBufs.queueCounters     offset:0 atIndex:19];
                    [enc setBuffer:wfBufs.queueDiffuse      offset:0 atIndex:20];
                    [enc setBuffer:wfBufs.queueMirror       offset:0 atIndex:21];
                    [enc setBuffer:wfBufs.queueGlass        offset:0 atIndex:22];
                    [enc setBuffer:wfBufs.queueEmissive     offset:0 atIndex:23];
                    [enc dispatchThreadgroups:threadgroups1D threadsPerThreadgroup:threadsPerGroup1D];
                    [enc endEncoding];
                }

                // Four shading kernels. queueLen is read from
                // queueCounters[t] by binding the counter buffer at the
                // queueLen slot with offset t*4 - the kernel sees this
                // as a 4-byte uint via its `constant uint &queueLen`
                // binding, which performs a non-atomic load. Metal's
                // command-buffer ordering guarantees compaction's
                // atomic writes are visible to subsequent shading
                // kernels in the same cmdbuf.
                auto encodeShading = [&](id<MTLComputePipelineState> pipeline,
                                         id<MTLBuffer> queueBuf,
                                         NSUInteger counterOffset,
                                         bool needsFullScene,
                                         bool needsColor) {
                    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                    [enc setComputePipelineState:pipeline];
                    [enc setBuffer:uniformsBuf       offset:p * uniformsStride atIndex:0];
                    [enc setBuffer:wfBufs.origin      offset:0 atIndex:1];
                    [enc setBuffer:wfBufs.dir         offset:0 atIndex:2];
                    [enc setBuffer:wfBufs.throughput  offset:0 atIndex:3];
                    if (needsColor)
                        [enc setBuffer:wfBufs.color   offset:0 atIndex:4];
                    [enc setBuffer:wfBufs.rngState    offset:0 atIndex:6];
                    [enc setBuffer:wfBufs.bounceDepth offset:0 atIndex:7];
                    [enc setBuffer:wfBufs.alive       offset:0 atIndex:8];
                    [enc setBuffer:wfBufs.matIdx      offset:0 atIndex:9];
                    [enc setBuffer:wfBufs.hit         offset:0 atIndex:10];
                    [enc setBuffer:wfBufs.normal      offset:0 atIndex:11];
                    [enc setBuffer:_impl->materialBuf offset:0 atIndex:14];
                    if (needsFullScene)
                    {
                        [enc setBuffer:_impl->sphereBuf   offset:0 atIndex:12];
                        [enc setBuffer:_impl->planeBuf    offset:0 atIndex:13];
                        [enc setBuffer:_impl->triangleBuf offset:0 atIndex:15];
                        [enc setBuffer:_impl->bvhBuf      offset:0 atIndex:16];
                        [enc setBuffer:_impl->lightBuf    offset:0 atIndex:17];
                        [enc setBuffer:_impl->lightTriBuf offset:0 atIndex:18];
                    }
                    [enc setBuffer:queueBuf            offset:0 atIndex:24];
                    [enc setBuffer:wfBufs.queueCounters offset:counterOffset atIndex:25];
                    [enc dispatchThreadgroups:threadgroups1D threadsPerThreadgroup:threadsPerGroup1D];
                    [enc endEncoding];
                };
                // Diffuse needs the full scene (for NEE shadow rays) and
                // the color buffer (writes direct-lighting contribution).
                // Mirror / glass don't accumulate light to color (delta
                // BSDFs). Emissive writes to color but doesn't need
                // scene geometry.
                encodeShading(_impl->pipelineWfShadeDiffuse,  wfBufs.queueDiffuse,
                              0 * sizeof(uint32_t), /*fullScene=*/true,
                              /*needsColor=*/true);
                encodeShading(_impl->pipelineWfShadeMirror,   wfBufs.queueMirror,
                              1 * sizeof(uint32_t), /*fullScene=*/false,
                              /*needsColor=*/false);
                encodeShading(_impl->pipelineWfShadeGlass,    wfBufs.queueGlass,
                              2 * sizeof(uint32_t), /*fullScene=*/false,
                              /*needsColor=*/false);
                encodeShading(_impl->pipelineWfShadeEmissive, wfBufs.queueEmissive,
                              3 * sizeof(uint32_t), /*fullScene=*/false,
                              /*needsColor=*/true);
            }

            // ---- Output writeback ----
            // Per-pixel reduction over the samplesPerPass rays for this
            // pixel, summing colors into the output texture with the
            // first-touch / accumulate convention.
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:_impl->pipelineWfWriteback];
                [enc setBuffer:uniformsBuf offset:p * uniformsStride atIndex:0];
                [enc setBuffer:wfBufs.color offset:0 atIndex:4];
                [enc setBuffer:wfBufs.pixelWelford offset:0 atIndex:26];
                [enc setTexture:_impl->outputTex atIndex:0];
                [enc dispatchThreadgroups:threadgroups2D threadsPerThreadgroup:threadsPerGroup];
                [enc endEncoding];
            }

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
    else
    {
        // Megakernel dispatch path: existing single-kernel-per-pass
        // loop. Kept unchanged from v1.4.1 so wavefront vs megakernel
        // is a clean A/B at runtime.
        for (int p = 0; p < totalPasses; p++)
        {
            if (cancelRequested && cancelRequested->load(std::memory_order_relaxed))
                break;

            const Uniforms &u = uniformsPtr[p];
            int tileH_p = u.yEnd - u.yOffset;
            MTLSize threadgroups = MTLSizeMake(
                (NSUInteger)((_width + tgX - 1) / tgX),
                (NSUInteger)((tileH_p + tgY - 1) / tgY),
                1);

            id<MTLCommandBuffer> cmdbuf = [_impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:activePipeline];
            [enc setBuffer:uniformsBuf      offset:(NSUInteger)(p * uniformsStride) atIndex:0];
            [enc setBuffer:_impl->sphereBuf   offset:0 atIndex:1];
            [enc setBuffer:_impl->planeBuf    offset:0 atIndex:2];
            [enc setBuffer:_impl->materialBuf offset:0 atIndex:3];
            [enc setBuffer:_impl->triangleBuf offset:0 atIndex:4];
            [enc setBuffer:_impl->bvhBuf      offset:0 atIndex:5];
            [enc setBuffer:_impl->lightBuf    offset:0 atIndex:6];
            [enc setBuffer:_impl->lightTriBuf offset:0 atIndex:7];
            if (useAdaptive)
                [enc setBuffer:welfordBuf    offset:0 atIndex:8];
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
    // in the output as black regions matching the killed dispatches.
    // Surface those failures explicitly rather than letting them
    // masquerade as "the kernel ran fine, the image just looks weird."
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
        std::cerr << "Metal render: " << erroredCmdBuffers << " of "
                  << [pending count] << " passes failed (likely "
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

    // Multi-pass normalization. For non-adaptive, the kernel accumulates
    // a sum of per-sample contributions in the output texture and the
    // CPU divides by total contribution count (aaSamples * samples) to
    // get the mean. For adaptive multi-pass, the kernel writes the
    // running Welford mean directly to the texture so the divide is
    // skipped. Either way, spectral mode moves XYZ -> linear sRGB to
    // the CPU here so the kernel can stay in the more numerically clean
    // XYZ space across passes (the conversion is linear so this is
    // mathematically identical to per-pass conversion).
    if (totalContributions > 0)
    {
        if (!useAdaptive)
        {
            float invTotal = 1.0f / (float)totalContributions;
            for (auto &c : hdr) c *= invTotal;
        }
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

    auto addText = [&](const char *key, const std::string &val) {
        pngAddTextBeforeIdat(&state.info_png, key, val);
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
    addText("ThreadgroupX",  std::to_string(tgX));
    addText("ThreadgroupY",  std::to_string(tgY));
    // Architecture records what ACTUALLY ran, not what was requested
    // (useWavefront+useAdaptive falls back to megakernel; alloc failure
    // also falls back). The wavefront sub-mode key disambiguates the
    // 1-sample-per-pass vs multi-sample-per-pass variants so post-hoc
    // A/B grepping by ThreadgroupX/Y/Architecture stays unambiguous.
    addText("Architecture", effectiveWavefront ? "wavefront" : "megakernel");
    if (effectiveWavefront)
        addText("WavefrontMode",
                wavefrontMultiSample ? "multi-spp" : "1spp");

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
