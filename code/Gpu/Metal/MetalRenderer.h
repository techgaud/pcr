#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "Vec3f.h"
#include "GpuDefaults.h"
#include "../../Scenes/Scene.h"

// Forward declare GLFWwindow so this header doesn't pull in the GLFW
// headers. We never actually use a borrowed GL context on Apple (Metal
// stands alone), but the constructor signature matches OpenglRenderer
// so Gui/main.cpp can construct either backend from the same call site.
struct GLFWwindow;

// Metal-backed path tracer. Apple-only. Public interface mirrors
// OpenglRenderer; the typedef in Gpu/GpuRenderer.h selects between them
// at compile time so Gui/main.cpp stays backend-agnostic.
//
// Implementation is in MetalRenderer.mm (Objective-C++). The header is
// kept pure C++ so other translation units can include it without
// pulling in the Objective-C runtime; all id<MTL...> state lives behind
// a forward-declared Impl pImpl.
class MetalRenderer
{
public:
    // sharedContext is unused on Apple (Metal doesn't borrow a GL
    // context), retained in the signature so callers don't need an
    // #ifdef.
    MetalRenderer(int width, int height,
                  int depth, int samples, int shadowSamples,
                  GLFWwindow *sharedContext);
    ~MetalRenderer();

    // Same hook contract as OpenglRenderer / CPU Renderer.
    std::atomic<int> *progressRows = nullptr;
    std::atomic<bool> *cancelRequested = nullptr;
    std::function<void(const std::vector<Vec3f> &, int width, int height)> onPartialFrame;

    // Same quality knobs as the OpenGL renderer; passed to MSL as
    // uniforms. Defaults match the OpenGL backend so the shared
    // Gui/main.cpp doesn't need backend-specific defaults.
    bool useDenoise   = false;
    bool useMIS       = false;
    bool useRussian   = false;
    bool useStratified = false;
    bool useACES      = false;
    int  aaSamples    = 1;
    bool useAdaptive  = false;
    bool useOIDN      = false;
    bool useSpectral  = false;

    // Hero-wavelength sample count. 4 = stratified hero (Wilkie 2014,
    // current default). 1 = legacy single-wavelength, exposed for
    // benchmarking and visual A/B. The MSL kernel takes a fast path
    // at 1: route to tracePathSpectralSingle (already exists for
    // glass dispersion) instead of the float4 hero kernel. Other
    // values map to 4 (full hero).
    int heroSamples = 4;

    // Metal compute threadgroup shape for the path_trace_pass{,_adaptive}
    // kernels. Default (8x8 as of v1.4.1) lives in GpuDefaults.h with
    // the rationale and A/B history; this field is the renderer-side
    // override knob, populated by the CLI / GUI before render(). Clamps
    // to 16x16 at dispatch time if the chosen shape exceeds the
    // pipeline's maxTotalThreadsPerThreadgroup.
    int threadgroupX = pcr::kDefaultThreadgroupX;
    int threadgroupY = pcr::kDefaultThreadgroupY;

    // Architecture toggle. false = megakernel (the v1.4.0+ single-kernel
    // path tracer with per-pixel multi-pass dispatch). true = wavefront
    // (rays rebatched per-material between bounces, divergence-free
    // shading kernels). When wavefront kernels haven't been implemented
    // yet, setting this to true falls back to megakernel and prints a
    // stderr warning once per render. Adaptive sampling is not supported
    // in wavefront mode and is silently ignored when both flags are set.
    // Architecture defaults centralized in GpuDefaults.h. useWavefront
    // = true since v1.4.2 (wavefront beat megakernel ~25% in A/B).
    bool useWavefront = pcr::kDefaultUseWavefront;
    bool wavefrontMultiSample = pcr::kDefaultWavefrontMultiSample;
    // Glass dispersion strategy in wavefront-spectral mode. See
    // GpuDefaults.h for the terminate-vs-fork trade-off. Only consulted
    // when useWavefront && useSpectral are both true; ignored otherwise.
    bool spectralFork = pcr::kDefaultSpectralFork;

    void render(const Scenes::SceneData &scene,
                std::chrono::steady_clock::time_point start,
                const std::string &outputDir);

    std::string lastOutputPath;

    // Opaque pImpl. Forward-declared here, defined inside MetalRenderer.mm
    // so this header stays Obj-C-free and includable from plain C++.
    // Public only so file-local helpers in the .mm can take Impl& without
    // tripping access control; the actual definition is .mm-local, so
    // nothing outside the .mm can do anything with it.
    struct Impl;

private:
    int _width, _height;
    int _maxDepth, _samples, _shadowSamples;
    Impl *_impl = nullptr;
};
