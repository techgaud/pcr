#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "Vec3f.h"
#include "GpuDefaults.h"
#include "../../Scenes/Scene.h"

// Forward declare GLFWwindow so this header doesn't pull in the GLFW headers.
struct GLFWwindow;

// GPU-side path tracer. OpenGL 4.3 compute shader. API mirrors the CPU
// Renderer so Gui/main.cpp can swap between them with a typedef.
//
// On Apple platforms the same Gui/main.cpp instead pulls in MetalRenderer
// via the Gpu/GpuRenderer.h shim header. The two backends expose the same
// public surface; only one of the .cpp/.mm sources gets compiled per
// build (CMake selects based on platform).
//
// Threading: the OpenglRenderer borrows a hidden, shared OpenGL context
// that the GUI hands it (created at app startup with glfwCreateWindow +
// GLFW_VISIBLE off + share=mainWindow). render() runs on a worker thread,
// makes the shared context current there, dispatches the compute shader
// in row strips, and releases the context before returning.
class OpenglRenderer
{
public:
    // sharedContext must be a hidden GLFW window created with the GUI's main
    // window passed as the share parameter. The OpenglRenderer doesn't own
    // it, doesn't destroy it.
    OpenglRenderer(int width, int height,
                   int depth, int samples, int shadowSamples,
                   GLFWwindow *sharedContext);
    ~OpenglRenderer();

    // Same hook contract as the CPU Renderer.
    std::atomic<int> *progressRows = nullptr;
    std::atomic<bool> *cancelRequested = nullptr;
    std::function<void(const std::vector<Vec3f> &, int width, int height)> onPartialFrame;

    // Same quality knobs as the CPU Renderer; passed to GLSL as uniforms.
    bool useDenoise   = false;
    bool useMIS       = false;
    bool useRussian   = false;
    bool useStratified = false;
    bool useACES      = false;
    int  aaSamples    = 1;
    bool useAdaptive  = false;
    bool useOIDN      = false;
    // GPU spectral path is phase 6, not yet implemented. Field
    // exists so the shared Gui/main.cpp source compiles against
    // both renderers; assignment from settings is a no-op until
    // the GLSL shader has a spectral mode to consume it.
    bool useSpectral  = false;

    // Hero-wavelength sample count. 4 = stratified hero (Wilkie 2014,
    // current default). 1 = legacy single-wavelength, exposed for
    // benchmarking and visual A/B. The GPU shader takes a fast path
    // at 1: route to tracePathSpectralSingle (already exists for
    // glass dispersion) instead of the vec4 hero kernel. Other
    // values map to 4 (full hero).
    int heroSamples = 4;

    // Field exists so Gui/main.cpp and Cli/Main.cpp compile against
    // either backend with the same surface. OpenGL's local_size_xy is
    // baked into the GLSL kernel string at compile time so changing
    // these at runtime has no effect on this backend. The Metal backend
    // honors them. Defaults centralized in GpuDefaults.h.
    int threadgroupX = pcr::kDefaultThreadgroupX;
    int threadgroupY = pcr::kDefaultThreadgroupY;

    void render(const Scenes::SceneData &scene,
                std::chrono::steady_clock::time_point start,
                const std::string &outputDir);

    std::string lastOutputPath;

private:
    int _width, _height;
    int _maxDepth, _samples, _shadowSamples;
    GLFWwindow *_sharedContext;

    // GL objects, lazily initialized on the worker thread the first time
    // render() is called (because GL calls require a current context).
    unsigned _program = 0;
    unsigned _outputTex = 0;
    unsigned _albedoTex = 0;  // OIDN aux: per-pixel albedo at first hit
    unsigned _normalTex = 0;  // OIDN aux: per-pixel shading normal at first hit
    unsigned _sphereSSBO = 0;
    unsigned _planeSSBO = 0;
    unsigned _triangleSSBO = 0;
    unsigned _materialSSBO = 0;
    unsigned _bvhSSBO = 0;
    unsigned _lightSSBO = 0;
    unsigned _lightTriSSBO = 0;
    bool _initialized = false;

    bool initGL();
    void uploadScene(const Scenes::SceneData &scene, float &outTotalLightArea);
    void destroyGL();
};
