#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "Vec3f.h"
#include "../Scenes/Scene.h"

// Forward declare GLFWwindow so this header doesn't pull in the GLFW headers.
struct GLFWwindow;

// GPU-side path tracer. OpenGL 4.3 compute shader. API matches the CPU
// Renderer so Gui/main.cpp can swap between them with a typedef.
//
// Threading: the GpuRenderer borrows a hidden, shared OpenGL context that
// the GUI hands it (created at app startup with glfwCreateWindow + GLFW_VISIBLE
// off + share=mainWindow). render() runs on a worker thread, makes the shared
// context current there, dispatches the compute shader in row strips, and
// releases the context before returning.
class GpuRenderer
{
public:
    // sharedContext must be a hidden GLFW window created with the GUI's main
    // window passed as the share parameter. The GpuRenderer doesn't own it,
    // doesn't destroy it.
    GpuRenderer(int width, int height, float fov,
                int depth, int samples, int shadowSamples,
                GLFWwindow *sharedContext);
    ~GpuRenderer();

    // Same hook contract as the CPU Renderer.
    std::atomic<int> *progressRows = nullptr;
    std::atomic<bool> *cancelRequested = nullptr;
    std::function<void(const std::vector<Vec3f> &, int width, int height)> onPartialFrame;

    void render(const Scenes::SceneData &scene,
                std::chrono::steady_clock::time_point start,
                const std::string &outputDir);

    std::string lastOutputPath;

private:
    int _width, _height;
    float _fov;
    int _maxDepth, _samples, _shadowSamples;
    GLFWwindow *_sharedContext;

    // GL objects, lazily initialized on the worker thread the first time
    // render() is called (because GL calls require a current context).
    unsigned _program = 0;
    unsigned _outputTex = 0;
    unsigned _sphereSSBO = 0;
    unsigned _planeSSBO = 0;
    unsigned _materialSSBO = 0;
    bool _initialized = false;

    bool initGL();
    void uploadScene(const Scenes::SceneData &scene, int &outLightIdx);
    void destroyGL();
};
