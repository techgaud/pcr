// physically-cringe-renderer-v1.0.0
//
// Rough v1 GUI for pcr-cornell. Sliders for the same flags the CLI exposes,
// Render button kicks the existing path tracer on a worker thread, progress
// bar polls atomic row counter, and on completion the resulting PNG is
// loaded back from disk and displayed. Settings persist to JSON next to the
// executable.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if defined(__APPLE__)
#include <OpenGL/gl3.h>
#else
#include "imgui/backends/imgui_impl_opengl3_loader.h"
#endif

#include "json.hpp"
#include "lodepng.h"

#include "Includes/Renderer.h"
#include "Scenes/Scene.h"
#include "Scenes/Cornell.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// --- Settings persistence -------------------------------------------------

struct Settings
{
    int depth = 4;
    int samples = 16;
    int shadowSamples = 4;
    int width = 720;
    int height = 720;
    bool square = true;
    int sceneIndex = 0;     // index into the scene combo
    int timezoneIndex = 0;  // index into the tz combo
    std::string outputDir;
};

static const char *kTimezones[] = {"local", "EST", "CST", "MST", "PST", "UTC"};

static fs::path settingsPath()
{
    // Sit next to the executable. Falls back to cwd if argv[0] resolution fails.
    return fs::current_path() / "physically-cringe-renderer.json";
}

static void loadSettings(Settings &s)
{
    fs::path p = settingsPath();
    if (!fs::exists(p))
        return;
    try
    {
        std::ifstream in(p);
        json j;
        in >> j;
        s.depth = j.value("depth", s.depth);
        s.samples = j.value("samples", s.samples);
        s.shadowSamples = j.value("shadowSamples", s.shadowSamples);
        s.width = j.value("width", s.width);
        s.height = j.value("height", s.height);
        s.square = j.value("square", s.square);
        s.sceneIndex = j.value("sceneIndex", s.sceneIndex);
        s.timezoneIndex = j.value("timezoneIndex", s.timezoneIndex);
        s.outputDir = j.value("outputDir", s.outputDir);
    }
    catch (...)
    {
        // Corrupt settings file, ignore and keep defaults.
    }
}

static void saveSettings(const Settings &s)
{
    json j;
    j["depth"] = s.depth;
    j["samples"] = s.samples;
    j["shadowSamples"] = s.shadowSamples;
    j["width"] = s.width;
    j["height"] = s.height;
    j["square"] = s.square;
    j["sceneIndex"] = s.sceneIndex;
    j["timezoneIndex"] = s.timezoneIndex;
    j["outputDir"] = s.outputDir;
    std::ofstream out(settingsPath());
    out << j.dump(2);
}

// --- Timezone application (matches CLI behavior) -------------------------

static std::string resolveTimezone(const std::string &userTz)
{
    static const std::unordered_map<std::string, std::string> map = {
        {"EST", "EST5EDT"}, {"CST", "CST6CDT"}, {"MST", "MST7MDT"},
        {"PST", "PST8PDT"}, {"UTC", "UTC"},     {"GMT", "UTC"},
    };
    auto it = map.find(userTz);
    return it != map.end() ? it->second : userTz;
}

static void applyTimezone(const std::string &tz)
{
    if (tz.empty() || tz == "local")
        return;
    std::string resolved = resolveTimezone(tz);
#ifdef _WIN32
    _putenv_s("TZ", resolved.c_str());
#else
    setenv("TZ", resolved.c_str(), 1);
#endif
    tzset();
}

// --- Scene registry ------------------------------------------------------

using SceneFactory = std::function<Scenes::SceneData()>;
struct SceneEntry { const char *name; SceneFactory factory; };

static const std::vector<SceneEntry> &sceneRegistry()
{
    static const std::vector<SceneEntry> r = {
        {"cornell", &Scenes::makeCornell},
    };
    return r;
}

// --- Render job state (shared between GUI thread and worker) -------------

struct RenderJob
{
    std::atomic<bool> running{false};
    std::atomic<bool> cancelRequested{false};
    std::atomic<int> rowsCompleted{0};
    std::atomic<int> totalRows{0};
    std::thread worker;
    std::string finishedPath;   // GUI reads after running becomes false
    std::string errorMessage;
};

static void runRender(RenderJob *job, Settings settings)
{
    job->running = true;
    job->cancelRequested = false;
    job->rowsCompleted = 0;
    job->totalRows = settings.height;
    job->finishedPath.clear();
    job->errorMessage.clear();

    auto start = std::chrono::steady_clock::now();
    try
    {
        applyTimezone(kTimezones[settings.timezoneIndex]);

        const auto &reg = sceneRegistry();
        if (settings.sceneIndex < 0 || settings.sceneIndex >= (int)reg.size())
        {
            job->errorMessage = "Invalid scene selection";
            job->running = false;
            return;
        }
        Scenes::SceneData sceneData = reg[settings.sceneIndex].factory();

        std::string outDir = settings.outputDir;
        if (outDir.empty())
            outDir = (fs::current_path() / "Image").string();

        Renderer renderer{settings.width, settings.height, 65.f,
                          settings.depth, settings.samples, settings.shadowSamples};
        renderer.progressRows = &job->rowsCompleted;
        renderer.cancelRequested = &job->cancelRequested;
        renderer.render(sceneData, start, outDir);
        job->finishedPath = renderer.lastOutputPath;
    }
    catch (const std::exception &ex)
    {
        job->errorMessage = ex.what();
    }
    catch (...)
    {
        job->errorMessage = "Unknown exception during render";
    }
    job->running = false;
}

// --- Texture helpers (load PNG from disk into a GL texture) --------------

struct LoadedImage
{
    GLuint tex = 0;
    int width = 0;
    int height = 0;
};

static LoadedImage loadPng(const std::string &path)
{
    LoadedImage img;
    std::vector<unsigned char> pixels;
    unsigned w = 0, h = 0;
    unsigned err = lodepng::decode(pixels, w, h, path);
    if (err)
    {
        std::fprintf(stderr, "lodepng decode error %u: %s\n", err, lodepng_error_text(err));
        return img;
    }
    img.width = (int)w;
    img.height = (int)h;
    glGenTextures(1, &img.tex);
    glBindTexture(GL_TEXTURE_2D, img.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)w, (GLsizei)h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    return img;
}

static void freeImage(LoadedImage &img)
{
    if (img.tex)
    {
        glDeleteTextures(1, &img.tex);
        img.tex = 0;
    }
}

// --- Main ----------------------------------------------------------------

static void glfwErrorCallback(int err, const char *desc)
{
    std::fprintf(stderr, "GLFW error %d: %s\n", err, desc);
}

int main(int, char **)
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        return 1;

    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow *window = glfwCreateWindow(1100, 800, "physically cringe renderer", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    Settings settings;
    loadSettings(settings);

    RenderJob job;
    LoadedImage previewImg;
    std::string previewLoadedFrom;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // One full-window panel.
        ImGuiViewport *vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("pcr", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        // Scene picker.
        const auto &reg = sceneRegistry();
        if (ImGui::BeginCombo("Scene", reg[settings.sceneIndex].name))
        {
            for (int i = 0; i < (int)reg.size(); i++)
            {
                bool sel = (i == settings.sceneIndex);
                if (ImGui::Selectable(reg[i].name, sel))
                    settings.sceneIndex = i;
                if (sel)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::SeparatorText("Quality");
        ImGui::SliderInt("Depth", &settings.depth, 1, 8);
        ImGui::SliderInt("Samples", &settings.samples, 1, 512);
        ImGui::SliderInt("Shadow", &settings.shadowSamples, 1, 32);

        ImGui::SeparatorText("Output");
        ImGui::SliderInt("Width", &settings.width, 64, 2160);
        ImGui::Checkbox("Square (height matches width)", &settings.square);
        if (settings.square)
            settings.height = settings.width;
        ImGui::BeginDisabled(settings.square);
        ImGui::SliderInt("Height", &settings.height, 64, 2160);
        ImGui::EndDisabled();

        // Output dir input. ImGui needs a fixed-size buffer.
        char outBuf[1024];
        std::strncpy(outBuf, settings.outputDir.c_str(), sizeof(outBuf));
        outBuf[sizeof(outBuf) - 1] = 0;
        if (ImGui::InputText("Output dir (blank = ./Image)", outBuf, sizeof(outBuf)))
            settings.outputDir = outBuf;

        if (ImGui::BeginCombo("Timezone", kTimezones[settings.timezoneIndex]))
        {
            for (int i = 0; i < (int)(sizeof(kTimezones) / sizeof(kTimezones[0])); i++)
            {
                bool sel = (i == settings.timezoneIndex);
                if (ImGui::Selectable(kTimezones[i], sel))
                    settings.timezoneIndex = i;
                if (sel)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::SeparatorText("Render");

        bool isRunning = job.running.load();
        ImGui::BeginDisabled(isRunning);
        if (ImGui::Button("Render", ImVec2(120, 0)))
        {
            // Capture settings by value, kick worker.
            if (job.worker.joinable())
                job.worker.join();
            freeImage(previewImg);
            previewLoadedFrom.clear();
            job.worker = std::thread(runRender, &job, settings);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!isRunning);
        if (ImGui::Button("Cancel", ImVec2(120, 0)))
            job.cancelRequested = true;
        ImGui::EndDisabled();

        // Progress block.
        if (isRunning || job.totalRows.load() > 0)
        {
            int done = job.rowsCompleted.load();
            int total = job.totalRows.load();
            float frac = total > 0 ? (float)done / (float)total : 0.f;
            char overlay[64];
            std::snprintf(overlay, sizeof(overlay), "%d / %d rows", done, total);
            ImGui::ProgressBar(frac, ImVec2(-1, 0), overlay);
        }

        if (!job.errorMessage.empty())
        {
            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "Error: %s", job.errorMessage.c_str());
        }

        // Once worker has finished, load the resulting PNG into a texture.
        if (!isRunning && !job.finishedPath.empty() && job.finishedPath != previewLoadedFrom)
        {
            freeImage(previewImg);
            previewImg = loadPng(job.finishedPath);
            previewLoadedFrom = job.finishedPath;
        }

        if (previewImg.tex)
        {
            ImGui::SeparatorText("Result");
            ImGui::Text("%s", previewLoadedFrom.c_str());

            // Fit preview to remaining region while preserving aspect.
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float aspect = (float)previewImg.width / (float)previewImg.height;
            float w = avail.x;
            float h = w / aspect;
            if (h > avail.y) { h = avail.y; w = h * aspect; }
            ImGui::Image((ImTextureID)(intptr_t)previewImg.tex, ImVec2(w, h));
        }

        ImGui::End();

        // Render frame.
        ImGui::Render();
        int dispW, dispH;
        glfwGetFramebufferSize(window, &dispW, &dispH);
        glViewport(0, 0, dispW, dispH);
        glClearColor(0.1f, 0.1f, 0.12f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (job.worker.joinable())
    {
        job.cancelRequested = true;
        job.worker.join();
    }

    saveSettings(settings);

    freeImage(previewImg);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
