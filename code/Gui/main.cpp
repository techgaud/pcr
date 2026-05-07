// pcr GUI — shared source for both binaries.
//
// Same source is built twice with different compile-time defines:
//   frank-based-rendering         PCR_USE_GPU=0   (CPU path tracer)
//   physically-cringe-rendering   PCR_USE_GPU=1   (GPU path tracer, in progress)
//
// Sliders for the same flags the CLI exposes, Render button kicks the path
// tracer on a worker thread, progress bar polls atomic row counter, and on
// completion the resulting PNG is loaded back from disk and displayed.
// Settings persist to JSON named after the binary.

#ifndef PCR_BINARY_NAME
#define PCR_BINARY_NAME "physically-cringe-rendering"
#endif
#ifndef PCR_USE_GPU
#define PCR_USE_GPU 0
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
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

// The minimal OpenGL3 loader vendored with ImGui exposes only the constants
// it uses, which doesn't include these. They're stable spec values.
#ifndef GL_RGB
#define GL_RGB 0x1907
#endif
#ifndef GL_UNPACK_ALIGNMENT
#define GL_UNPACK_ALIGNMENT 0x0CF5
#endif

#include "json.hpp"
#include "lodepng.h"
#include "portable-file-dialogs.h"

#include "Includes/Renderer.h"
#include "Includes/Vec3f.h"
#include "Scenes/Scene.h"
#include "Scenes/SceneDiscovery.h"
#include "Scenes/SceneLoader.h"

#if PCR_USE_GPU
#include "Gpu/GpuRenderer.h"
using PCRRenderer = GpuRenderer;
#else
using PCRRenderer = Renderer;
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

// --- Settings persistence -------------------------------------------------

struct Preset
{
    std::string name;
    int depth = 4;
    int samples = 16;
    int shadowSamples = 4;
};

// Per-binary defaults. The GPU is fast enough that "Picture" can sensibly
// crank to 6 bounces / 2048 samples / 32 shadow rays — that takes ~4 min on
// a Threadripper-class GPU. The CPU "Picture" stays at 4/256/8 because the
// same settings on CPU would take many hours.
static std::vector<Preset> defaultPresets()
{
#if PCR_USE_GPU
    return {
        {"Quick",      2, 4,    2},
        {"Decent",     4, 16,   4},
        {"Production", 4, 256,  8},
        {"Picture",    6, 2048, 32},
    };
#else
    return {
        {"Quick",      2, 4,   2},
        {"Decent",     4, 16,  4},
        {"Production", 4, 64,  8},
        {"Picture",    4, 256, 8},
    };
#endif
}

struct Settings
{
    int depth = 4;
    int samples = 16;
    int shadowSamples = 4;
    int width = 720;
    int height = 720;
    bool square = true;
    // Scene by name, not by combo index, so reordering or adding scenes
    // doesn't silently rebind the user's last choice.
    std::string sceneName = "cornell";
    int timezoneIndex = 0;  // index into the tz combo
    std::string outputDir;
    bool darkTheme = true;
    bool useDenoise = false;
    bool useMIS = false;
    bool useRussian = false;
    bool useStratified = false;

    // Editable from the Edit Presets popup; persisted in <binary>.json.
    // Default to per-binary code constants when settings file is missing
    // or doesn't carry presets yet.
    std::vector<Preset> presets = defaultPresets();
};

static const char *kTimezones[] = {"local", "EST", "CST", "MST", "PST", "UTC"};

static fs::path settingsPath()
{
    // Sit next to the executable, named after the binary so the CPU and GPU
    // builds keep separate state.
    return fs::current_path() / (std::string(PCR_BINARY_NAME) + ".json");
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
        s.sceneName = j.value("sceneName", s.sceneName);
        s.timezoneIndex = j.value("timezoneIndex", s.timezoneIndex);
        s.outputDir = j.value("outputDir", s.outputDir);
        s.darkTheme = j.value("darkTheme", s.darkTheme);
        s.useDenoise   = j.value("useDenoise",   s.useDenoise);
        s.useMIS       = j.value("useMIS",       s.useMIS);
        s.useRussian   = j.value("useRussian",   s.useRussian);
        s.useStratified = j.value("useStratified", s.useStratified);
        if (j.contains("presets") && j["presets"].is_array() && !j["presets"].empty())
        {
            std::vector<Preset> loaded;
            for (const auto &pj : j["presets"])
            {
                if (!pj.is_object()) continue;
                Preset p;
                p.name = pj.value("name", std::string{});
                p.depth = pj.value("depth", 4);
                p.samples = pj.value("samples", 16);
                p.shadowSamples = pj.value("shadow", 4);
                if (p.name.empty()) continue;
                loaded.push_back(std::move(p));
            }
            if (!loaded.empty())
                s.presets = std::move(loaded);
        }
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
    j["sceneName"] = s.sceneName;
    j["timezoneIndex"] = s.timezoneIndex;
    j["outputDir"] = s.outputDir;
    j["darkTheme"] = s.darkTheme;
    j["useDenoise"]   = s.useDenoise;
    j["useMIS"]       = s.useMIS;
    j["useRussian"]   = s.useRussian;
    j["useStratified"] = s.useStratified;
    json arr = json::array();
    for (const auto &p : s.presets)
    {
        json pj;
        pj["name"] = p.name;
        pj["depth"] = p.depth;
        pj["samples"] = p.samples;
        pj["shadow"] = p.shadowSamples;
        arr.push_back(std::move(pj));
    }
    j["presets"] = std::move(arr);
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

// Tone-mapped 8-bit framebuffer snapshot for live preview, written by the
// renderer's snapshot thread and read by the GUI thread for texture upload.
struct LivePreview
{
    std::mutex mu;
    std::vector<unsigned char> rgb;
    int width = 0;
    int height = 0;
    std::atomic<bool> dirty{false};

    void reset()
    {
        std::lock_guard<std::mutex> lk(mu);
        rgb.clear();
        width = 0;
        height = 0;
        dirty.store(false);
    }
};

// gpuShared is the hidden GLFW window the GpuRenderer should make current
// on its worker thread. Created at startup with share=mainWindow so they can
// see each other's GL resources. Ignored by the CPU build.
static void runRender(RenderJob *job, LivePreview *live, Settings settings,
                      std::function<Scenes::SceneData()> sceneLoader,
                      GLFWwindow *gpuShared)
{
    job->running = true;
    job->cancelRequested = false;
    job->rowsCompleted = 0;
    job->totalRows = settings.height;
    job->finishedPath.clear();
    job->errorMessage.clear();
    if (live) live->reset();

    auto start = std::chrono::steady_clock::now();
    try
    {
        applyTimezone(kTimezones[settings.timezoneIndex]);

        if (!sceneLoader)
        {
            job->errorMessage = "No scene selected";
            job->running = false;
            return;
        }
        Scenes::SceneData sceneData = sceneLoader();

        std::string outDir = settings.outputDir;
        if (outDir.empty())
            outDir = (fs::current_path() / "Image").string();

#if PCR_USE_GPU
        PCRRenderer renderer{settings.width, settings.height,
                             settings.depth, settings.samples, settings.shadowSamples,
                             gpuShared};
#else
        (void)gpuShared;
        PCRRenderer renderer{settings.width, settings.height,
                             settings.depth, settings.samples, settings.shadowSamples};
#endif
        renderer.progressRows = &job->rowsCompleted;
        renderer.cancelRequested = &job->cancelRequested;
        renderer.useDenoise   = settings.useDenoise;
        renderer.useMIS       = settings.useMIS;
        renderer.useRussian   = settings.useRussian;
        renderer.useStratified = settings.useStratified;
        if (live)
        {
            renderer.onPartialFrame = [live](const std::vector<Vec3f> &fb, int w, int h) {
                std::lock_guard<std::mutex> lk(live->mu);
                live->width = w;
                live->height = h;
                live->rgb.resize((size_t)w * h * 3);
                for (size_t i = 0; i < (size_t)w * h; i++)
                {
                    Vec3f c = fb[i];
                    // Reinhard tone-map, matching what the final write does.
                    float r = c[0] / (c[0] + 1.f);
                    float g = c[1] / (c[1] + 1.f);
                    float b = c[2] / (c[2] + 1.f);
                    live->rgb[i * 3 + 0] = (unsigned char)(255 * r + 0.5f);
                    live->rgb[i * 3 + 1] = (unsigned char)(255 * g + 0.5f);
                    live->rgb[i * 3 + 2] = (unsigned char)(255 * b + 0.5f);
                }
                live->dirty.store(true, std::memory_order_release);
            };
        }
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

// --- PNG tEXt chunk reader ----------------------------------------------
//
// Returns the (key, value) pairs from any tEXt chunks in the PNG. Skips
// zTXt (compressed) and iTXt (international) — we only write tEXt from the
// renderer, so this is enough.
static std::vector<std::pair<std::string, std::string>>
readPngTextChunks(const std::string &path)
{
    std::vector<std::pair<std::string, std::string>> out;
    std::ifstream in(path, std::ios::binary);
    if (!in) return out;

    unsigned char sig[8];
    in.read((char *)sig, 8);
    static const unsigned char kPngSig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (std::memcmp(sig, kPngSig, 8) != 0) return out;

    while (in)
    {
        unsigned char hdr[8];
        if (!in.read((char *)hdr, 8)) break;
        unsigned int length = (hdr[0] << 24) | (hdr[1] << 16) | (hdr[2] << 8) | hdr[3];
        char type[5] = {(char)hdr[4], (char)hdr[5], (char)hdr[6], (char)hdr[7], 0};

        if (std::strcmp(type, "tEXt") == 0)
        {
            std::vector<char> buf(length);
            if (!in.read(buf.data(), length)) break;
            // Format: keyword\0text-string. Both ASCII.
            auto sep = std::find(buf.begin(), buf.end(), '\0');
            if (sep != buf.end())
            {
                std::string key(buf.begin(), sep);
                std::string val(sep + 1, buf.end());
                out.emplace_back(std::move(key), std::move(val));
            }
            in.seekg(4, std::ios::cur); // skip CRC
        }
        else if (std::strcmp(type, "IEND") == 0)
        {
            break;
        }
        else
        {
            in.seekg(length + 4, std::ios::cur); // skip data + CRC
        }
    }
    return out;
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

// --- Render-time estimator -----------------------------------------------
//
// Very rough. Models per-pixel work as linear in samples * depth (NOT
// samples^depth — branched paths terminate fast in practice via emissive
// hits, scene exits, and Russian roulette, so the worst-case-fanout model
// dramatically over-estimates at high sample counts). Per-target coefficient
// is calibrated from one measured point on the dev hardware.
//
// CPU calibration: d=4 s=16 S=4 720x720 took 241 sec on the homelab CPU.
//   cost = 4 * 16 * 4 * 720 * 720 = 1.33e8
//   coef = 241000 / 1.33e8 ~= 1.8e-3
//
// GPU calibration: d=6 s=2048 S=32 1080x1080 took 245 sec on Nate's desktop.
//   cost = 6 * 2048 * 32 * 1080 * 1080 = 4.59e11
//   coef = 245000 / 4.59e11 ~= 5.3e-7
//
// Expect 2-5x error in either direction. Useful for "minutes or hours"
// intuition, not for SLAs.
static double estimateRenderMs(int d, int s, int S, int w, int h)
{
#if PCR_USE_GPU
    constexpr double kPerCostUnit = 5.3e-7;
#else
    constexpr double kPerCostUnit = 1.8e-3;
#endif
    double cost = (double)s * (double)d * (double)S * (double)w * (double)h;
    return kPerCostUnit * cost;
}

// Render-duration formatter that doesn't UB on huge values. The previous
// version cast `ms / 3600000` straight to int, which on extreme estimates
// (e.g. samples^depth bug producing 1e19 ms) overflowed signed-int and
// printed nonsense like "3206175 hr 54 min" through saturated truncation.
// Use 64-bit ints for hour-and-up arithmetic and add a ceiling tier so
// estimates beyond a few months print "(very long)" instead of garbage.
static std::string formatDurationMs(double ms)
{
    if (ms < 0) ms = 0;
    if (ms < 1000) return "< 1 sec";

    constexpr double kMin = 60000.0;
    constexpr double kHour = 3600000.0;
    constexpr double kDay = 24.0 * kHour;

    char buf[96];
    if (ms < kMin) {
        std::snprintf(buf, sizeof(buf), "~%d sec", (int)(ms / 1000.0));
    } else if (ms < kHour) {
        int m = (int)(ms / kMin);
        int s = (int)((ms - m * kMin) / 1000.0);
        std::snprintf(buf, sizeof(buf), "~%d min %d sec", m, s);
    } else if (ms < kDay) {
        long long h = (long long)(ms / kHour);
        long long m = (long long)((ms - (double)h * kHour) / kMin);
        std::snprintf(buf, sizeof(buf), "~%lld hr %lld min", h, m);
    } else if (ms < 365.0 * kDay) {
        long long d = (long long)(ms / kDay);
        long long h = (long long)((ms - (double)d * kDay) / kHour);
        std::snprintf(buf, sizeof(buf), "~%lld days %lld hr", d, h);
    } else {
        std::snprintf(buf, sizeof(buf), "(very long — try fewer samples)");
    }
    return buf;
}

// --- Open path in OS file manager / image viewer ------------------------
//
// Best-effort. Uses xdg-open on Linux, open on macOS, ShellExecute on
// Windows. Failures are silent — the buttons are conveniences, not core.
static void openWithSystem(const std::string &path)
{
    if (path.empty()) return;
#if defined(_WIN32)
    std::string cmd = "start \"\" \"" + path + "\"";
    std::system(cmd.c_str());
#elif defined(__APPLE__)
    std::string cmd = "open \"" + path + "\"";
    std::system(cmd.c_str());
#else
    std::string cmd = "xdg-open \"" + path + "\" >/dev/null 2>&1 &";
    std::system(cmd.c_str());
#endif
}

// --- Slider helper -------------------------------------------------------
//
// Drop-in replacement for ImGui::SliderInt that adds an inline +/- input
// field for direct typing, and arrow-key handling when the slider is hovered.
// Right/Up = +step, Left/Down = -step. Shift accelerates to fastStep.
// Hold for OS-rate auto-repeat. Ctrl+Click on the slider still pops the
// built-in text edit (free from ImGui).
static bool pcrSliderInt(const char *label, int *v, int min, int max,
                         int step, int fastStep)
{
    bool changed = false;
    ImGui::PushID(label);

    float full = ImGui::CalcItemWidth();

    // Slider takes ~60% of the row, input field ~30% with the label after.
    ImGui::SetNextItemWidth(full * 0.6f);
    if (ImGui::SliderInt("##slider", v, min, max, "%d"))
        changed = true;
    bool sliderHovered = ImGui::IsItemHovered();

    ImGui::SameLine();
    ImGui::SetNextItemWidth(full * 0.3f);
    // InputInt has built-in +/- buttons that hold-repeat.
    if (ImGui::InputInt(label, v, step, fastStep))
        changed = true;

    // Arrow-key handling fires only while the slider is hovered, not the
    // input — when typing in the input field, arrows should move the cursor.
    if (sliderHovered)
    {
        int delta = 0;
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true) ||
            ImGui::IsKeyPressed(ImGuiKey_UpArrow, true))
            delta = +step;
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true) ||
            ImGui::IsKeyPressed(ImGuiKey_DownArrow, true))
            delta = -step;
        if (delta && (ImGui::IsKeyDown(ImGuiKey_LeftShift) ||
                      ImGui::IsKeyDown(ImGuiKey_RightShift)))
            delta = (delta > 0 ? +fastStep : -fastStep);
        if (delta)
        {
            *v += delta;
            changed = true;
        }
    }

    if (*v < min) *v = min;
    if (*v > max) *v = max;

    ImGui::PopID();
    return changed;
}

// --- Main ----------------------------------------------------------------

static void glfwErrorCallback(int err, const char *desc)
{
    std::fprintf(stderr, "GLFW error %d: %s\n", err, desc);
}

int main(int, char **)
{
#ifdef _WIN32
    // /SUBSYSTEM:WINDOWS detaches stderr/stdout from any console, so error
    // output from the renderer (shader compile failures, GL errors, GPU
    // readback errors, GLFW errors, exception text from the worker
    // thread's catch blocks) is silently discarded — particularly painful
    // when the GPU dies via TDR and there's no on-screen indication of
    // what happened. Redirect both streams to a log file next to the
    // current working directory so errors survive crashes.
    {
        std::string logPath = std::string(PCR_BINARY_NAME) + ".log";
        // Truncate on launch; a stale log from a previous run shouldn't
        // confuse the next debugging session.
        std::freopen(logPath.c_str(), "w", stderr);
        std::freopen(logPath.c_str(), "a", stdout);
        std::fprintf(stderr, "%s starting (build " __DATE__ " " __TIME__ ")\n",
                     PCR_BINARY_NAME);
        std::fflush(stderr);
    }
#endif

    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        return 1;

    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow *window = glfwCreateWindow(1100, 800, PCR_BINARY_NAME, nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Hidden second window with shared GL context. The GpuRenderer worker
    // thread makes this current while it dispatches compute shaders, so the
    // GUI thread can keep drawing on `window` independently. CPU build
    // creates it too (cheap) and just doesn't use it.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *gpuShared = glfwCreateWindow(1, 1, "pcr-gpu-shared", nullptr, window);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    if (!gpuShared)
        std::fprintf(stderr, "warning: could not create shared GL context; GPU renderer will fail\n");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    Settings settings;
    loadSettings(settings);

    // Discovered scenes are cached and refreshed on demand. The GUI rescans
    // every time the user opens the scene combo so JSON files dropped into
    // Scenes/ at runtime show up without restarting.
    std::vector<Scenes::DiscoveredScene> sceneRegistry;
    std::string sceneScanError;
    auto rescanScenes = [&]() {
        sceneScanError.clear();
        sceneRegistry = Scenes::discoverScenes({}, [&](const std::string &msg) {
            if (!sceneScanError.empty()) sceneScanError += "\n";
            sceneScanError += msg;
        });
    };
    rescanScenes();

    // Find the index for the scene the settings pointed at; fall back to the
    // first scene in the registry if the saved name no longer exists.
    auto findSceneIndex = [&](const std::string &name) -> int {
        for (int i = 0; i < (int)sceneRegistry.size(); i++)
            if (sceneRegistry[i].name == name) return i;
        return sceneRegistry.empty() ? -1 : 0;
    };

    auto applyTheme = [](bool dark) {
        if (dark) ImGui::StyleColorsDark();
        else      ImGui::StyleColorsLight();
    };
    applyTheme(settings.darkTheme);

    RenderJob job;
    LivePreview live;
    GLuint liveTex = 0;
    int liveTexW = 0, liveTexH = 0;
    LoadedImage previewImg;
    std::string previewLoadedFrom;
    std::vector<std::pair<std::string, std::string>> previewMetadata;

    struct HistoryEntry
    {
        Settings settings;
        std::string path;
        std::string label;
    };
    std::deque<HistoryEntry> history;
    constexpr size_t kHistoryMax = 10;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // If the render's snapshot thread has produced a new framebuffer
        // since last frame, upload it to the live-preview texture.
        if (live.dirty.load(std::memory_order_acquire))
        {
            std::lock_guard<std::mutex> lk(live.mu);
            if (live.width > 0 && live.height > 0 && !live.rgb.empty())
            {
                if (liveTex == 0)
                {
                    glGenTextures(1, &liveTex);
                    glBindTexture(GL_TEXTURE_2D, liveTex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                }
                glBindTexture(GL_TEXTURE_2D, liveTex);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, live.width, live.height, 0,
                             GL_RGB, GL_UNSIGNED_BYTE, live.rgb.data());
                liveTexW = live.width;
                liveTexH = live.height;
            }
            live.dirty.store(false, std::memory_order_relaxed);
        }

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

        // Top-right Theme + About buttons.
        {
            const float aboutW = 70.f;
            const float themeW = 70.f;
            const float gap = ImGui::GetStyle().ItemSpacing.x;
            float windowW = ImGui::GetWindowSize().x;
            float padX = ImGui::GetStyle().WindowPadding.x;
            ImGui::SetCursorPosX(windowW - aboutW - themeW - gap - padX);
            if (ImGui::Button(settings.darkTheme ? "Light" : "Dark", ImVec2(themeW, 0)))
            {
                settings.darkTheme = !settings.darkTheme;
                applyTheme(settings.darkTheme);
            }
            ImGui::SameLine();
            if (ImGui::Button("About", ImVec2(aboutW, 0)))
                ImGui::OpenPopup("About##popup");
        }

        if (ImGui::BeginPopupModal("About##popup", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::Text("%s", PCR_BINARY_NAME);
#if PCR_USE_GPU
            ImGui::TextDisabled("GPU path tracer (OpenGL 4.3 compute shaders)");
#else
            ImGui::TextDisabled("CPU path tracer (multi-threaded)");
#endif
            ImGui::Separator();
            ImGui::TextWrapped("Cornell Box scene path tracer.");
            ImGui::Spacing();
            ImGui::Text("Source: github.com/techgaud/pcr");
            ImGui::Spacing();
            ImGui::SeparatorText("Vendored libraries");
            ImGui::BulletText("Dear ImGui v1.91.5 (MIT)");
            ImGui::BulletText("GLFW 3.4 (zlib)");
            ImGui::BulletText("lodepng v20260119 (zlib)");
            ImGui::BulletText("nlohmann json v3.11.3 (MIT)");
            ImGui::BulletText("portable-file-dialogs (WTFPL)");
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(120, 0)))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // Scene picker. Rescan every time the user opens the combo so a JSON
        // file dropped into Scenes/ shows up without an app restart.
        int curSceneIdx = findSceneIndex(settings.sceneName);
        const char *curSceneLabel = curSceneIdx >= 0
            ? sceneRegistry[curSceneIdx].name.c_str()
            : "(none)";
        bool comboOpen = ImGui::BeginCombo("Scene", curSceneLabel);
        // BeginCombo returns true the frame the popup is open, but we only
        // want to rescan once when it transitions closed -> open.
        static bool wasComboOpen = false;
        if (comboOpen && !wasComboOpen)
            rescanScenes();
        wasComboOpen = comboOpen;
        if (comboOpen)
        {
            for (int i = 0; i < (int)sceneRegistry.size(); i++)
            {
                const auto &s = sceneRegistry[i];
                bool sel = (i == curSceneIdx);
                if (ImGui::Selectable(s.name.c_str(), sel))
                    settings.sceneName = s.name;
                if (sel)
                    ImGui::SetItemDefaultFocus();
                if (ImGui::IsItemHovered())
                {
                    if (s.source == Scenes::DiscoveredScene::Source::Hardcoded)
                        ImGui::SetTooltip("v%s — built into binary", s.version.c_str());
                    else
                        ImGui::SetTooltip("v%s — %s", s.version.c_str(), s.filePath.c_str());
                }
            }
            ImGui::EndCombo();
        }
        if (!sceneScanError.empty())
            ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "Scene scan: %s", sceneScanError.c_str());

        ImGui::SeparatorText("Quality");

        // Preset buttons snap depth/samples/shadow. Width/height are
        // intentionally not touched so resolution is independent of preset.
        // Presets are editable via the Edit button, persisted in settings.
        for (size_t i = 0; i < settings.presets.size(); i++)
        {
            if (i > 0) ImGui::SameLine();
            const auto &p = settings.presets[i];
            if (ImGui::Button(p.name.c_str()))
            {
                settings.depth = p.depth;
                settings.samples = p.samples;
                settings.shadowSamples = p.shadowSamples;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("d=%d s=%d S=%d", p.depth, p.samples, p.shadowSamples);
        }
        ImGui::SameLine();
        if (ImGui::Button("Edit##presets"))
            ImGui::OpenPopup("Edit Presets##popup");

        if (ImGui::BeginPopupModal("Edit Presets##popup", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::TextUnformatted("Customize the quality preset buttons. Saved next to the binary.");
            ImGui::Separator();
            if (ImGui::BeginTable("presets-edit", 5,
                                  ImGuiTableFlags_SizingStretchProp))
            {
                ImGui::TableSetupColumn("Name");
                ImGui::TableSetupColumn("Depth");
                ImGui::TableSetupColumn("Samples");
                ImGui::TableSetupColumn("Shadow");
                ImGui::TableSetupColumn("Reset");
                ImGui::TableHeadersRow();

                auto defaults = defaultPresets();
                for (size_t i = 0; i < settings.presets.size(); i++)
                {
                    auto &p = settings.presets[i];
                    ImGui::PushID((int)i);
                    ImGui::TableNextRow();

                    ImGui::TableNextColumn();
                    char nameBuf[64];
                    std::strncpy(nameBuf, p.name.c_str(), sizeof(nameBuf));
                    nameBuf[sizeof(nameBuf) - 1] = 0;
                    ImGui::SetNextItemWidth(140);
                    if (ImGui::InputText("##name", nameBuf, sizeof(nameBuf)))
                        p.name = nameBuf;

                    ImGui::TableNextColumn();
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("##d", &p.depth);
                    if (p.depth < 1) p.depth = 1;
                    if (p.depth > 8) p.depth = 8;

                    ImGui::TableNextColumn();
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("##s", &p.samples);
                    if (p.samples < 1) p.samples = 1;
                    if (p.samples > 4096) p.samples = 4096;

                    ImGui::TableNextColumn();
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("##S", &p.shadowSamples);
                    if (p.shadowSamples < 1) p.shadowSamples = 1;
                    if (p.shadowSamples > 64) p.shadowSamples = 64;

                    ImGui::TableNextColumn();
                    if (i < defaults.size() && ImGui::SmallButton("Default"))
                        p = defaults[i];

                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
            ImGui::Spacing();
            if (ImGui::Button("Reset all to defaults"))
                settings.presets = defaultPresets();
            ImGui::SameLine();
            if (ImGui::Button("Close", ImVec2(120, 0)))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        pcrSliderInt("Depth",   &settings.depth,         1, 8,    1, 2);
        pcrSliderInt("Samples", &settings.samples,       1, 4096, 1, 16);
        pcrSliderInt("Shadow",  &settings.shadowSamples, 1, 64,   1, 4);

        ImGui::SeparatorText("Techniques");
        ImGui::Checkbox("Denoise (5x5 cross-bilateral on output)", &settings.useDenoise);
        ImGui::Checkbox("MIS (light-side balance heuristic, partial)", &settings.useMIS);
        ImGui::Checkbox("Russian roulette (terminate paths at depth >= 1)", &settings.useRussian);
        ImGui::Checkbox("Stratified samples (jittered grid first bounce)", &settings.useStratified);

        ImGui::SeparatorText("Output");
        pcrSliderInt("Width", &settings.width, 64, 2160, 8, 64);
        ImGui::Checkbox("Square (height matches width)", &settings.square);
        if (settings.square)
            settings.height = settings.width;
        ImGui::BeginDisabled(settings.square);
        pcrSliderInt("Height", &settings.height, 64, 2160, 8, 64);
        ImGui::EndDisabled();

        // Output dir input + Browse button. ImGui needs a fixed-size buffer.
        char outBuf[1024];
        std::strncpy(outBuf, settings.outputDir.c_str(), sizeof(outBuf));
        outBuf[sizeof(outBuf) - 1] = 0;
        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() * 0.75f);
        if (ImGui::InputText("##outdir", outBuf, sizeof(outBuf)))
            settings.outputDir = outBuf;
        ImGui::SameLine();
        if (ImGui::Button("Browse..."))
        {
            std::string startDir = settings.outputDir.empty()
                ? fs::current_path().string()
                : settings.outputDir;
            auto result = pfd::select_folder("Pick output directory", startDir).result();
            if (!result.empty())
                settings.outputDir = result;
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("Output dir (blank = ./Image)");

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

        // Estimate label updates live as sliders move.
        {
            double estMs = estimateRenderMs(settings.depth, settings.samples,
                                            settings.shadowSamples,
                                            settings.width, settings.height);
            ImGui::TextDisabled("Estimate: %s  (very rough; expect 2-5x error)",
                                formatDurationMs(estMs).c_str());
        }

        bool isRunning = job.running.load();
        // Snapshot the loader closure now so the worker uses what was
        // selected at click-time, not whatever the user picks during render.
        std::function<Scenes::SceneData()> sceneLoader;
        if (curSceneIdx >= 0)
            sceneLoader = sceneRegistry[curSceneIdx].load;
        ImGui::BeginDisabled(isRunning || !sceneLoader);
        if (ImGui::Button("Render", ImVec2(120, 0)))
        {
            // Capture settings by value, kick worker.
            if (job.worker.joinable())
                job.worker.join();
            freeImage(previewImg);
            previewLoadedFrom.clear();
            job.worker = std::thread(runRender, &job, &live, settings,
                                     sceneLoader, gpuShared);
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

        // Once worker has finished, load the resulting PNG into a texture
        // and append to history.
        if (!isRunning && !job.finishedPath.empty() && job.finishedPath != previewLoadedFrom)
        {
            freeImage(previewImg);
            previewImg = loadPng(job.finishedPath);
            previewLoadedFrom = job.finishedPath;
            previewMetadata = readPngTextChunks(job.finishedPath);

            HistoryEntry entry;
            entry.settings = settings;
            entry.path = job.finishedPath;
            // Short label for the list. Use the filename's stem so the
            // panel doesn't get overrun by long absolute paths.
            entry.label = fs::path(job.finishedPath).filename().string();
            history.push_front(std::move(entry));
            while (history.size() > kHistoryMax)
                history.pop_back();
        }

        // Live preview while rendering: show the in-progress framebuffer.
        if (isRunning && liveTex && liveTexW > 0 && liveTexH > 0)
        {
            ImGui::SeparatorText("Live preview (rendering...)");
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float aspect = (float)liveTexW / (float)liveTexH;
            float w = avail.x;
            float h = w / aspect;
            if (h > avail.y) { h = avail.y; w = h * aspect; }
            ImGui::Image((ImTextureID)(intptr_t)liveTex, ImVec2(w, h));
        }

        if (previewImg.tex)
        {
            ImGui::SeparatorText("Result");
            ImGui::Text("%s", previewLoadedFrom.c_str());

            if (ImGui::Button("Open folder"))
            {
                std::string dir = settings.outputDir.empty()
                    ? (fs::current_path() / "Image").string()
                    : settings.outputDir;
                openWithSystem(dir);
            }
            ImGui::SameLine();
            if (ImGui::Button("Open render"))
                openWithSystem(previewLoadedFrom);
            ImGui::SameLine();
            if (ImGui::Button("Render again") && !job.running.load())
            {
                if (job.worker.joinable())
                    job.worker.join();
                freeImage(previewImg);
                previewLoadedFrom.clear();
                job.worker = std::thread(runRender, &job, &live, settings,
                                         sceneLoader, gpuShared);
            }

            if (!previewMetadata.empty() && ImGui::TreeNode("Metadata (PNG tEXt chunks)"))
            {
                if (ImGui::BeginTable("md", 2,
                                      ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
                {
                    for (const auto &kv : previewMetadata)
                    {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(kv.first.c_str());
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(kv.second.c_str());
                    }
                    ImGui::EndTable();
                }
                ImGui::TreePop();
            }

            // Fit preview to remaining region while preserving aspect.
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float aspect = (float)previewImg.width / (float)previewImg.height;
            float w = avail.x;
            float h = w / aspect;
            if (h > avail.y) { h = avail.y; w = h * aspect; }
            ImGui::Image((ImTextureID)(intptr_t)previewImg.tex, ImVec2(w, h));
        }

        if (!history.empty() && ImGui::CollapsingHeader("History (most recent first)"))
        {
            // Each click restores that render's settings into the live sliders
            // and reloads the PNG as the current preview.
            for (size_t i = 0; i < history.size(); i++)
            {
                ImGui::PushID((int)i);
                if (ImGui::Selectable(history[i].label.c_str()))
                {
                    settings = history[i].settings;
                    if (fs::exists(history[i].path))
                    {
                        freeImage(previewImg);
                        previewImg = loadPng(history[i].path);
                        previewLoadedFrom = history[i].path;
                        previewMetadata = readPngTextChunks(history[i].path);
                    }
                }
                ImGui::PopID();
            }
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
    if (liveTex) glDeleteTextures(1, &liveTex);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
