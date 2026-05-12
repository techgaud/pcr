// pcr GUI. shared source for both binaries.
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

#ifdef _WIN32
// CMake's pcr_apply_msvc_release_flags target already defines NOMINMAX and
// WIN32_LEAN_AND_MEAN for the GUI binaries, so this include is safe to pull
// in for the AllocConsole / pipe / fan-out plumbing in main().
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

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

#include "Includes/DllSearch.h"
#include "Includes/GpuDefaults.h"
#include "Includes/LutDiscovery.h"
#include "Includes/RGBToSpectrum.h"
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

// --- Job queue (ephemeral, in-session only) -----------------------------

// Per-render slice of Settings, captured at "Queue" click time. Global UI
// state (output dir, theme, presets list, etc.) is NOT part of a JobConfig
// since it doesn't make sense to vary per job in a batch.
struct JobConfig
{
    std::string sceneName;
    int  depth = 4;
    int  samples = 16;
    int  shadowSamples = 4;
    int  width = 720;
    int  height = 720;
    bool square = true;
    bool useDenoise = false;
    bool useMIS = false;
    // BSDF-side MIS, wavefront-only extension. Off by default;
    // requires useMIS to be on and useWavefront true to have any effect.
    bool useBsdfMis = false;
    bool useRussian = false;
    bool useStratified = false;
    bool useACES = false;
    bool useSpectral = false;
    // false = Wyman 2013 piecewise-Gaussian (default, ~1% off CIE).
    // true = CIE 1931 tabulated. Spectral renders only.
    bool useCieCmf = false;
    bool useAA = false;
    int  aaSamples = 4;
    bool useAdaptive = false;
    bool useOIDN = false;
    int  heroSamples = 4;
    std::string lutChoice = "off";
    int  threadgroupX = pcr::kDefaultThreadgroupX;
    int  threadgroupY = pcr::kDefaultThreadgroupY;

    // Architecture toggle. Defaults centralized in GpuDefaults.h (true /
    // false as of v1.5.0 = wavefront-1spp default, picked by A/B). GUI
    // exposes the radio under Architecture (debug) for opt-out / mode
    // switching; CLI exposes --no-wavefront for the same.
    bool useWavefront = pcr::kDefaultUseWavefront;
    bool wavefrontMultiSample = pcr::kDefaultWavefrontMultiSample;
    bool spectralFork = pcr::kDefaultSpectralFork;
};

struct JobResult
{
    enum Status { Pending, Running, Done, Failed };
    JobConfig config;
    Status status = Pending;
    std::string outputPath;
    std::string errorMessage;
};

enum BatchPhase
{
    BP_INACTIVE,     // no batch running
    BP_RUNNING_JOB,  // batch is active, current job's worker is running
    BP_ADVANCING,    // job just finished, need to record + start next or wrap up
};

// --- Settings persistence -------------------------------------------------

struct Preset
{
    std::string name;
    int depth = 4;
    int samples = 16;
    int shadowSamples = 4;

    // Optional snaps. When clicked, a preset always applies depth /
    // samples / shadow above; the fields below are applied only if
    // non-default sentinel values, so most presets stay minimal and
    // only "Picture" pulls its weight as a one-click hero render
    // configuration.
    //
    // width/height: 0 = leave settings.width / settings.height alone.
    // technique flags: -1 = leave unchanged, 0 = off, 1 = on.
    int width = 0;
    int height = 0;
    int snapSquare = -1;
    int useDenoise = -1;
    int useMIS = -1;
    int useRussian = -1;
    int useStratified = -1;
    int useAA = -1;
    int useAdaptive = -1;
    int useOIDN = -1;
};

// Per-binary defaults. The GPU is fast enough that "Picture" can sensibly
// crank to 6 bounces / 2048 samples / 32 shadow rays AND turn on every
// quality technique AND snap to 1080 square - the post-Metal-port M1
// Ultra finishes that hero configuration in minutes, not hours. CPU
// "Picture" stays at 4/256/8 with no extra snaps because the same
// settings on CPU would take many hours.
static std::vector<Preset> defaultPresets()
{
#if PCR_USE_GPU
    Preset picture;
    picture.name = "Picture";
    picture.depth = 6;
    picture.samples = 2048;
    picture.shadowSamples = 32;
    picture.width = 1080;
    picture.height = 1080;
    picture.snapSquare = 1;
    picture.useDenoise = 1;
    picture.useMIS = 1;
    picture.useRussian = 1;
    picture.useStratified = 1;
    picture.useAA = 1;
    // Adaptive deliberately OFF: at 2048 hero samples the convergence
    // early-exit doesn't save meaningful work, and turning it on routes
    // the Metal renderer to its strip path instead of the saturation-
    // friendly multi-pass kernel (~2x slower on Picture-class
    // workloads). User can flip it on manually if they want the older
    // path.
    picture.useAdaptive = 0;
    picture.useOIDN = 1;

    // Production on GPU: the daily-driver hero render. Same one-click
    // snap surface as Picture (resolution, square, every quality
    // technique except adaptive sampling) but at d=4 / s=1024 / S=16
    // instead of Picture's 6/2048/32. Converges fast enough on M1
    // Ultra that it's the right starting point for most renders, with
    // Picture reserved for the slower max-quality runs.
    Preset production;
    production.name = "Production";
    production.depth = 4;
    production.samples = 1024;
    production.shadowSamples = 16;
    production.width = 1080;
    production.height = 1080;
    production.snapSquare = 1;
    production.useDenoise = 1;
    production.useMIS = 1;
    production.useRussian = 1;
    production.useStratified = 1;
    production.useAA = 1;
    production.useAdaptive = 0;
    production.useOIDN = 1;

    return {
        {"Quick",      2, 4,    2},
        {"Decent",     4, 16,   4},
        production,
        picture,
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
    // Mode toggles. ACES and Spectral are presented as radio choices
    // in the GUI's Mode section since they pick between two
    // alternatives (Reinhard vs ACES tone-map; RGB vs Spectral
    // pipeline) rather than turning a noise-reduction technique on
    // or off. Stored as booleans because there are only two options
    // for each axis right now.
    bool useACES = false;
    bool useSpectral = false;
    // false = Wyman 2013 (default), true = CIE 1931 tabulated.
    // Spectral renders only.
    bool useCieCmf = false;
    // BSDF-side MIS, wavefront-only. Persisted alongside useMIS.
    bool useBsdfMis = false;
    bool useAA = false;
    int aaSamples = 4;
    bool useAdaptive = false;
    bool useOIDN = false;

    // Hero wavelength sample count. Default 4 = stratified hero
    // (Wilkie 2014). 1 = legacy single-wavelength path, only meaningful
    // for benchmarking and visual A/B against the hero default. The
    // GUI control only appears in debug mode (when settings.debugMode
    // is on AND settings.useSpectral is on). The setting persists
    // regardless so toggling debug off doesn't silently revert.
    int heroSamples = 4;

    // Metal compute threadgroup shape. Effective on Apple Silicon
    // backends only; OpenGL ignores them. Default lives in
    // GpuDefaults.h (8x8 as of v1.5.0, picked by A/B). The GUI's
    // debug-mode tuning panel surfaces a row of preset buttons for
    // re-running the A/B if the kernel shape changes.
    int threadgroupX = pcr::kDefaultThreadgroupX;
    int threadgroupY = pcr::kDefaultThreadgroupY;

    // Architecture toggle. false = megakernel, true = wavefront. Persisted
    // even though the GUI doesn't surface a switch yet so future flips of
    // the GUI radio don't lose state across launches.
    bool useWavefront = pcr::kDefaultUseWavefront;
    bool wavefrontMultiSample = pcr::kDefaultWavefrontMultiSample;
    bool spectralFork = pcr::kDefaultSpectralFork;

    // LUT for spectral RGB-to-spectrum upsampling. The control only
    // appears in the GUI when useSpectral == true; its setting is
    // persisted regardless so flipping spectral on later restores the
    // user's last choice.
    //
    // Values:
    //   "off"       - runtime homotopy, no LUT (default; no startup cost)
    //   "build"     - build LUT in-process at scene-load (~4 sec)
    //   "<name>"    - load luts/<name>.lut from disk (milliseconds)
    //
    // Anything not matching one of those falls back to "off". A name
    // that points at a file that's gone since the choice was saved
    // also falls back to "off" with a console warning at render time.
    std::string lutChoice = "off";

    // Pop a debug console + log file at startup. Toggled by the Debug
    // button in the GUI top-right (or the PCR_DEBUG env var). Persisted
    // so it survives across launches.
    bool debugMode = false;

    // Editable from the Edit Presets popup; persisted in <binary>.json.
    // Default to per-binary code constants when settings file is missing
    // or doesn't carry presets yet.
    std::vector<Preset> presets = defaultPresets();
};

// Snapshot the render-relevant subset of Settings into a JobConfig.
static JobConfig makeJobConfig(const Settings &s)
{
    JobConfig j;
    j.sceneName     = s.sceneName;
    j.depth         = s.depth;
    j.samples       = s.samples;
    j.shadowSamples = s.shadowSamples;
    j.width         = s.width;
    j.height        = s.height;
    j.square        = s.square;
    j.useDenoise    = s.useDenoise;
    j.useMIS        = s.useMIS;
    j.useBsdfMis    = s.useBsdfMis;
    j.useRussian    = s.useRussian;
    j.useStratified = s.useStratified;
    j.useACES       = s.useACES;
    j.useSpectral   = s.useSpectral;
    j.useCieCmf     = s.useCieCmf;
    j.useAA         = s.useAA;
    j.aaSamples     = s.aaSamples;
    j.useAdaptive   = s.useAdaptive;
    j.useOIDN       = s.useOIDN;
    j.heroSamples   = s.heroSamples;
    j.lutChoice     = s.lutChoice;
    j.threadgroupX  = s.threadgroupX;
    j.threadgroupY  = s.threadgroupY;
    j.useWavefront  = s.useWavefront;
    j.wavefrontMultiSample = s.wavefrontMultiSample;
    j.spectralFork  = s.spectralFork;
    return j;
}

// Apply a JobConfig over an existing Settings (preserves global UI state
// like outputDir, theme, presets list, timezone).
static void applyJobConfig(const JobConfig &j, Settings &s)
{
    s.sceneName     = j.sceneName;
    s.depth         = j.depth;
    s.samples       = j.samples;
    s.shadowSamples = j.shadowSamples;
    s.width         = j.width;
    s.height        = j.height;
    s.square        = j.square;
    s.useDenoise    = j.useDenoise;
    s.useMIS        = j.useMIS;
    s.useBsdfMis    = j.useBsdfMis;
    s.useRussian    = j.useRussian;
    s.useStratified = j.useStratified;
    s.useACES       = j.useACES;
    s.useSpectral   = j.useSpectral;
    s.useCieCmf     = j.useCieCmf;
    s.useAA         = j.useAA;
    s.aaSamples     = j.aaSamples;
    s.useAdaptive   = j.useAdaptive;
    s.useOIDN       = j.useOIDN;
    s.heroSamples   = j.heroSamples;
    s.lutChoice     = j.lutChoice;
    s.threadgroupX  = j.threadgroupX;
    s.threadgroupY  = j.threadgroupY;
    s.useWavefront  = j.useWavefront;
    s.wavefrontMultiSample = j.wavefrontMultiSample;
    s.spectralFork  = j.spectralFork;
}

// Returns nullptr if the job will run with its requested architecture,
// or a short human-readable reason if the renderer will silently
// rewrite the architecture at dispatch time (falling back to
// megakernel). The queue UI uses this to color the displaced wavefront
// tag light red and to print a "-> megakernel (reason)" hint below.
//
// Current fallback rules in MetalRenderer (kept in sync here):
//   - useWavefront + useAdaptive + multi-sample-per-pass: adaptive
//     works in 1spp wavefront but the multi-spp writeback's
//     per-pass-multi-sample reduction isn't ported yet.
static const char *wavefrontFallbackReason(const JobConfig &j)
{
    if (!j.useWavefront) return nullptr;
    if (j.useAdaptive && j.wavefrontMultiSample)
        return "adaptive + multi-spp";
    return nullptr;
}

static bool wavefrontWillFallback(const JobConfig &j)
{
    return wavefrontFallbackReason(j) != nullptr;
}

// Render one queue row's summary inline. Each tag becomes its own
// ImGui::Text so we can color individual tags differently (light red
// for settings that get overridden at dispatch time).
static void renderJobSummary(const JobConfig &j)
{
    char prefix[128];
    std::snprintf(prefix, sizeof(prefix), "%s  %dx%d  d%d/s%d/S%d",
                  j.sceneName.c_str(), j.width, j.height,
                  j.depth, j.samples, j.shadowSamples);
    ImGui::TextUnformatted(prefix);

    auto tag = [](const char *label) {
        ImGui::SameLine();
        ImGui::TextUnformatted(label);
    };
    auto tagRed = [](const char *label) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.55f, 1.0f), "%s", label);
    };

    if (j.useSpectral)   tag("spec");
    // CMF only meaningful in spectral renders. Wyman is the default;
    // tag the cie alternative so the queue row makes the choice
    // visible at a glance. Same convention as the wavefront fork/
    // terminate suffix below.
    if (j.useSpectral && j.useCieCmf) tag("cie");
    if (j.useAA) {
        char aa[16]; std::snprintf(aa, sizeof(aa), "aa%d", j.aaSamples);
        tag(aa);
    }
    if (j.useAdaptive)   tag("adapt");
    if (j.useOIDN)       tag("oidn");
    if (j.useACES)       tag("aces");
    if (j.useDenoise && !j.useOIDN) tag("denoise");
    if (j.useMIS)        tag("mis");
    // BSDF-side MIS only fires when light-side MIS AND wavefront are
    // both on; tag only when the combo is active so the queue row
    // doesn't mislead about silently-ignored flags.
    if (j.useMIS && j.useWavefront && j.useBsdfMis) tag("mis-bsdf");
    if (j.useRussian)    tag("rr");
    if (j.useStratified) tag("strat");
    if (j.useWavefront) {
        // Append "-terminate" to the wavefront tag when the cheaper
        // terminate strategy is selected in spectral mode (fork is the
        // default since the post-A/B flip; plain "wave-1spp" / "wave-
        // mspp" implies fork). Tag only shows in spectral renders --
        // the setting is ignored in RGB.
        char waveBuf[28];
        const char *waveBase = j.wavefrontMultiSample ? "wave-mspp" : "wave-1spp";
        if (j.useSpectral && !j.spectralFork)
            std::snprintf(waveBuf, sizeof(waveBuf), "%s-terminate", waveBase);
        else
            std::snprintf(waveBuf, sizeof(waveBuf), "%s", waveBase);
        if (wavefrontWillFallback(j)) tagRed(waveBuf);
        else                          tag(waveBuf);
    }
    char tgInfo[32];
    std::snprintf(tgInfo, sizeof(tgInfo), "tg%dx%d", j.threadgroupX, j.threadgroupY);
    tag(tgInfo);
}

// Legacy single-line string version of the summary, kept for callers
// that want a flat string (e.g. logging, PNG metadata). The GUI queue
// row uses renderJobSummary() above instead so it can color individual
// tags.
static std::string summarizeJob(const JobConfig &j)
{
    char buf[256];
    std::string tags;
    if (j.useSpectral)   tags += " spec";
    if (j.useSpectral && j.useCieCmf) tags += " cie";
    if (j.useAA)         { char t[16]; std::snprintf(t, sizeof(t), " aa%d", j.aaSamples); tags += t; }
    if (j.useAdaptive)   tags += " adapt";
    if (j.useOIDN)       tags += " oidn";
    if (j.useACES)       tags += " aces";
    if (j.useDenoise && !j.useOIDN) tags += " denoise";
    if (j.useMIS)        tags += " mis";
    if (j.useMIS && j.useWavefront && j.useBsdfMis) tags += " mis-bsdf";
    if (j.useRussian)    tags += " rr";
    if (j.useStratified) tags += " strat";
    if (j.useWavefront) {
        tags += j.wavefrontMultiSample ? " wave-mspp" : " wave-1spp";
        if (j.useSpectral && !j.spectralFork) tags += "-terminate";
    }
    std::snprintf(buf, sizeof(buf),
                  "%s  %dx%d  d%d/s%d/S%d%s  tg%dx%d",
                  j.sceneName.c_str(), j.width, j.height,
                  j.depth, j.samples, j.shadowSamples,
                  tags.c_str(),
                  j.threadgroupX, j.threadgroupY);
    return std::string(buf);
}

static const char *kTimezones[] = {"local", "EST", "CST", "MST", "PST", "UTC"};

#ifdef _WIN32
// Forward decls for the activity-log helpers below. runRender (further up
// in the file than the helpers themselves) calls them inside #ifdef _WIN32
// blocks. Linux/GCC builds skip those blocks entirely, but MSVC compiles
// them and needs visible declarations before the call sites.
static void setActivity(const std::string &what);
static void clearActivity();
#endif

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
        s.useBsdfMis   = j.value("useBsdfMis",   s.useBsdfMis);
        s.useRussian   = j.value("useRussian",   s.useRussian);
        s.useStratified = j.value("useStratified", s.useStratified);
        s.useACES       = j.value("useACES",       s.useACES);
        s.useSpectral   = j.value("useSpectral",   s.useSpectral);
        s.useCieCmf     = j.value("useCieCmf",     s.useCieCmf);
        s.useAA         = j.value("useAA",         s.useAA);
        s.aaSamples     = j.value("aaSamples",     s.aaSamples);
        s.useAdaptive   = j.value("useAdaptive",   s.useAdaptive);
        s.useOIDN       = j.value("useOIDN",       s.useOIDN);
        s.heroSamples   = j.value("heroSamples",   s.heroSamples);
        s.threadgroupX  = j.value("threadgroupX",  s.threadgroupX);
        s.threadgroupY  = j.value("threadgroupY",  s.threadgroupY);
        s.useWavefront  = j.value("useWavefront",  s.useWavefront);
        s.wavefrontMultiSample = j.value("wavefrontMultiSample", s.wavefrontMultiSample);
        s.spectralFork  = j.value("spectralFork",  s.spectralFork);
        s.lutChoice     = j.value("lutChoice",     s.lutChoice);
        s.debugMode    = j.value("debugMode",    s.debugMode);
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
                // Optional snaps. Defaults to "don't change" sentinels
                // when missing, so older settings files migrate cleanly.
                p.width         = pj.value("width",         0);
                p.height        = pj.value("height",        0);
                p.snapSquare    = pj.value("snapSquare",    -1);
                p.useDenoise    = pj.value("useDenoise",    -1);
                p.useMIS        = pj.value("useMIS",        -1);
                p.useRussian    = pj.value("useRussian",    -1);
                p.useStratified = pj.value("useStratified", -1);
                p.useAA         = pj.value("useAA",         -1);
                p.useAdaptive   = pj.value("useAdaptive",   -1);
                p.useOIDN       = pj.value("useOIDN",       -1);
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

// Build the JSON representation of the current Settings. Pulled out of
// saveSettings so we can capture a snapshot at startup, compare to the
// snapshot at exit, and skip writing the file if nothing changed.
// avoids creating a settings file for users who never customize anything.
static json buildSettingsJson(const Settings &s)
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
    j["useBsdfMis"]   = s.useBsdfMis;
    j["useRussian"]   = s.useRussian;
    j["useStratified"] = s.useStratified;
    j["useACES"]       = s.useACES;
    j["useSpectral"]   = s.useSpectral;
    j["useCieCmf"]     = s.useCieCmf;
    j["useAA"]         = s.useAA;
    j["aaSamples"]     = s.aaSamples;
    j["useAdaptive"]   = s.useAdaptive;
    j["useOIDN"]       = s.useOIDN;
    j["heroSamples"]   = s.heroSamples;
    j["threadgroupX"]  = s.threadgroupX;
    j["threadgroupY"]  = s.threadgroupY;
    j["useWavefront"]  = s.useWavefront;
    j["wavefrontMultiSample"] = s.wavefrontMultiSample;
    j["spectralFork"]  = s.spectralFork;
    j["lutChoice"]     = s.lutChoice;
    j["debugMode"]    = s.debugMode;
    json arr = json::array();
    for (const auto &p : s.presets)
    {
        json pj;
        pj["name"] = p.name;
        pj["depth"] = p.depth;
        pj["samples"] = p.samples;
        pj["shadow"] = p.shadowSamples;
        // Optional snaps. Always serialize so round-tripping doesn't
        // silently lose them; loaders treat the sentinel values
        // (0 / -1) as "no snap."
        pj["width"]         = p.width;
        pj["height"]        = p.height;
        pj["snapSquare"]    = p.snapSquare;
        pj["useDenoise"]    = p.useDenoise;
        pj["useMIS"]        = p.useMIS;
        pj["useRussian"]    = p.useRussian;
        pj["useStratified"] = p.useStratified;
        pj["useAA"]         = p.useAA;
        pj["useAdaptive"]   = p.useAdaptive;
        pj["useOIDN"]       = p.useOIDN;
        arr.push_back(std::move(pj));
    }
    j["presets"] = std::move(arr);
    return j;
}

static void saveSettings(const Settings &s)
{
    std::ofstream out(settingsPath());
    out << buildSettingsJson(s).dump(2);
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

// --- LUT cache + apply-before-render -------------------------------------
//
// Process-scoped cache so swapping back and forth between LUT choices in a
// session doesn't re-pay the build/load cost. Keyed by:
//   "build"        - the in-process built LUT
//   "<name>"       - a disk-loaded LUT (display name from LutDiscovery)
// "off" is not cached; it just means setActiveLUT(nullptr).
//
// applyLutChoice runs on the GUI thread right before kicking off the
// render worker. It's synchronous and can take several seconds the first
// time "build" is selected (~4 sec at kRes=16). Subsequent calls hit the
// cache and return in microseconds.
//
// The setActiveLUT call is process-global (single-threaded scene-load
// contract); so long as we only swap the active LUT while no render is
// in flight, this is safe. The GUI guarantees that by gating the apply
// call behind "Render button pressed, no worker running yet."
namespace
{
    std::unordered_map<std::string, std::unique_ptr<RGBToSpectrum::LUT>> g_lutCache;

    // Returns "" on success; otherwise a one-line warning to surface in
    // the UI. Does not throw - a missing file or load error simply falls
    // back to off and reports the message.
    std::string applyLutChoice(const std::string &choice)
    {
        if (choice == "off" || choice.empty())
        {
            RGBToSpectrum::setActiveLUT(nullptr);
            return {};
        }
        if (choice == "build")
        {
            auto it = g_lutCache.find("build");
            if (it == g_lutCache.end())
            {
                auto lut = std::make_unique<RGBToSpectrum::LUT>();
                RGBToSpectrum::buildLUT(*lut);
                it = g_lutCache.emplace("build", std::move(lut)).first;
            }
            RGBToSpectrum::setActiveLUT(it->second.get());
            return {};
        }
        // Disk-backed LUT: resolve via discovery, load if not cached.
        auto it = g_lutCache.find(choice);
        if (it == g_lutCache.end())
        {
            auto registry = LutDiscovery::discoverLUTs();
            auto found = std::find_if(registry.begin(), registry.end(),
                [&](const LutDiscovery::DiscoveredLUT &d) { return d.name == choice; });
            if (found == registry.end())
            {
                RGBToSpectrum::setActiveLUT(nullptr);
                return "LUT '" + choice + "' not found; falling back to runtime homotopy";
            }
            auto lut = std::make_unique<RGBToSpectrum::LUT>();
            std::string err;
            if (!RGBToSpectrum::loadLUT(found->filePath, *lut, &err))
            {
                RGBToSpectrum::setActiveLUT(nullptr);
                return "loadLUT(" + found->filePath + "): " + err
                       + "; falling back to runtime homotopy";
            }
            it = g_lutCache.emplace(choice, std::move(lut)).first;
        }
        RGBToSpectrum::setActiveLUT(it->second.get());
        return {};
    }
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
        // Apply the LUT choice before any material spectra get fit. Only
        // matters when spectral mode is on - RGB ignores the active LUT
        // entirely - so skip the work otherwise. setActiveLUT(nullptr)
        // resets any prior session's LUT cleanly.
        if (settings.useSpectral)
        {
            std::string warn = applyLutChoice(settings.lutChoice);
            if (!warn.empty())
                std::fprintf(stderr, "[lut] %s\n", warn.c_str());
        }
        else
        {
            RGBToSpectrum::setActiveLUT(nullptr);
        }

        Scenes::SceneData sceneData = sceneLoader();

        std::string outDir = settings.outputDir;
        if (outDir.empty())
            outDir = (fs::current_path() / "Image").string();

#ifdef _WIN32
        // Stamp the activity log just before the renderer touches the GPU.
        // If the next thing that happens is a TDR-driven process death, the
        // next launch will MessageBox this content. Cleared on the success
        // path below; the exception catch leaves it intact too (the
        // exception filter MessageBox will reference it).
        {
            // Build a short technique list so the post-mortem MessageBox
            // tells us which knobs were on if the GPU dies. Order matches
            // the GUI checkboxes for easy mental mapping.
            std::string techs;
            auto add = [&](bool on, const char *label) {
                if (!on) return;
                if (!techs.empty()) techs += ",";
                techs += label;
            };
            add(settings.useDenoise,    "denoise");
            add(settings.useMIS,        "mis");
            add(settings.useRussian,    "russian");
            add(settings.useStratified, "strat");
            add(settings.useACES,       "aces");
            add(settings.useSpectral,   "spectral");
            if (settings.aaSamples > 1)
                add(true, ("aa" + std::to_string(settings.aaSamples)).c_str());
            add(settings.useAdaptive,   "adaptive");
            add(settings.useOIDN,       "oidn");
            if (techs.empty()) techs = "(none)";

            char act[512];
            std::snprintf(act, sizeof(act),
                "Rendering '%s' at d=%d s=%d S=%d w=%d h=%d (%s)\n"
                "Techniques: %s",
                sceneData.name.c_str(),
                settings.depth, settings.samples, settings.shadowSamples,
                settings.width, settings.height,
#if PCR_USE_GPU
                "GPU"
#else
                "CPU"
#endif
                , techs.c_str()
            );
            setActivity(act);
        }
#endif

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
        renderer.useACES       = settings.useACES;
        renderer.useSpectral   = settings.useSpectral;
        renderer.useCieCmf     = settings.useCieCmf;
        renderer.useBsdfMis    = settings.useBsdfMis;
        renderer.heroSamples   = settings.heroSamples;
        renderer.aaSamples     = settings.useAA ? std::max(1, settings.aaSamples) : 1;
        renderer.useAdaptive   = settings.useAdaptive;
        renderer.useOIDN       = settings.useOIDN;
#if PCR_USE_GPU
        renderer.threadgroupX  = settings.threadgroupX;
        renderer.threadgroupY  = settings.threadgroupY;
        renderer.useWavefront  = settings.useWavefront;
        renderer.wavefrontMultiSample = settings.wavefrontMultiSample;
        renderer.spectralFork  = settings.spectralFork;
#endif
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
#ifdef _WIN32
        // Render returned without crashing. the activity log can go.
        clearActivity();
#endif
    }
    catch (const std::exception &ex)
    {
        job->errorMessage = ex.what();
        // Leave activity log in place; reportPreviousCrash on next launch
        // surfaces it. We also surface ex.what() inline in the GUI right
        // now, so the user sees both.
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
// zTXt (compressed) and iTXt (international). we only write tEXt from the
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
// samples^depth. branched paths terminate fast in practice via emissive
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
        std::snprintf(buf, sizeof(buf), "(very long, try fewer samples)");
    }
    return buf;
}

// --- Open path in OS file manager / image viewer ------------------------
//
// Best-effort. Uses xdg-open on Linux, open on macOS, ShellExecute on
// Windows. Failures are silent. the buttons are conveniences, not core.
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
    // input. when typing in the input field, arrows should move the cursor.
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

#ifdef _WIN32
// Activity log: a tiny "what was the renderer doing" file that we write
// at render start and delete on clean exit. If the process dies before
// the delete runs (the canonical TDR case. Windows kernel hard-resets
// the GPU and terminates us, no exception filter ever fires), the file
// survives. The next launch finds it and shows a MessageBox so the user
// gets some signal that the previous run died and what it was rendering.
static fs::path activityLogPath()
{
    return fs::current_path() / (std::string(PCR_BINARY_NAME) + ".lastrun.txt");
}

static void setActivity(const std::string &what)
{
    std::ofstream f(activityLogPath());
    if (f) f << what;
}

static void clearActivity()
{
    std::error_code ec;
    fs::remove(activityLogPath(), ec);
}

static void reportPreviousCrash()
{
    fs::path p = activityLogPath();
    std::error_code ec;
    if (!fs::exists(p, ec)) return;
    std::ifstream f(p);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    fs::remove(p, ec);
    if (content.empty()) return;

    std::string body =
        "The previous run was forcibly terminated.\n\n"
        "Last activity:\n" + content + "\n\n"
        "We can't capture the actual error from inside the process. When "
        "this happens it's almost always a Windows kernel-level GPU reset "
        "(TDR), and the kernel kills the process before any of our error-"
        "reporting code runs. There's no exception to catch, no GL error "
        "code to print.\n\n"
        "To find the real cause:\n"
        "  1. Open Event Viewer (Windows+R -> eventvwr).\n"
        "  2. Windows Logs -> System.\n"
        "  3. Look for an Error/Warning in the last few minutes from\n"
        "     source nvlddmkm / amdkmdag / Display.\n\n"
        "If those events appear, it's TDR. The GPU watchdog killed a "
        "single dispatch that took longer than ~2 seconds. Common fixes: "
        "lower preset, smaller samples, smaller resolution.\n\n"
        "If those events DON'T appear, the failure was something else "
        "(driver bug, memory pressure, our code). Click the Debug button "
        "in the top-right before the next render. That pops a console "
        "with renderer error output.";
    MessageBoxA(nullptr, body.c_str(),
                PCR_BINARY_NAME ": previous run crashed",
                MB_OK | MB_ICONWARNING);
}

// In-process unhandled exception (segfault, divide-by-zero, etc.. NOT
// TDR; TDR doesn't go through SEH). Show a MessageBox with what we know
// before unwinding. The activity-log path also fires for these because
// we don't get to clearActivity().
static LONG WINAPI unhandledExceptionFilter(EXCEPTION_POINTERS *ep)
{
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s crashed with unhandled exception.\n\n"
                  "Code: 0x%08lx\nAddress: %p\n\n"
                  "See %s.lastrun.txt next to the binary for what the\n"
                  "renderer was doing.\n\n"
                  "Set PCR_DEBUG=1 for verbose logging on the next run.",
                  PCR_BINARY_NAME,
                  ep->ExceptionRecord->ExceptionCode,
                  ep->ExceptionRecord->ExceptionAddress,
                  PCR_BINARY_NAME);
    MessageBoxA(nullptr, buf, PCR_BINARY_NAME ": crashed",
                MB_OK | MB_ICONERROR);
    return EXCEPTION_EXECUTE_HANDLER;
}

// Optional debug console + log (via the same pipe-tee as before, but now
// gated behind PCR_DEBUG so the default GUI launch is silent). Returns
// true on success.
static bool openDebugConsole()
{
    if (GetConsoleWindow() != nullptr) return false; // already attached
    if (!AllocConsole()) return false;
    SetConsoleTitleA(PCR_BINARY_NAME ": debug log");
    SetConsoleCtrlHandler([](DWORD ev) -> BOOL {
        return ev == CTRL_CLOSE_EVENT;
    }, TRUE);

    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    std::string logPath = std::string(PCR_BINARY_NAME) + ".log";
    HANDLE hLog = CreateFileA(logPath.c_str(),
                              FILE_APPEND_DATA, FILE_SHARE_READ,
                              nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);

    HANDLE hPipeRead = INVALID_HANDLE_VALUE, hPipeWrite = INVALID_HANDLE_VALUE;
    if (CreatePipe(&hPipeRead, &hPipeWrite, nullptr, 0))
    {
        int writeFd = _open_osfhandle((intptr_t)hPipeWrite, _O_TEXT);
        if (writeFd >= 0)
        {
            _dup2(writeFd, _fileno(stderr));
            _dup2(writeFd, _fileno(stdout));
            _close(writeFd);
            std::setvbuf(stderr, nullptr, _IONBF, 0);
            std::setvbuf(stdout, nullptr, _IONBF, 0);

            std::thread([hPipeRead, hConsole, hLog]() {
                char buf[4096];
                DWORD n;
                while (ReadFile(hPipeRead, buf, sizeof(buf), &n, nullptr) && n > 0)
                {
                    DWORD written;
                    WriteFile(hConsole, buf, n, &written, nullptr);
                    if (hLog != INVALID_HANDLE_VALUE)
                        WriteFile(hLog, buf, n, &written, nullptr);
                }
            }).detach();

            std::fprintf(stderr,
                         "%s debug log (build " __DATE__ " " __TIME__ ")\n"
                         "Output is also tee'd to %s in the cwd.\n"
                         "Closing this window won't quit the app.\n\n",
                         PCR_BINARY_NAME, logPath.c_str());
            std::fflush(stderr);
            return true;
        }
        CloseHandle(hPipeWrite);
        CloseHandle(hPipeRead);
    }
    if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
    // Fallback: console without the log-file tee.
    std::freopen("CONOUT$", "w", stderr);
    std::freopen("CONOUT$", "w", stdout);
    return true;
}
#endif

int main(int, char **)
{
    pcrSetupLibSearch();
#ifdef _WIN32
    // Default behavior is silent. no popup console, no log file. The GUI
    // is the GUI. Two diagnostic hooks layered on top:
    //
    //  - reportPreviousCrash() shows a MessageBox if the previous launch
    //    didn't clean up its activity log (the TDR signature, since
    //    TDR-killed processes can't run any cleanup code).
    //  - unhandledExceptionFilter shows a MessageBox on any in-process
    //    crash that isn't TDR (segfault, etc.).
    //
    // PCR_DEBUG=1 in the environment opts back into the always-on
    // console + tee'd log file for cases where the crash output isn't
    // enough. typically when developing rather than just rendering.
    SetUnhandledExceptionFilter(unhandledExceptionFilter);
    reportPreviousCrash();
#endif

    // Load settings early. debug-mode opt-in lives in there, and we want
    // the console to pop before any renderer activity starts. Snapshot
    // the loaded JSON so we can skip writing the file on exit if the user
    // never customized anything (avoids creating <binary>.json by default
    // for fresh-install users).
    Settings settings;
    loadSettings(settings);
    std::string loadedSettingsJson = buildSettingsJson(settings).dump(2);

#ifdef _WIN32
    {
        const char *debugEnv = std::getenv("PCR_DEBUG");
        bool envOn = debugEnv && debugEnv[0] && debugEnv[0] != '0';
        if (envOn || settings.debugMode)
            openDebugConsole();
    }
#endif

    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        return 1;

    // macOS NSGL only exposes legacy 2.1 or Core 3.2+; asking for 3.0
    // returns "macOS does not support OpenGL 3.0 or 3.1." Use the
    // standard Apple GL3 incantation (3.2 Core + forward-compat) and
    // pair it with #version 150 GLSL for ImGui's renderer; everywhere
    // else, the looser 3.0 hint still produces whatever the driver
    // hands back (typically 4.x compat) which the existing Win/Linux
    // ImGui shaders already target.
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glsl_version = "#version 150";
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

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

    // Job batching state. Queue is ephemeral (in-memory, in-session only).
    // Adding to queue snapshots current settings into a JobConfig; "Run
    // queue" walks the list sequentially, advancing in the per-frame
    // driver below. Cancel during a batch halts the whole queue; failed
    // jobs are logged and the batch continues with the next pending one
    // so a single typo doesn't lose the other queued renders.
    std::vector<JobResult> jobQueue;
    BatchPhase batchPhase = BP_INACTIVE;
    int batchCurrentIdx = -1;
    bool batchCancelled = false;

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

        // Top-right Theme + Debug + About buttons.
        {
            const float aboutW = 70.f;
            const float themeW = 70.f;
            const float debugW = 70.f;
            const float gap = ImGui::GetStyle().ItemSpacing.x;
            float windowW = ImGui::GetWindowSize().x;
            float padX = ImGui::GetStyle().WindowPadding.x;
            ImGui::SetCursorPosX(windowW - aboutW - debugW - themeW - 2 * gap - padX);
            if (ImGui::Button(settings.darkTheme ? "Light" : "Dark", ImVec2(themeW, 0)))
            {
                settings.darkTheme = !settings.darkTheme;
                applyTheme(settings.darkTheme);
            }
            ImGui::SameLine();
            // Debug button: toggles persistent debugMode. On Windows,
            // turning ON immediately allocates a console + log file
            // (handy when you want output for the render you're about
            // to start). Turning OFF only persists for next launch.
            // closing an already-allocated console mid-session would
            // leave stderr pointed at a dead handle.
            if (settings.debugMode)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.40f, 0.55f, 0.30f, 1.0f));
            if (ImGui::Button("Debug", ImVec2(debugW, 0)))
            {
                settings.debugMode = !settings.debugMode;
#ifdef _WIN32
                if (settings.debugMode && GetConsoleWindow() == nullptr)
                    openDebugConsole();
#endif
            }
            if (settings.debugMode)
                ImGui::PopStyleColor();
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip(
                    "Pop a console + log file with renderer diagnostics.\n"
                    "Persists across sessions. Once opened during a\n"
                    "session, the console stays until app exit.");
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
                        ImGui::SetTooltip("v%s, built into binary", s.version.c_str());
                    else
                        ImGui::SetTooltip("v%s: %s", s.version.c_str(), s.filePath.c_str());
                }
            }
            ImGui::EndCombo();
        }
        if (!sceneScanError.empty())
            ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "Scene scan: %s", sceneScanError.c_str());

        ImGui::SeparatorText("Quality");

        // Preset buttons always snap depth/samples/shadow. Optional
        // per-preset fields (width/height/square + technique flags)
        // also apply if set; that's how "Picture" doubles as a
        // one-click hero render configuration without forcing every
        // preset to carry the same surface area.
        for (size_t i = 0; i < settings.presets.size(); i++)
        {
            if (i > 0) ImGui::SameLine();
            const auto &p = settings.presets[i];
            if (ImGui::Button(p.name.c_str()))
            {
                settings.depth = p.depth;
                settings.samples = p.samples;
                settings.shadowSamples = p.shadowSamples;
                if (p.width > 0)         settings.width = p.width;
                if (p.height > 0)        settings.height = p.height;
                if (p.snapSquare >= 0)   settings.square = (p.snapSquare != 0);
                if (settings.square)     settings.height = settings.width;
                if (p.useDenoise >= 0)   settings.useDenoise   = (p.useDenoise   != 0);
                if (p.useMIS >= 0)       settings.useMIS       = (p.useMIS       != 0);
                if (p.useRussian >= 0)   settings.useRussian   = (p.useRussian   != 0);
                if (p.useStratified >= 0) settings.useStratified = (p.useStratified != 0);
                if (p.useAA >= 0)        settings.useAA        = (p.useAA        != 0);
                if (p.useAdaptive >= 0)  settings.useAdaptive  = (p.useAdaptive  != 0);
                if (p.useOIDN >= 0)      settings.useOIDN      = (p.useOIDN      != 0);
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

#if PCR_USE_GPU
        // Architecture: dispatch-level GPU choices that change HOW the
        // renderer runs (not what it computes). Gated behind the Debug-
        // mode toggle because the day-to-day defaults are A/B-locked and
        // the selectors only matter when re-running A/Bs or testing new
        // architectures. Houses both the megakernel/wavefront radio and
        // the threadgroup-shape preset row.
        //
        // Metal backend only - OpenGL bakes local_size into the GLSL
        // string and ignores these fields. Read ThreadgroupX/Y and
        // Architecture from the PNG tEXt metadata after a render to
        // confirm what shipped.
        if (settings.debugMode)
        {
            ImGui::SeparatorText("Architecture (debug)");

            // Backend / dispatch architecture radio. Three states
            // encoded across two persisted bools:
            //   (useWavefront=false,   *)               -> Megakernel
            //   (useWavefront=true,  wavefrontMultiSample=false) -> Wavefront (1spp)
            //   (useWavefront=true,  wavefrontMultiSample=true)  -> Wavefront (multi-spp)
            // Style mirrors the Mode section's tone-map / color radios.
            ImGui::TextUnformatted("Backend:");
            ImGui::SameLine();
            if (ImGui::RadioButton("Megakernel", !settings.useWavefront))
                settings.useWavefront = false;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Single MSL kernel runs the full path-"
                                  "tracing loop per pixel. The v1.4.0 "
                                  "baseline; well-tuned, supports every "
                                  "rendering feature. Pick this if you "
                                  "suspect a wavefront regression, or for "
                                  "dispersive-glass renders where its per-"
                                  "wavelength forking beats wavefront's "
                                  "terminate-secondaries strategy.");
            ImGui::SameLine();
            bool isWf1spp = settings.useWavefront && !settings.wavefrontMultiSample;
            bool isWfMspp = settings.useWavefront &&  settings.wavefrontMultiSample;
            if (ImGui::RadioButton("Wavefront (1spp)", isWf1spp))
            {
                settings.useWavefront = true;
                settings.wavefrontMultiSample = false;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Rays split across per-material shading "
                                  "kernels. One sample per pipeline run "
                                  "(smaller working set, more dispatches). "
                                  "v1.5.0 baseline; beat megakernel by "
                                  "~25%% in early A/B.");
            ImGui::SameLine();
            if (ImGui::RadioButton("Wavefront (multi-spp)", isWfMspp))
            {
                settings.useWavefront = true;
                settings.wavefrontMultiSample = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Wavefront with budget-derived samples per "
                                  "pipeline run (matches megakernel's "
                                  "samplesPerPass). Fewer dispatches, much "
                                  "larger working set. A/B against 1spp to "
                                  "find the workload-specific optimum - the "
                                  "memory bandwidth pressure may eat the "
                                  "dispatch-overhead savings, or it may "
                                  "stack on top of them.");

            // Glass dispersion strategy. Only meaningful when wavefront +
            // spectral are both on; disabled and grayed out otherwise, but
            // the setting persists across mode flips.
            bool glassRowActive = settings.useWavefront && settings.useSpectral;
            ImGui::BeginDisabled(!glassRowActive);
            ImGui::TextUnformatted("Glass:");
            ImGui::SameLine();
            if (ImGui::RadioButton("Terminate", !settings.spectralFork))
                settings.spectralFork = false;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("At a dispersive refraction, kill the three "
                                  "secondary hero wavelengths and continue the "
                                  "hero scalar with a 4x energy compensation. "
                                  "Cheap (1x SoA buffers, no append queue) "
                                  "but noisier dispersive caustics since "
                                  "post-glass paths sample one wavelength.");
            ImGui::SameLine();
            if (ImGui::RadioButton("Fork", settings.spectralFork))
                settings.spectralFork = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("At a dispersive refraction, fork into four "
                                  "monochromatic sub-paths each refracting at "
                                  "its own wavelength's Cauchy IOR. Matches "
                                  "megakernel's tracePathSpectral. Cleaner "
                                  "dispersive caustics at the cost of 4x SoA "
                                  "footprint and a variable-output append "
                                  "queue in the glass kernel.");
            ImGui::EndDisabled();

            ImGui::TextUnformatted("Threadgroup:");
            // Presets ordered by total threads. Three are below Apple's
            // SIMD width of 32 (2x2, 4x4, 8x4) so they partially or fully
            // fill a single SIMD group; the rest are multiples of 32 for
            // clean SIMD packing. Sub-SIMD presets remain so empirical
            // A/B can re-measure if the kernel shape changes.
            struct TgPreset { const char *label; int x; int y; };
            static const TgPreset kTgPresets[] = {
                {"2x2",   2,  2},   // 4 threads, ⅛ SIMD - extreme small-group probe
                {"4x4",   4,  4},   // 16 threads, ½ SIMD
                {"8x4",   8,  4},   // 32 threads, exactly 1 SIMD group
                {"8x8",   8,  8},   // 64 threads, 2 SIMD groups (current default)
                {"16x16", 16, 16},  // 256 threads, 8 SIMD groups
                {"32x8",  32, 8},   // 256 threads, wide layout
                {"32x32", 32, 32},  // 1024 threads, full threadgroup
            };
            for (const auto &tp : kTgPresets)
            {
                ImGui::SameLine();
                bool isActive = (settings.threadgroupX == tp.x &&
                                 settings.threadgroupY == tp.y);
                if (isActive)
                    ImGui::PushStyleColor(ImGuiCol_Button,
                                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                if (ImGui::Button(tp.label))
                {
                    settings.threadgroupX = tp.x;
                    settings.threadgroupY = tp.y;
                }
                if (isActive)
                    ImGui::PopStyleColor();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%d x %d = %d threads/group (%d SIMD groups)",
                                      tp.x, tp.y, tp.x * tp.y, (tp.x * tp.y) / 32);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(current %dx%d)",
                                settings.threadgroupX, settings.threadgroupY);
        }
#endif

        // Mode: pipeline-shaping choices that change WHAT the
        // renderer computes, not how efficiently. ACES / Reinhard
        // pick the tone curve; RGB / Spectral pick the color
        // representation through the path tracer. Distinct from the
        // Techniques section below, which only changes convergence
        // speed without changing the target image.
        ImGui::SeparatorText("Mode");
        ImGui::TextUnformatted("Tone-map:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Reinhard", !settings.useACES)) settings.useACES = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("ACES filmic", settings.useACES)) settings.useACES = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("ACES filmic preserves midtone contrast better\n"
                              "than Reinhard at the cost of mild hue shifts in\n"
                              "saturated highlights. Output filename gets -aces.");

        ImGui::TextUnformatted("Color:   ");
        ImGui::SameLine();
        if (ImGui::RadioButton("RGB", !settings.useSpectral)) settings.useSpectral = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("Spectral", settings.useSpectral)) settings.useSpectral = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Spectral mode samples one wavelength per ray and\n"
                              "tracks scalar radiance through bounces. Slower\n"
                              "convergence than RGB at equal sample count;\n"
                              "AA samples >= 16 recommended. Output filename\n"
                              "gets -spectral.");

        // CMF selection. Spectrum -> XYZ conversion can use the cheap
        // Wyman 2013 piecewise-Gaussian fit (default, ~1% off true CIE)
        // or the canonical CIE 1931 tabulated 2-deg observer. The
        // toggle is always visible but only affects spectral-mode
        // output; RGB renders ignore it (same convention as the
        // wavefront-dispersion radio).
        ImGui::TextUnformatted("CMF:     ");
        ImGui::SameLine();
        if (ImGui::RadioButton("Wyman 2013", !settings.useCieCmf)) settings.useCieCmf = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("CIE 1931", settings.useCieCmf)) settings.useCieCmf = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("CIE 1931 = canonical tabulated standard\n"
                              "observer. Wyman 2013 = analytic piecewise-\n"
                              "Gaussian fit, ~1% off CIE, compounds to\n"
                              "~25% drift in integrated RGB equivalents.\n"
                              "Spectral mode only. Recorded in PNG\n"
                              "metadata as CMF: cie or CMF: wyman.");

        // LUT section. Only meaningful when spectral mode is on - the
        // active LUT influences how RGBToSpectrum::fitSigmoidCoefficients
        // dispatches at scene-load, and RGB renders never call into that
        // path. Hide when not spectral so the UI doesn't carry knobs that
        // do nothing.
        //
        // Lifetime trick: discoveredLUTs is a static refreshed only at
        // startup or on the user clicking Refresh. The cost of a rescan
        // is a few filesystem stat calls; it's not the per-frame cost
        // that bothers us, it's that an unstable list would shuffle the
        // dropdown order. Static + explicit refresh keeps the list
        // predictable.
        if (settings.useSpectral)
        {
            // Hero wavelength sampling control. Debug-only because
            // production users have no reason to flip it - hero is
            // always the right default for spectral mode. Exposed for
            // benchmarking and visual A/B comparison against the
            // legacy single-wavelength path. Setting persists across
            // launches; debug mode only controls visibility.
            if (settings.debugMode)
            {
                ImGui::SeparatorText("Hero wavelength (debug)");
                ImGui::TextUnformatted("Channels:");
                ImGui::SameLine();
                bool isHero = (settings.heroSamples >= 4);
                if (ImGui::RadioButton("1 (single-wavelength legacy)", !isHero))
                    settings.heroSamples = 1;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Single wavelength per primary ray. Slower convergence\n"
                        "(more noise per sample). Only useful for benchmarking\n"
                        "the hero implementation against the legacy path.");
                ImGui::SameLine();
                if (ImGui::RadioButton("4 (hero, default)", isHero))
                    settings.heroSamples = 4;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Wilkie 2014 hero wavelength sampling. 4 stratified\n"
                        "wavelengths per primary ray share the same path\n"
                        "geometry; ~3x faster convergence than single-\n"
                        "wavelength on diffuse multi-bounce scenes.");
            }

            static std::vector<LutDiscovery::DiscoveredLUT> discoveredLUTs =
                LutDiscovery::discoverLUTs();

            ImGui::SeparatorText("LUT");

            // Compose the dropdown's current-label. "off" and "build"
            // are synthetic; anything else is a discovered file name.
            const char *currentLabel = "Off (runtime homotopy)";
            if (settings.lutChoice == "build")
                currentLabel = "Build in-memory (~4 sec, not saved)";
            else if (settings.lutChoice != "off" && !settings.lutChoice.empty())
                currentLabel = settings.lutChoice.c_str();

            ImGui::TextUnformatted("Source:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::CalcItemWidth() * 0.7f);
            if (ImGui::BeginCombo("##lutChoice", currentLabel))
            {
                if (ImGui::Selectable("Off (runtime homotopy)",
                                      settings.lutChoice == "off"))
                    settings.lutChoice = "off";
                if (ImGui::Selectable("Build in-memory (~4 sec, not saved)",
                                      settings.lutChoice == "build"))
                    settings.lutChoice = "build";
                if (!discoveredLUTs.empty())
                    ImGui::Separator();
                for (const auto &d : discoveredLUTs)
                {
                    bool sel = (settings.lutChoice == d.name);
                    if (ImGui::Selectable(d.name.c_str(), sel))
                        settings.lutChoice = d.name;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", d.filePath.c_str());
                }
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            if (ImGui::Button("Refresh##luts"))
                discoveredLUTs = LutDiscovery::discoverLUTs();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Rescan luts/ for *.lut files. Useful after dropping\n"
                    "a new LUT into the directory or after clicking Save\n"
                    "below to write a built LUT to disk.");

            // When "Build in-memory" is the choice and a built LUT lives
            // in the cache (i.e. at least one render has run with this
            // setting since launch), offer to save it to disk. The save
            // popups via pfd::save_file so the user picks a filename in
            // luts/ that becomes a first-class entry on next Refresh.
            if (settings.lutChoice == "build")
            {
                bool haveBuilt = (g_lutCache.find("build") != g_lutCache.end());
                ImGui::BeginDisabled(!haveBuilt);
                if (ImGui::Button("Save built LUT to luts/..."))
                {
                    fs::path defaultDir = fs::current_path() / "luts";
                    if (!fs::exists(defaultDir))
                        fs::create_directories(defaultDir);
                    auto sel = pfd::save_file(
                        "Save LUT",
                        (defaultDir / "my-lut.lut").string(),
                        {"PCR LUT", "*.lut"}).result();
                    if (!sel.empty())
                    {
                        // Append .lut if the user typed a bare name.
                        fs::path p = sel;
                        if (p.extension() != ".lut")
                            p += ".lut";
                        const auto &lut = *g_lutCache.at("build");
                        if (RGBToSpectrum::saveLUT(lut, p.string()))
                        {
                            // Refresh the registry so the new file shows
                            // up immediately (saves the user a click).
                            discoveredLUTs = LutDiscovery::discoverLUTs();
                        }
                        else
                        {
                            std::fprintf(stderr, "[lut] saveLUT failed: %s\n",
                                         p.string().c_str());
                        }
                    }
                }
                ImGui::EndDisabled();
                if (!haveBuilt && ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Run a spectral render first - the LUT gets\n"
                        "built lazily on the first render that needs it,\n"
                        "and only then can it be saved.");
            }
        }

        ImGui::SeparatorText("Techniques");
        ImGui::Checkbox("Denoise (5x5 cross-bilateral on output)", &settings.useDenoise);
        ImGui::Checkbox("MIS (light-side balance heuristic, partial)", &settings.useMIS);
        // BSDF-side MIS is a wavefront-only extension that closes the
        // partial MIS into the full balance heuristic. Gray it out when
        // light-side MIS is off (the BSDF side alone is uniformly worse
        // than no MIS) or when running megakernel (where the kernels
        // don't read the flag).
        {
            const bool wfMisAvailable = settings.useMIS && settings.useWavefront;
            ImGui::BeginDisabled(!wfMisAvailable);
            ImGui::Checkbox("MIS BSDF-side (upgrades to full balance heuristic)",
                            &settings.useBsdfMis);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Adds the symmetric BSDF-side weight to MIS.\n"
                                  "Closes the variance-reduction loop: light-side\n"
                                  "and BSDF-side combine via the balance heuristic\n"
                                  "(power beta=2). Wavefront-only; megakernel\n"
                                  "ignores. Disabled when MIS or wavefront is off.");
        }
        ImGui::Checkbox("Russian roulette (terminate paths at depth >= 1)", &settings.useRussian);
        ImGui::Checkbox("Stratified samples (jittered grid first bounce)", &settings.useStratified);
        ImGui::Checkbox("Anti-aliasing (jittered primary rays)", &settings.useAA);
        if (settings.useAA)
        {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            ImGui::InputInt("AA samples##aacount", &settings.aaSamples, 1, 4);
            if (settings.aaSamples < 1) settings.aaSamples = 1;
            if (settings.aaSamples > 64) settings.aaSamples = 64;
        }
        ImGui::Checkbox("Adaptive sampling (early-exit converged pixels)", &settings.useAdaptive);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Only meaningful when AA is on. Stops sampling a\n"
                              "pixel once its relative variance is below 5%%.");
        ImGui::Checkbox("OIDN denoise (Intel Open Image Denoise)", &settings.useOIDN);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Replaces the 5x5 bilateral with Intel OIDN\n"
                              "(neural-net denoiser). Uses albedo + shading\n"
                              "normal aux buffers from first hit. Requires\n"
                              "the binary to be built with PCR_USE_OIDN=ON.");

        ImGui::SeparatorText("Output");
        // Cap at 16384 (2^14) - the Apple Silicon MTLTexture max
        // dimension on every M-series generation, and at or above the
        // OpenGL GL_MAX_TEXTURE_SIZE most desktop drivers report. Going
        // any higher would fail texture allocation rather than just
        // render slowly. RGBA32Float at 16384^2 is ~4.3 GB per texture
        // x 3 (output + OIDN aux) = ~13 GB peak GPU memory, which is
        // fine on M1/M2/M3 Ultra (64+ GB unified) but worth knowing
        // about before pushing the slider all the way right on a
        // smaller box.
        pcrSliderInt("Width", &settings.width, 64, 16384, 8, 64);
        ImGui::Checkbox("Square (height matches width)", &settings.square);
        if (settings.square)
            settings.height = settings.width;
        ImGui::BeginDisabled(settings.square);
        pcrSliderInt("Height", &settings.height, 64, 16384, 8, 64);
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
        ImGui::BeginDisabled(!sceneLoader);
        if (ImGui::Button("Queue", ImVec2(80, 0)))
        {
            JobResult r;
            r.config = makeJobConfig(settings);
            r.status = JobResult::Pending;
            jobQueue.push_back(std::move(r));
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Snapshot the current settings and add as a "
                              "pending job to the queue. Use 'Run queue' "
                              "below to execute all pending jobs sequentially.");
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!isRunning);
        if (ImGui::Button("Cancel", ImVec2(120, 0)))
        {
            job.cancelRequested = true;
            // If a batch is running, halt it after the current job stops
            // rather than auto-advancing to the next queued one.
            if (batchPhase != BP_INACTIVE)
                batchCancelled = true;
        }
        ImGui::EndDisabled();

        // Queue panel. Shown whenever the queue has entries; lets the
        // user review queued configurations before kicking off, see
        // status during/after a batch, and remove individual entries.
        // Entries past their first run (Done/Failed) stay until the user
        // removes or clears them so post-batch comparison is possible.
        if (!jobQueue.empty())
        {
            int pendingCount = 0;
            for (const auto &r : jobQueue)
                if (r.status == JobResult::Pending) pendingCount++;

            char qHeader[64];
            std::snprintf(qHeader, sizeof(qHeader),
                          "Queue (%d total, %d pending)",
                          (int)jobQueue.size(), pendingCount);
            ImGui::SeparatorText(qHeader);

            if (ImGui::BeginTable("queue", 3,
                                  ImGuiTableFlags_SizingStretchProp |
                                  ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_BordersInnerH))
            {
                ImGui::TableSetupColumn("Status",
                                        ImGuiTableColumnFlags_WidthFixed, 80.f);
                ImGui::TableSetupColumn("Config",
                                        ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("",
                                        ImGuiTableColumnFlags_WidthFixed, 30.f);

                int removeIdx = -1;
                for (size_t i = 0; i < jobQueue.size(); i++)
                {
                    const auto &r = jobQueue[i];
                    ImGui::PushID((int)i);
                    ImGui::TableNextRow();

                    ImGui::TableNextColumn();
                    switch (r.status)
                    {
                    case JobResult::Pending:
                        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "pending");
                        break;
                    case JobResult::Running:
                        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1), "running");
                        break;
                    case JobResult::Done:
                        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1), "done");
                        break;
                    case JobResult::Failed:
                        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1), "failed");
                        break;
                    }

                    ImGui::TableNextColumn();
                    renderJobSummary(r.config);
                    if (const char *reason = wavefrontFallbackReason(r.config))
                        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.55f, 1.0f),
                                           "  -> megakernel (wavefront doesn't "
                                           "support %s yet)", reason);
                    if (r.status == JobResult::Done && !r.outputPath.empty())
                        ImGui::TextDisabled("  -> %s",
                            fs::path(r.outputPath).filename().string().c_str());
                    else if (r.status == JobResult::Failed && !r.errorMessage.empty())
                        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1),
                                           "  %s", r.errorMessage.c_str());

                    ImGui::TableNextColumn();
                    // Disallow removing the currently-running batch job;
                    // every other row is fair game.
                    bool isRunningThisRow = (batchPhase == BP_RUNNING_JOB &&
                                             (int)i == batchCurrentIdx);
                    ImGui::BeginDisabled(isRunningThisRow);
                    if (ImGui::SmallButton("X"))
                        removeIdx = (int)i;
                    ImGui::EndDisabled();

                    ImGui::PopID();
                }
                ImGui::EndTable();

                if (removeIdx >= 0)
                {
                    // If we're deleting an entry before the currently-
                    // running job, batchCurrentIdx shifts down by one to
                    // stay pointing at the right slot.
                    if (batchPhase == BP_RUNNING_JOB &&
                        removeIdx < batchCurrentIdx)
                        batchCurrentIdx--;
                    jobQueue.erase(jobQueue.begin() + removeIdx);
                }
            }

            // Buttons: Run queue (only if pending and no batch active),
            // Clear queue (always, blocks removing the currently-running
            // entry which is gated above per-row).
            bool canRun = (batchPhase == BP_INACTIVE) && (pendingCount > 0)
                          && !isRunning;
            ImGui::BeginDisabled(!canRun);
            char runLabel[64];
            std::snprintf(runLabel, sizeof(runLabel),
                          "Run queue (%d)", pendingCount);
            if (ImGui::Button(runLabel, ImVec2(160, 0)))
            {
                // Kick off the state machine; the per-frame driver above
                // picks up BP_ADVANCING on the next iteration and starts
                // the first pending job.
                batchCancelled = false;
                batchCurrentIdx = -1;
                batchPhase = BP_ADVANCING;
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(batchPhase != BP_INACTIVE);
            if (ImGui::Button("Clear queue", ImVec2(120, 0)))
                jobQueue.clear();
            ImGui::EndDisabled();
            if (batchPhase != BP_INACTIVE)
            {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1),
                                   "(running batch...)");
            }
        }

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

        // Batch driver. Runs AFTER the completion handler so the just-
        // finished job's result has already been loaded into the preview
        // and pushed onto the history deque. Then we record the outcome on
        // the queue entry and start the next pending job (or wrap up the
        // batch). Scene-loader lookup is by name from the current scene
        // registry; if a job's scene was removed between Queue and Run,
        // we mark it Failed and advance to the next one rather than
        // wedge the batch.
        if (batchPhase == BP_RUNNING_JOB && !isRunning)
        {
            // The current batch job's worker just exited. Record outcome
            // on jobQueue[batchCurrentIdx] using the same flags the GUI
            // already exposes (job.errorMessage / job.finishedPath).
            if (batchCurrentIdx >= 0 && batchCurrentIdx < (int)jobQueue.size())
            {
                auto &slot = jobQueue[batchCurrentIdx];
                if (!job.errorMessage.empty())
                {
                    slot.status = JobResult::Failed;
                    slot.errorMessage = job.errorMessage;
                }
                else if (!job.finishedPath.empty())
                {
                    slot.status = JobResult::Done;
                    slot.outputPath = job.finishedPath;
                }
                else
                {
                    // Cancelled mid-render: no file, no error. Reset
                    // the slot back to Pending so a subsequent Run-queue
                    // re-attempts it rather than treating the user-
                    // initiated halt as a permanent failure.
                    slot.status = JobResult::Pending;
                }
            }
            batchPhase = BP_ADVANCING;
        }

        while (batchPhase == BP_ADVANCING)
        {
            if (batchCancelled)
            {
                batchPhase = BP_INACTIVE;
                batchCurrentIdx = -1;
                batchCancelled = false;
                break;
            }
            int next = -1;
            for (int i = 0; i < (int)jobQueue.size(); i++)
            {
                if (jobQueue[i].status == JobResult::Pending) { next = i; break; }
            }
            if (next == -1)
            {
                batchPhase = BP_INACTIVE;
                batchCurrentIdx = -1;
                break;
            }
            // Look up the queued job's scene by name. The registry can
            // change between Queue-click and Run-queue (user edited a
            // JSON, scenes refreshed), so we resolve at dispatch time.
            std::function<Scenes::SceneData()> nextLoader;
            for (const auto &s : sceneRegistry)
            {
                if (s.name == jobQueue[next].config.sceneName)
                {
                    nextLoader = s.load;
                    break;
                }
            }
            if (!nextLoader)
            {
                jobQueue[next].status = JobResult::Failed;
                jobQueue[next].errorMessage = "scene not found: " +
                    jobQueue[next].config.sceneName;
                batchCurrentIdx = next;
                continue;  // try the next pending job
            }
            // Kick off this job. Construct a per-job Settings copy by
            // overlaying the JobConfig on the current settings so global
            // state (outputDir, timezone, theme) carries through.
            Settings perJob = settings;
            applyJobConfig(jobQueue[next].config, perJob);
            jobQueue[next].status = JobResult::Running;
            batchCurrentIdx = next;
            if (job.worker.joinable())
                job.worker.join();
            freeImage(previewImg);
            previewLoadedFrom.clear();
            job.worker = std::thread(runRender, &job, &live, perJob,
                                     nextLoader, gpuShared);
            batchPhase = BP_RUNNING_JOB;
            break;
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

    // Only write <binary>.json if the user actually changed something
    // during the session. Compares the current Settings serialization
    // against the snapshot taken at load time. If they match, the file
    // (which may not even exist) is left alone.
    {
        std::string currentSettingsJson = buildSettingsJson(settings).dump(2);
        if (currentSettingsJson != loadedSettingsJson)
            saveSettings(settings);
    }

    freeImage(previewImg);
    if (liveTex) glDeleteTextures(1, &liveTex);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
