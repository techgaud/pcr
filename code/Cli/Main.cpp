#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <cstdlib>
#include <ctime>

#include "Includes/CLI11.hpp"
#include "Includes/DllSearch.h"
#include "Includes/Renderer.h"
#include "Scenes/Scene.h"
#include "Scenes/SceneDiscovery.h"
#include "Scenes/SceneLoader.h"

namespace
{
    // Map common short zone abbreviations to DST-aware POSIX strings so
    // --tz EST gives EDT in summer and EST in winter, etc. Anything not
    // in this table is passed through verbatim, so power users can do
    // --tz America/New_York or --tz Etc/GMT+5.
    std::string resolveTimezone(const std::string &userTz)
    {
        static const std::unordered_map<std::string, std::string> map = {
            {"EST", "EST5EDT"},
            {"CST", "CST6CDT"},
            {"MST", "MST7MDT"},
            {"PST", "PST8PDT"},
            {"UTC", "UTC"},
            {"GMT", "UTC"},
        };
        auto it = map.find(userTz);
        return it != map.end() ? it->second : userTz;
    }

    void applyTimezone(const std::string &tz)
    {
        if (tz.empty())
            return;
        std::string resolved = resolveTimezone(tz);
#ifdef _WIN32
        _putenv_s("TZ", resolved.c_str());
#else
        setenv("TZ", resolved.c_str(), 1);
#endif
        tzset();
    }

}

int main(int argc, char *argv[])
{
    pcrSetupLibSearch();
    auto start = std::chrono::steady_clock::now();

    int depth = 4;
    int samples = 16;
    int shadowSamples = 4;
    int width = 720;
    int height = -1; // sentinel: if unset, copies width
    std::string scene = "cornell";
    std::string timezone;
    std::string outputDir;
    std::vector<std::string> extraSceneDirs;
    bool listScenes = false;
    bool useDenoise = false;
    bool useMIS = false;
    bool useRussian = false;
    bool useStratified = false;
    bool useACES = false;
    bool useAA = false;
    int aaSamples = 4;
    bool useAdaptive = false;
    bool useOIDN = false;
    bool useSpectral = false;

    CLI::App app{"frank-based-rendering-cli - CPU path tracer"};
    app.add_option("--scene", scene, "Scene to render (default: cornell)")
        ->default_str("cornell");
    app.add_option("-d,--depth", depth, "Max ray bounces")
        ->default_str("4")
        ->check(CLI::PositiveNumber);
    app.add_option("-s,--samples", samples, "Indirect-light samples per hit")
        ->default_str("16")
        ->check(CLI::PositiveNumber);
    app.add_option("-S,--shadow", shadowSamples, "Direct-light shadow rays per hit")
        ->default_str("4")
        ->check(CLI::PositiveNumber);
    app.add_option("-w,--width", width, "Output width in pixels")
        ->default_str("720")
        ->check(CLI::PositiveNumber);
    app.add_option("--height", height,
                   "Output height in pixels. If omitted, defaults to width (square output).")
        ->check(CLI::PositiveNumber);
    app.add_option("--tz,--timezone", timezone,
                   "Timezone for output filename. Friendly names (EST, CST, MST, PST, UTC) "
                   "are mapped to DST-aware POSIX strings; anything else passes through "
                   "(e.g. America/New_York). Default: system local time.");
    app.add_option("-o,--output", outputDir, "Output directory (default: $PWD/Image)");
    app.add_option("--scenes-dir", extraSceneDirs,
                   "Additional directory to search for *.json scene files. May be "
                   "passed multiple times. Searched before the default $PWD/Scenes "
                   "and <binary-dir>/Scenes locations.");
    app.add_flag("--list-scenes", listScenes,
                 "Print all discovered scenes (hardcoded + JSON) with name, version, "
                 "and source, then exit.");

    app.add_flag("--denoise", useDenoise,
                 "Apply a 5x5 cross-bilateral filter to the output to reduce noise.");
    app.add_flag("--mis", useMIS,
                 "Multiple importance sampling on direct lighting (partial impl, "
                 "light-side weighting only). Slight effect on diffuse-only scenes.");
    app.add_flag("--russian", useRussian,
                 "Russian roulette path termination at depth >= 1. Cheaper paths, "
                 "unbiased estimator.");
    app.add_flag("--stratified", useStratified,
                 "Jittered stratified samples for the first indirect bounce. Lower "
                 "variance per sample at no extra cost.");
    app.add_flag("--aces", useACES,
                 "ACES filmic tone mapping (Narkowicz approximation) instead of "
                 "the default Reinhard. Better midtone contrast; slight hue shift "
                 "in saturated highlights. Output filename gets -aces appended.");
    app.add_flag("--aa", useAA,
                 "Anti-aliasing via jittered primary rays. When set, each pixel "
                 "fires --aa-samples primary rays at sub-pixel jitters (default 4) "
                 "and averages, integrating pixel-edge coverage. Linear cost in "
                 "the AA sample count. Filename gets -aa<N> appended.");
    app.add_option("--aa-samples", aaSamples,
                   "Number of jittered primary rays per pixel when --aa is set. "
                   "Default 4. Ignored when --aa is off.")
        ->default_str("4")
        ->check(CLI::PositiveNumber);
    app.add_flag("--adaptive", useAdaptive,
                 "Adaptive sampling: within the per-pixel AA loop, early-exit "
                 "once relative variance converges below threshold. Speeds up "
                 "well-converged regions at the cost of more samples in noisy "
                 "regions. Only meaningful with --aa.");
    app.add_flag("--oidn", useOIDN,
                 "Run Intel Open Image Denoise on the HDR framebuffer "
                 "before tone mapping, with albedo and shading-normal aux "
                 "buffers populated at primary-ray first hit. Replaces the "
                 "5x5 bilateral (--denoise) when both are set. Requires the "
                 "binary to be built with -DPCR_USE_OIDN=ON.");
    app.add_flag("--spectral", useSpectral,
                 "Spectral rendering mode. Each primary ray samples a "
                 "single wavelength in [400, 700] nm and tracks scalar "
                 "radiance through bounces; per-pixel CIE XYZ accumulator "
                 "converts to linear sRGB after all samples done. Slower "
                 "convergence than RGB for the same sample count "
                 "(aaSamples >= 16 recommended). Output filename gets "
                 "-spectral appended.");

    CLI11_PARSE(app, argc, argv);

    if (height < 0)
        height = width; // square by default

    applyTimezone(timezone);

    if (outputDir.empty())
        outputDir = (std::filesystem::current_path() / "Image").string();

    std::vector<std::filesystem::path> extraDirPaths;
    extraDirPaths.reserve(extraSceneDirs.size());
    for (const auto &d : extraSceneDirs)
        extraDirPaths.emplace_back(d);

    auto onDiscoveryError = [](const std::string &msg) {
        std::cerr << "warning: " << msg << "\n";
    };
    auto registry = Scenes::discoverScenes(extraDirPaths, onDiscoveryError);

    if (listScenes)
    {
        std::cout << "Available scenes:\n";
        for (const auto &s : registry)
        {
            std::cout << "  " << s.name << "  " << s.version << "  ";
            if (s.source == Scenes::DiscoveredScene::Source::Hardcoded)
                std::cout << "(hardcoded)";
            else
                std::cout << "(" << s.filePath << ")";
            std::cout << "\n";
        }
        return 0;
    }

    auto it = std::find_if(registry.begin(), registry.end(),
                           [&](const Scenes::DiscoveredScene &s) { return s.name == scene; });
    if (it == registry.end())
    {
        std::cerr << "Unknown scene: " << scene << "\nAvailable scenes:\n";
        for (const auto &s : registry)
            std::cerr << "  " << s.name << "\n";
        return 1;
    }

    Scenes::SceneData sceneData;
    try
    {
        sceneData = it->load();
    }
    catch (const Scenes::SceneLoaderError &e)
    {
        std::cerr << "Error loading scene: " << e.what() << "\n";
        return 1;
    }

    Renderer renderer{width, height, depth, samples, shadowSamples};
    renderer.useDenoise   = useDenoise;
    renderer.useMIS       = useMIS;
    renderer.useRussian   = useRussian;
    renderer.useStratified = useStratified;
    renderer.useACES      = useACES;
    renderer.aaSamples    = useAA ? std::max(1, aaSamples) : 1;
    renderer.useAdaptive  = useAdaptive;
    renderer.useOIDN      = useOIDN;
    renderer.useSpectral  = useSpectral;
    renderer.render(sceneData, start, outputDir);

    return 0;
}
