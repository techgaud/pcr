#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <unordered_map>
#include <functional>
#include <filesystem>
#include <cstdlib>
#include <ctime>

#include "Includes/CLI11.hpp"
#include "Includes/Renderer.h"
#include "Scenes/Scene.h"
#include "Scenes/Cornell.h"

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

    // One entry per available scene. Add new scenes by including their header
    // above and adding a row here.
    using SceneFactory = std::function<Scenes::SceneData()>;
    const std::unordered_map<std::string, SceneFactory> &sceneRegistry()
    {
        static const std::unordered_map<std::string, SceneFactory> map = {
            {"cornell", &Scenes::makeCornell},
        };
        return map;
    }
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::steady_clock::now();

    int depth = 4;
    int samples = 16;
    int shadowSamples = 4;
    int width = 720;
    int height = -1; // sentinel: if unset, copies width
    std::string scene = "cornell";
    std::string timezone;
    std::string outputDir;
    bool useDenoise = false;
    bool useMIS = false;
    bool useRussian = false;
    bool useStratified = false;

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

    CLI11_PARSE(app, argc, argv);

    if (height < 0)
        height = width; // square by default

    applyTimezone(timezone);

    if (outputDir.empty())
        outputDir = (std::filesystem::current_path() / "Image").string();

    const auto &registry = sceneRegistry();
    auto it = registry.find(scene);
    if (it == registry.end())
    {
        std::cerr << "Unknown scene: " << scene << "\nAvailable scenes:\n";
        for (const auto &kv : registry)
            std::cerr << "  " << kv.first << "\n";
        return 1;
    }

    Scenes::SceneData sceneData = it->second();

    Renderer renderer{width, height, 65.f, depth, samples, shadowSamples};
    renderer.useDenoise   = useDenoise;
    renderer.useMIS       = useMIS;
    renderer.useRussian   = useRussian;
    renderer.useStratified = useStratified;
    renderer.render(sceneData, start, outputDir);

    return 0;
}
