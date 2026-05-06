#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <cstdlib>
#include <ctime>

#include "Includes/CLI11.hpp"
#include "Includes/Renderer.h"
#include "Includes/Sphere.h"

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
    auto start = std::chrono::steady_clock::now();

    int depth = 4;
    int samples = 16;
    int shadowSamples = 4;
    std::string timezone;
    std::string outputDir;

    CLI::App app{"pcr-cornell - CPU path tracer (Cornell Box scene)"};
    app.add_option("-d,--depth", depth, "Max ray bounces")
        ->default_str("4")
        ->check(CLI::PositiveNumber);
    app.add_option("-s,--samples", samples, "Indirect-light samples per hit")
        ->default_str("16")
        ->check(CLI::PositiveNumber);
    app.add_option("-S,--shadow", shadowSamples, "Direct-light shadow rays per hit")
        ->default_str("4")
        ->check(CLI::PositiveNumber);
    app.add_option("--tz,--timezone", timezone,
                   "Timezone for output filename. Friendly names (EST, CST, MST, PST, UTC) "
                   "are mapped to DST-aware POSIX strings; anything else passes through "
                   "(e.g. America/New_York). Default: system local time.");
    app.add_option("-o,--output", outputDir, "Output directory (default: $PWD/Image)");

    CLI11_PARSE(app, argc, argv);

    applyTimezone(timezone);

    if (outputDir.empty())
        outputDir = (std::filesystem::current_path() / "Image").string();

    Material nonemissive{Vec3f(0.4, 0.4, 0.3), Vec3f(0.f, 0.f, 0.f)};
    Material emissive{Vec3f(0.f, 0.f, 0.f), Vec3f{1.0f, 0.85f, 0.6f}};
    emissive.emissive *= 80.f;

    Plane lightSource(Vec3f{-0.375f, 2.f, -4.25f}, Vec3f{0.75f, 0, 0}, Vec3f{0, 0, 0.4f}, emissive);

    std::vector<Sphere> spheres{
        Sphere(Vec3f(0.f, -1.f, -4.5f), 0.75f, nonemissive)};

    Renderer renderer{712, 712, 65.f, depth, samples, shadowSamples, lightSource};
    renderer.render(spheres, start, outputDir);

    return 0;
}
