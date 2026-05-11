#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "Includes/CLI11.hpp"
#include "Includes/DllSearch.h"
#include "Includes/GpuDefaults.h"
#include "Includes/LutDiscovery.h"
#include "Includes/NumGen.h"
#include "Includes/RGBToSpectrum.h"
#if PCR_USE_GPU
    #include "Gpu/GpuRenderer.h"
    #if !defined(__APPLE__)
        #include <GLFW/glfw3.h>
    #endif
#else
    #include "Includes/Renderer.h"
#endif
#include "Scenes/Scene.h"
#include "Scenes/SceneDiscovery.h"
#include "Scenes/SceneLoader.h"

#ifndef PCR_BINARY_NAME
#define PCR_BINARY_NAME "frank-based-rendering-cli"
#endif
#if PCR_USE_GPU
    #define PCR_CLI_DESC "GPU path tracer (headless)"
#else
    #define PCR_CLI_DESC "CPU path tracer"
#endif

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
    int heroSamples = 4;
    bool useLUT = false;
    std::string lutFile;
    uint64_t seed = 0;
    int threadgroupX = pcr::kDefaultThreadgroupX;
    int threadgroupY = pcr::kDefaultThreadgroupY;
    bool useWavefront = pcr::kDefaultUseWavefront;
    bool wavefrontMultiSample = pcr::kDefaultWavefrontMultiSample;

    CLI::App app{std::string(PCR_BINARY_NAME) + " - " + PCR_CLI_DESC};
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

    auto *techniques = app.add_option_group(
        "Techniques",
        "Quality and rendering technique flags. All default off; toggle "
        "per render via CLI flag. Most can be combined.");
    techniques->add_flag("--denoise", useDenoise,
                 "Apply a 5x5 cross-bilateral filter to the output to reduce noise.");
    techniques->add_flag("--mis", useMIS,
                 "Multiple importance sampling on direct lighting (partial impl, "
                 "light-side weighting only). Slight effect on diffuse-only scenes.");
    techniques->add_flag("--russian", useRussian,
                 "Russian roulette path termination at depth >= 1. Cheaper paths, "
                 "unbiased estimator.");
    techniques->add_flag("--stratified", useStratified,
                 "Jittered stratified samples for the first indirect bounce. Lower "
                 "variance per sample at no extra cost.");
    techniques->add_flag("--aces", useACES,
                 "ACES filmic tone mapping (Narkowicz approximation) instead of "
                 "the default Reinhard. Better midtone contrast; slight hue shift "
                 "in saturated highlights. Output filename gets -aces appended.");
    techniques->add_flag("--aa", useAA,
                 "Anti-aliasing via jittered primary rays. When set, each pixel "
                 "fires --aa-samples primary rays at sub-pixel jitters (default 4) "
                 "and averages, integrating pixel-edge coverage. Linear cost in "
                 "the AA sample count. Filename gets -aa<N> appended.");
    techniques->add_option("--aa-samples", aaSamples,
                   "Number of jittered primary rays per pixel when --aa is set. "
                   "Default 4. Ignored when --aa is off.")
        ->default_str("4")
        ->check(CLI::PositiveNumber);
    techniques->add_flag("--adaptive", useAdaptive,
                 "Adaptive sampling: within the per-pixel AA loop, early-exit "
                 "once relative variance converges below threshold. Speeds up "
                 "well-converged regions at the cost of more samples in noisy "
                 "regions. Only meaningful with --aa.");
    techniques->add_flag("--oidn", useOIDN,
                 "Run Intel Open Image Denoise on the HDR framebuffer "
                 "before tone mapping, with albedo and shading-normal aux "
                 "buffers populated at primary-ray first hit. Replaces the "
                 "5x5 bilateral (--denoise) when both are set. Requires the "
                 "binary to be built with -DPCR_USE_OIDN=ON.");
    techniques->add_flag("--spectral", useSpectral,
                 "Spectral rendering mode. Each primary ray samples a "
                 "single wavelength in [400, 700] nm and tracks scalar "
                 "radiance through bounces; per-pixel CIE XYZ accumulator "
                 "converts to linear sRGB after all samples done. Slower "
                 "convergence than RGB for the same sample count "
                 "(aaSamples >= 16 recommended). Output filename gets "
                 "-spectral appended.");

    auto *options = app.add_option_group(
        "Options",
        "Configuration knobs that change how data is computed but not "
        "the algorithmic behavior of the path tracer.");
    options->add_option("--hero-samples", heroSamples,
                 "Hero wavelength sample count for spectral mode (Wilkie 2014). "
                 "Default 4 = stratified hero sampling (recommended). "
                 "1 = legacy single-wavelength path, exposed for benchmarking "
                 "and visual A/B against the hero default. Other values map "
                 "to 4 internally. Ignored in RGB mode.")
        ->default_str("4")
        ->check(CLI::Range(1, 4));
    options->add_option("--seed", seed,
                 "Fixed PRNG seed for bit-deterministic renders. When set "
                 "to non-zero, all per-thread Monte Carlo PRNG state "
                 "initializes from this seed and the renderer drops to "
                 "single-threaded execution to avoid thread-interleaving "
                 "non-determinism. Renders are reproducible across runs "
                 "on the same machine; cross-machine reproducibility is "
                 "still subject to floating-point rounding differences "
                 "between compilers and CPU architectures. Default 0 = "
                 "random_device per thread (the historical behavior).");
    auto *lutFlag = options->add_flag("--lut", useLUT,
                 "Use a precomputed lookup table for RGB-to-spectrum "
                 "upsampling instead of running the Newton-Raphson + "
                 "homotopy fit per material at scene-load. Builds a "
                 "16^3 sigmoid-coefficient table at startup (~4 seconds) "
                 "by running the fit per cell, then per-material lookup "
                 "is trilinear interpolation in nanoseconds. Same fit "
                 "quality as the runtime homotopy in expectation; pays "
                 "off when scenes have hundreds-plus distinct materials "
                 "(amortizes the per-material homotopy cost). Cornell-"
                 "class scenes hit the build cost without the lookup "
                 "savings, so leave it off there. Only matters in "
                 "--spectral mode; RGB renders ignore the flag.");
    auto *lutFileOpt = options->add_option("--lut-file", lutFile,
                 "Load a precomputed LUT from disk instead of building one. "
                 "Argument is either a path to a .lut file (PLUT binary "
                 "format produced by saveLUT / the GUI 'Save LUT' button) "
                 "or a bare name resolved against the luts/ search path "
                 "($PWD/luts then <binary-dir>/luts; no .lut extension "
                 "needed). Loading from disk is milliseconds vs the ~4 "
                 "second build, so this is the fast-startup path once you "
                 "have a saved LUT. Mutually exclusive with --lut.");
    lutFlag->excludes(lutFileOpt);
    lutFileOpt->excludes(lutFlag);

#if PCR_USE_GPU
    options->add_option("--threadgroup-x", threadgroupX,
                 "Metal compute threadgroup width (in threads). Together "
                 "with --threadgroup-y selects the per-dispatch threadgroup "
                 "shape. Effective values on Apple Silicon are multiples of "
                 "32 in total threads (SIMD width = 32); common shapes are "
                 "8x8, 16x16, 32x8, 32x32. Default selected by v1.4.1 A/B. "
                 "Ignored on the OpenGL backend (local_size baked into the "
                 "GLSL kernel).")
        ->default_str(std::to_string(pcr::kDefaultThreadgroupX))
        ->check(CLI::PositiveNumber);
    options->add_option("--threadgroup-y", threadgroupY,
                 "Metal compute threadgroup height (in threads). See "
                 "--threadgroup-x.")
        ->default_str(std::to_string(pcr::kDefaultThreadgroupY))
        ->check(CLI::PositiveNumber);
    options->add_flag("--wavefront,!--no-wavefront", useWavefront,
                 "Use the wavefront path tracer architecture (rays "
                 "rebatched per-material between bounces) instead of "
                 "the megakernel single-kernel architecture. ON by "
                 "default since v1.4.2 (wavefront beat megakernel by "
                 "~25% in A/B). --no-wavefront forces megakernel for "
                 "this render. Metal backend only (Apple Silicon); "
                 "the flag is a no-op on Win/Linux OpenGL.");
    options->add_flag("--wavefront-multi-sample", wavefrontMultiSample,
                 "When wavefront is on, process the megakernel-equivalent "
                 "samplesPerPass count of samples per pipeline run instead "
                 "of the default one-sample-per-run. Trades dispatch "
                 "overhead (fewer passes) for higher per-ray memory "
                 "bandwidth (larger working set). The 1-spp default won "
                 "the early A/B by ~2-3%, so this flag mostly exists for "
                 "re-measuring at different resolutions / sample counts. "
                 "Ignored when wavefront is off.");
#endif

    CLI11_PARSE(app, argc, argv);

    if (height < 0)
        height = width; // square by default

    if (seed != 0)
        NumGen::setSeed(seed);

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

    // Build or load the spectral upsampling LUT if requested. Lifetime is
    // the rest of the process - holds ~144 KB at kRes=16 and gets read
    // once per material at scene-load. setActiveLUT makes it visible to
    // RGBToSpectrum::fitSigmoidCoefficients which is what populateSpectra
    // calls. --lut and --lut-file are mutually exclusive (CLI11 enforces).
    std::unique_ptr<RGBToSpectrum::LUT> lut;
    if (useLUT)
    {
        std::cout << "Building RGB-to-spectrum LUT..." << std::flush;
        auto t0 = std::chrono::steady_clock::now();
        lut = std::make_unique<RGBToSpectrum::LUT>();
        RGBToSpectrum::buildLUT(*lut);
        RGBToSpectrum::setActiveLUT(lut.get());
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << " " << ms << " ms\n";
    }
    else if (!lutFile.empty())
    {
        // Resolve: if the user passed a path with directory components or
        // a .lut extension, take it as-is. Otherwise treat it as a bare
        // name and look it up in the luts/ registry. Falling back to a
        // direct file open after a registry miss covers ad-hoc paths
        // (e.g. ../experimental.lut) without needing a separate flag.
        std::string resolved = lutFile;
        bool looksLikePath = lutFile.find('/') != std::string::npos
                          || lutFile.find('\\') != std::string::npos
                          || (lutFile.size() >= 4 &&
                              lutFile.substr(lutFile.size() - 4) == ".lut");
        if (!looksLikePath)
        {
            auto registry = LutDiscovery::discoverLUTs();
            auto found = std::find_if(registry.begin(), registry.end(),
                [&](const LutDiscovery::DiscoveredLUT &d) { return d.name == lutFile; });
            if (found == registry.end())
            {
                std::cerr << "Unknown LUT name: " << lutFile << "\n";
                if (registry.empty())
                {
                    std::cerr << "No .lut files found in $PWD/luts or <binary-dir>/luts.\n"
                              << "Pass --lut to build one in-process, or use --lut-file <path>.\n";
                }
                else
                {
                    std::cerr << "Available LUTs:\n";
                    for (const auto &d : registry)
                        std::cerr << "  " << d.name << " (" << d.filePath << ")\n";
                }
                return 1;
            }
            resolved = found->filePath;
        }

        std::cout << "Loading LUT from " << resolved << "..." << std::flush;
        auto t0 = std::chrono::steady_clock::now();
        lut = std::make_unique<RGBToSpectrum::LUT>();
        std::string err;
        if (!RGBToSpectrum::loadLUT(resolved, *lut, &err))
        {
            std::cout << " FAIL\n";
            std::cerr << "loadLUT(" << resolved << "): " << err << "\n";
            return 1;
        }
        RGBToSpectrum::setActiveLUT(lut.get());
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << " " << ms << " ms\n";
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

#if PCR_USE_GPU
    // GPU CLI needs an OpenGL context on non-Apple (OpenglRenderer
    // borrows it via glfwMakeContextCurrent inside render()). On Apple
    // Metal stands alone and the constructor takes a nullptr placeholder
    // just so the signature matches OpenglRenderer.
    GLFWwindow *gpuShared = nullptr;
    #if !defined(__APPLE__)
        if (!glfwInit())
        {
            std::cerr << "GLFW: glfwInit failed; cannot create the hidden "
                      << "GPU context this binary needs." << std::endl;
            return 1;
        }
        // Match the headers-and-version dance Gui/main.cpp does on Apple
        // (3.2 Core + forward-compat). The OpenGL compute shader wants
        // 4.3 but we ask for 3.2 here because the driver hands back the
        // newest profile it can; on every Win/Linux box that runs pcr
        // that's >= 4.3 anyway. macOS only matters in the Metal branch
        // above so the Apple-only hints don't apply here.
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        gpuShared = glfwCreateWindow(1, 1, "pcr-gpu-headless", nullptr, nullptr);
        if (!gpuShared)
        {
            std::cerr << "GLFW: could not create hidden GPU context window."
                      << std::endl;
            glfwTerminate();
            return 1;
        }
    #endif

    GpuRenderer renderer{width, height, depth, samples, shadowSamples, gpuShared};
#else
    Renderer renderer{width, height, depth, samples, shadowSamples};
#endif

    renderer.useDenoise   = useDenoise;
    renderer.useMIS       = useMIS;
    renderer.useRussian   = useRussian;
    renderer.useStratified = useStratified;
    renderer.useACES      = useACES;
    renderer.aaSamples    = useAA ? std::max(1, aaSamples) : 1;
    renderer.useAdaptive  = useAdaptive;
    renderer.useOIDN      = useOIDN;
    renderer.useSpectral  = useSpectral;
    renderer.heroSamples  = heroSamples;
#if PCR_USE_GPU
    renderer.threadgroupX = threadgroupX;
    renderer.threadgroupY = threadgroupY;
    renderer.useWavefront = useWavefront;
    renderer.wavefrontMultiSample = wavefrontMultiSample;
#else
    (void)threadgroupX;
    (void)threadgroupY;
    (void)useWavefront;
    (void)wavefrontMultiSample;
#endif
    renderer.render(sceneData, start, outputDir);

#if PCR_USE_GPU && !defined(__APPLE__)
    glfwDestroyWindow(gpuShared);
    glfwTerminate();
#endif

    return 0;
}
