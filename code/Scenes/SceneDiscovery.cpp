#include "SceneDiscovery.h"

#include <algorithm>
#include <unordered_set>

#include "Cornell.h"
#include "CornellSpheres.h"
#include "CornellLargeLight.h"
#include "SceneLoader.h"

#if defined(_WIN32)
  #include <windows.h>
#elif defined(__APPLE__)
  #include <mach-o/dyld.h>
  #include <climits>
#else
  #include <unistd.h>
  #include <climits>
#endif

namespace Scenes
{
    namespace fs = std::filesystem;

    namespace
    {
        // Hardcoded scenes baked into the binary. JSON files with the same
        // name override these; if no JSON cornell ships, this is the fallback
        // that keeps the binary runnable with no Scenes/ folder beside it.
        struct HardcodedEntry
        {
            const char *name;
            const char *version;
            SceneData (*factory)();
        };

        const HardcodedEntry kHardcoded[] = {
            {"cornell",             CORNELL_VERSION,             &makeCornell},
            {"cornell-spheres",     CORNELL_SPHERES_VERSION,     &makeCornellSpheres},
            {"cornell-large-light", CORNELL_LARGE_LIGHT_VERSION, &makeCornellLargeLight},
        };

        // Probe a directory for *.json files, parse each, and append valid
        // results to `out`. Files already present in `seenNames` are skipped
        // (later dirs in the search order lose to earlier ones).
        void scanDir(const fs::path &dir,
                     std::vector<DiscoveredScene> &out,
                     std::unordered_set<std::string> &seenNames,
                     const std::function<void(const std::string &)> &onError)
        {
            std::error_code ec;
            if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
                return;

            std::vector<fs::path> files;
            for (const auto &entry : fs::directory_iterator(dir, ec))
            {
                if (ec) break;
                if (!entry.is_regular_file()) continue;
                if (entry.path().extension() != ".json") continue;
                files.push_back(entry.path());
            }
            // Stable order so --list-scenes is deterministic.
            std::sort(files.begin(), files.end());

            for (const auto &file : files)
            {
                try
                {
                    SceneData s = loadSceneFromFile(file.string());
                    if (seenNames.count(s.name))
                        continue;
                    DiscoveredScene d;
                    d.name = s.name;
                    d.version = s.version;
                    d.source = DiscoveredScene::Source::Json;
                    d.filePath = file.string();
                    std::string capturedPath = d.filePath;
                    d.load = [capturedPath]() { return loadSceneFromFile(capturedPath); };
                    seenNames.insert(d.name);
                    out.push_back(std::move(d));
                }
                catch (const std::exception &e)
                {
                    if (onError)
                        onError(std::string("Skipping ") + file.string() + ": " + e.what());
                }
            }
        }
    } // namespace

    fs::path executableDir()
    {
#if defined(_WIN32)
        wchar_t buf[MAX_PATH];
        DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
        if (n == 0 || n == MAX_PATH) return fs::current_path();
        return fs::path(buf).parent_path();
#elif defined(__APPLE__)
        char buf[PATH_MAX];
        uint32_t size = sizeof(buf);
        if (_NSGetExecutablePath(buf, &size) != 0) return fs::current_path();
        std::error_code ec;
        fs::path canonical = fs::canonical(buf, ec);
        if (ec) return fs::path(buf).parent_path();
        return canonical.parent_path();
#else
        char buf[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n <= 0) return fs::current_path();
        buf[n] = 0;
        return fs::path(buf).parent_path();
#endif
    }

    std::vector<DiscoveredScene> discoverScenes(
        const std::vector<fs::path> &extraDirs,
        const std::function<void(const std::string &)> &onError)
    {
        std::vector<DiscoveredScene> out;
        std::unordered_set<std::string> seenNames;

        // JSON dirs first so JSON overrides hardcoded.
        // Order: --scenes-dir overrides, then $PWD/Scenes, then exeDir/Scenes.
        std::vector<fs::path> dirs = extraDirs;
        dirs.push_back(fs::current_path() / "Scenes");
        fs::path ed = executableDir();
        if (!ed.empty())
            dirs.push_back(ed / "Scenes");

        // Dedup so we don't double-scan when cwd == exeDir.
        std::unordered_set<std::string> seenDirs;
        for (const auto &d : dirs)
        {
            std::error_code ec;
            std::string canonical;
            fs::path c = fs::weakly_canonical(d, ec);
            canonical = ec ? d.string() : c.string();
            if (seenDirs.count(canonical)) continue;
            seenDirs.insert(canonical);
            scanDir(d, out, seenNames, onError);
        }

        // Hardcoded entries fill in any names not yet seen.
        for (const auto &h : kHardcoded)
        {
            if (seenNames.count(h.name)) continue;
            DiscoveredScene d;
            d.name = h.name;
            d.version = h.version;
            d.source = DiscoveredScene::Source::Hardcoded;
            auto factory = h.factory;
            d.load = [factory]() { return factory(); };
            seenNames.insert(d.name);
            out.push_back(std::move(d));
        }

        // Sort by name for stable presentation across the registry.
        std::sort(out.begin(), out.end(),
                  [](const DiscoveredScene &a, const DiscoveredScene &b) {
                      return a.name < b.name;
                  });
        return out;
    }
}
