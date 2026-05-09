#include "Includes/LutDiscovery.h"

#include <algorithm>
#include <unordered_set>

#include "Scenes/SceneDiscovery.h"  // Scenes::executableDir()

namespace LutDiscovery
{
    namespace fs = std::filesystem;

    namespace
    {
        void scanDir(const fs::path &dir,
                     std::vector<DiscoveredLUT> &out,
                     std::unordered_set<std::string> &seenNames)
        {
            std::error_code ec;
            if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
                return;

            std::vector<fs::path> files;
            for (const auto &entry : fs::directory_iterator(dir, ec))
            {
                if (ec) break;
                if (!entry.is_regular_file()) continue;
                if (entry.path().extension() != ".lut") continue;
                files.push_back(entry.path());
            }
            std::sort(files.begin(), files.end());

            for (const auto &file : files)
            {
                std::string name = file.stem().string();
                if (seenNames.count(name)) continue;
                DiscoveredLUT d;
                d.name = name;
                d.filePath = file.string();
                seenNames.insert(name);
                out.push_back(std::move(d));
            }
        }
    }

    std::vector<DiscoveredLUT> discoverLUTs(const std::vector<fs::path> &extraDirs)
    {
        std::vector<DiscoveredLUT> out;
        std::unordered_set<std::string> seenNames;

        std::vector<fs::path> dirs = extraDirs;
        dirs.push_back(fs::current_path() / "luts");
        fs::path ed = Scenes::executableDir();
        if (!ed.empty())
            dirs.push_back(ed / "luts");

        std::unordered_set<std::string> seenDirs;
        for (const auto &d : dirs)
        {
            std::error_code ec;
            fs::path c = fs::weakly_canonical(d, ec);
            std::string canonical = ec ? d.string() : c.string();
            if (seenDirs.count(canonical)) continue;
            seenDirs.insert(canonical);
            scanDir(d, out, seenNames);
        }

        std::sort(out.begin(), out.end(),
                  [](const DiscoveredLUT &a, const DiscoveredLUT &b) {
                      return a.name < b.name;
                  });
        return out;
    }
}
