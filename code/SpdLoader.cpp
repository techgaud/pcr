#include "Includes/SpdLoader.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "Scenes/SceneDiscovery.h"  // Scenes::executableDir()

namespace SpdLoader
{
    namespace fs = std::filesystem;

    namespace
    {
        // Linear interpolation. Clamps to endpoints outside [pairs.front,
        // pairs.back] - if a file gives 4 samples at 400/500/600/700 nm,
        // the result for 700+ extends as the 700 nm value rather than
        // dropping to zero (real measurements stop at the last sample,
        // not at zero).
        float interp(const std::vector<std::pair<float, float>> &pairs, float lambda)
        {
            if (pairs.empty()) return 0.f;
            if (lambda <= pairs.front().first)  return pairs.front().second;
            if (lambda >= pairs.back().first)   return pairs.back().second;

            // pairs is sorted ascending, so binary-search for upper bound.
            auto hi = std::upper_bound(pairs.begin(), pairs.end(),
                                       std::make_pair(lambda, 0.f),
                                       [](const auto &a, const auto &b) {
                                           return a.first < b.first;
                                       });
            auto lo = hi - 1;
            float t = (lambda - lo->first) / (hi->first - lo->first);
            return lo->second + t * (hi->second - lo->second);
        }
    }

    bool loadSpdFile(const std::string &absolutePath, Spectrum &out, std::string *outError)
    {
        auto fail = [&](const std::string &msg) {
            if (outError) *outError = msg;
            return false;
        };

        std::ifstream in(absolutePath);
        if (!in) return fail("cannot open file: " + absolutePath);

        std::vector<std::pair<float, float>> pairs;
        std::string line;
        int lineNo = 0;
        while (std::getline(in, line))
        {
            lineNo++;
            // Strip leading whitespace; skip blank/comment lines.
            size_t s = 0;
            while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) s++;
            if (s >= line.size()) continue;
            if (line[s] == '#') continue;

            std::istringstream is(line.substr(s));
            float lam, val;
            if (!(is >> lam >> val))
                return fail(absolutePath + ":" + std::to_string(lineNo) + ": expected 'wavelength value'");
            pairs.emplace_back(lam, val);
        }
        if (pairs.empty())
            return fail("no data samples in " + absolutePath);

        // Verify ascending wavelengths. A non-monotonic file is almost
        // certainly an editor error and silently sorting would mask it.
        for (size_t i = 1; i < pairs.size(); i++)
        {
            if (pairs[i].first <= pairs[i - 1].first)
                return fail(absolutePath + ": wavelengths must be strictly ascending");
        }

        for (int i = 0; i < Spectrum::kSamples; i++)
            out[i] = interp(pairs, Spectrum::lambdaAt(i));
        return true;
    }

    bool loadSpdByName(const std::string &name,
                       const std::vector<fs::path> &extraDirs,
                       Spectrum &out,
                       std::string *outPath,
                       std::string *outError)
    {
        auto fail = [&](const std::string &msg) {
            if (outError) *outError = msg;
            return false;
        };

        // Build search-root list (caller-supplied dirs, then $PWD/spd,
        // then exeDir/spd) and try resolving "<root>/<name>.spd" in each.
        std::vector<fs::path> roots = extraDirs;
        roots.push_back(fs::current_path() / "spd");
        fs::path ed = Scenes::executableDir();
        if (!ed.empty()) roots.push_back(ed / "spd");

        for (const auto &root : roots)
        {
            fs::path candidate = root / (name + ".spd");
            std::error_code ec;
            if (fs::exists(candidate, ec))
            {
                if (outPath) *outPath = candidate.string();
                return loadSpdFile(candidate.string(), out, outError);
            }
        }

        // Fall back: try the name verbatim as an absolute or cwd-relative path.
        std::error_code ec;
        if (fs::exists(name, ec))
        {
            if (outPath) *outPath = fs::absolute(name, ec).string();
            return loadSpdFile(name, out, outError);
        }

        std::string msg = "SPD '" + name + "' not found in any of: ";
        for (const auto &r : roots) msg += r.string() + " ";
        return fail(msg);
    }
}
