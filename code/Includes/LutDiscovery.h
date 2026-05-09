#pragma once

#include <filesystem>
#include <string>
#include <vector>

// LUT registry. The CLI and GUI both ask discoverLUTs() for the list of
// .lut files available on disk under any luts/ search path. The result
// is stable-sorted by display name so dropdowns and --help output don't
// shuffle between runs.
//
// Search order, parallel to Scenes/:
//   1. Caller-supplied extra dirs (e.g. --luts-dir on CLI, override on GUI)
//   2. $PWD/luts
//   3. <executable-dir>/luts (so a binary distributed with luts/ beside
//      it works regardless of cwd)
//
// Within a dir, files matching *.lut are surfaced. Display name is the
// filename stem (no directory, no .lut extension).
namespace LutDiscovery
{
    struct DiscoveredLUT
    {
        std::string name;        // stem, e.g. "default-kres16"
        std::string filePath;    // absolute path to load from
    };

    std::vector<DiscoveredLUT> discoverLUTs(
        const std::vector<std::filesystem::path> &extraDirs = {});
}
