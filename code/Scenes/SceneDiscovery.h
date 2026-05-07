#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "Scene.h"

namespace Scenes
{
    // A scene the binary knows how to render. Either built into the binary
    // (Source::Hardcoded, e.g. cornell) or read from a JSON file
    // (Source::Json). UI / CLI code asks discoverScenes() for the merged list.
    struct DiscoveredScene
    {
        enum class Source { Hardcoded, Json };
        std::string name;
        std::string version;
        Source source = Source::Hardcoded;
        std::string filePath; // populated when source == Json

        // Returns a populated SceneData. Hardcoded entries call their C++
        // factory; JSON entries re-parse the file. May throw SceneLoaderError
        // for JSON entries with malformed input (e.g. file edited between
        // discovery and load).
        std::function<SceneData()> load;
    };

    // Returns the merged list of available scenes:
    //   1. Hardcoded scenes from the C++ registry (cornell, cornell-spheres,
    //      cornell-large-light)
    //   2. JSON files in default search dirs ($PWD/Scenes, then dir-of-exe/Scenes)
    //   3. JSON files in extraDirs (from --scenes-dir / GUI override)
    //
    // Resolution rule: JSON scenes win over hardcoded on name collision (a
    // cornell.json supersedes the C++ cornell). Within JSON dirs, earlier
    // dirs in the search order win.
    //
    // Malformed JSON files are reported via the optional onError callback
    // and skipped, so one broken file doesn't hide all the others.
    std::vector<DiscoveredScene> discoverScenes(
        const std::vector<std::filesystem::path> &extraDirs = {},
        const std::function<void(const std::string &)> &onError = nullptr);

    // Returns the directory containing the running executable. Used as one
    // of the default search roots so a binary distributed with a Scenes/
    // folder beside it works regardless of cwd.
    std::filesystem::path executableDir();
}
