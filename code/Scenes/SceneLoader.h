#pragma once

#include <stdexcept>
#include <string>

#include "Scene.h"

namespace Scenes
{
    // Thrown by loadSceneFromFile when a JSON scene file is missing,
    // malformed, schema-incompatible, or semantically invalid (e.g. zero
    // primitives flagged as the light). Message is human-readable and
    // includes the file path plus, where available, line/column from
    // the underlying JSON parser.
    struct SceneLoaderError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    // Reads a JSONC scene file from disk and returns a populated SceneData.
    // Comments (// to end of line) are supported. See project-knowledge.md
    // for the schema; SceneLoader.cpp is the authoritative implementation.
    SceneData loadSceneFromFile(const std::string &path);
}
