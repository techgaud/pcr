#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "Spectrum.h"

// Tabulated SPD (spectral power distribution) loader.
//
// File format: plain text. Lines starting with '#' are comments. Each
// non-empty data line is "wavelength_nm value", whitespace-separated.
// Wavelengths must be ascending. Values are interpreted by caller
// (reflectance, radiance, etc.).
//
// loadSpd reads the file, then resamples to our 5 nm / 61-sample grid
// via linear interpolation between the file's sample pairs. Wavelengths
// outside the file's range are clamped to the nearest endpoint (so a
// 4-sample light SPD with values at 400/500/600/700 fills cleanly).
//
// Search resolution mirrors LutDiscovery / SceneDiscovery: a name like
// "cornell/white-paint" resolves to "<spd-search-root>/cornell/white-paint.spd"
// where the search roots are (in order):
//   1. caller-supplied extra dirs (--spd-dir on CLI, GUI override)
//   2. $PWD/spd
//   3. <executable-dir>/spd
//
// Returns true on success. On error (file missing, parse failure,
// non-monotonic wavelengths, no samples) returns false and writes a
// short message to outError if non-null.
namespace SpdLoader
{
    bool loadSpdFile(const std::string &absolutePath,
                     Spectrum &out,
                     std::string *outError = nullptr);

    // Resolve a name (e.g. "cornell/white-paint") through the spd/ search
    // path and load. Returns absolute path used on success via outPath.
    bool loadSpdByName(const std::string &name,
                       const std::vector<std::filesystem::path> &extraDirs,
                       Spectrum &out,
                       std::string *outPath = nullptr,
                       std::string *outError = nullptr);
}
