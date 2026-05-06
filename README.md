# pcr

A small CPU path tracer in C++20. Ships with a Cornell Box scene; the architecture is designed for adding more scenes.

Multi-threaded, cosine-weighted hemisphere sampling for indirect light, explicit area-light sampling with shadow rays for direct light, soft shadows, emissive surfaces, Reinhard tone mapping. Output is lossless PNG with embedded tEXt metadata describing scene name, version, and render parameters.

## Requirements

A C++20 compiler and CMake 3.20+:

- **Linux**, GCC 10+ or Clang 14+
- **macOS**, Apple Clang 14+ (Xcode 14, macOS 12 or newer)
- **Windows**, MSVC from Visual Studio 2019 16.10+ (2022 recommended)

## Layout

Source code lives under `code/`. Renders are written to `Image/` next to it by default, so the source tree stays clean while past renders are preserved beside the code that produced them.

## Build

From the repo root:

```bash
cmake -S code -B code/Build
cmake --build code/Build --config Release
```

This produces two executables:

- **`pcr-cornell`** — the CLI renderer (small, headless, scriptable)
- **`physically-cringe-renderer-v1.0.0`** — the GUI app (sliders, render button, image preview)

On Windows with Visual Studio they land at `code/Build/Release/<name>.exe`. On Linux and macOS they're at `code/Build/<name>`.

The GUI target uses GLFW (fetched + built statically by CMake on first configure, ~30 sec) and Dear ImGui (vendored under `code/Includes/imgui/`). On Linux you'll need X11 dev headers installed:

```bash
sudo apt install libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev pkg-config
```

Windows and macOS need no additional packages beyond Visual Studio / Xcode CLT.

## Run (GUI)

Double-click the GUI binary, or from a shell:

```bash
./code/Build/physically-cringe-renderer-v1.0.0
```

Sliders set every parameter the CLI exposes (depth, samples, shadow, width, square checkbox + height, scene, timezone, output dir). Click **Render** and the path tracer runs on a worker thread; progress is reported as rows-completed-of-total. **Cancel** mid-render works. When the render finishes the resulting PNG is loaded and shown in the same window. Settings persist to `physically-cringe-renderer.json` next to the executable.

## Run (CLI)

Run from the repo root so the default `$PWD/Image/` resolves to `pcr/Image/`:

```bash
./code/Build/pcr-cornell -d 4 -s 16 -S 4
```

All flags are optional — defaults render a 720x720 Cornell Box at decent quality. Pass `-o <dir>` to send renders elsewhere.

### Flags

| Short | Long | Default | Meaning |
|-------|------|---------|---------|
| | `--scene` | `cornell` | Which scene to render |
| `-d` | `--depth` | `4` | Max ray bounces |
| `-s` | `--samples` | `16` | Indirect-light samples per hit |
| `-S` | `--shadow` | `4` | Direct-light shadow rays per hit |
| `-w` | `--width` | `720` | Output width in pixels |
| | `--height` | (matches `-w`) | Output height; if omitted, output is square |
| | `--tz`, `--timezone` | system local | Timezone for filename, see below |
| `-o` | `--output` | `$PWD/Image` | Output directory |
| `-h` | `--help` | | Print usage and exit |

### Scenes

Currently shipped:

| Name | Version |
|------|---------|
| `cornell` | `1.0.0` |

Each scene declares its own version number that bumps when the scene definition changes. The scene name and version are baked into the output filename and the PNG metadata, so renders stay traceable to the exact scene revision that produced them.

### Output filename

```
<scene>-<version>-<timestamp>-d#-s#-S#-w#[-h#]-t<ms>.png
```

Example with defaults:

```
cornell-1.0.0-20260506-143234-EDT-d4-s16-S4-w720-t12345.png
```

The `-h#` segment is omitted when output is square (height equals width). Render time in milliseconds is appended as `t<ms>` and also printed to stdout.

### PNG metadata

Each render embeds the following uncompressed `tEXt` chunks, readable by any PNG-aware tool:

| Key | Example |
|-----|---------|
| `Software` | `pcr-cornell` |
| `Scene` | `cornell` |
| `SceneVersion` | `1.0.0` |
| `CreationTime` | `20260506-143234-EDT` |
| `Depth` | `4` |
| `Samples` | `16` |
| `ShadowSamples` | `4` |
| `Width` | `720` |
| `Height` | `720` |
| `RenderTimeMs` | `12345` |

Inspect with ImageMagick (`identify -verbose foo.png | grep -A1 tEXt`), Pillow (`PIL.Image.open(path).text`), or any other PNG tool that reads metadata.

### Timezones

`--tz` accepts friendly short names that map to DST-aware POSIX strings:

| You pass | Internally set to | Result label in May | Result label in December |
|----------|-------------------|---------------------|--------------------------|
| `EST` | `EST5EDT` | `EDT` | `EST` |
| `CST` | `CST6CDT` | `CDT` | `CST` |
| `MST` | `MST7MDT` | `MDT` | `MST` |
| `PST` | `PST8PDT` | `PDT` | `PST` |
| `UTC` | `UTC` | `UTC` | `UTC` |

Anything not in the table passes through verbatim, so power users can do `--tz America/New_York` or `--tz Etc/GMT+5`. Without `--tz`, the binary uses the system's local time.

### Example invocations

Times below are wall-clock from a Linux x86_64 container, multi-threaded across the host's CPU cores. Your machine will vary. Cost grows roughly linearly in pixel count and faster than linearly in `samples` (this is a branched path tracer).

```bash
# Quick noisy preview (~150 ms at 720, ~325 ms at 1080)
./code/Build/pcr-cornell -d 2 -s 4 -S 2

# Decent quality, visible noise (~4 min at 720, ~10 min at 1080)
./code/Build/pcr-cornell -d 4 -s 16 -S 4

# Picture-perfect, near-zero noise (~1 hr at 720, ~2-3 hr at 1080)
./code/Build/pcr-cornell -d 4 -s 256 -S 4 -w 1080 --tz EST -o ~/renders

# Non-square (1920x1080)
./code/Build/pcr-cornell -w 1920 --height 1080
```

Rule of thumb: doubling `--samples` roughly doubles render time and roughly halves visible noise (sqrt scaling). To go from "decent" to "clean" usually costs 4-16x more time.

View the resulting `.png` with any image viewer:

```bash
xdg-open Image/*.png   # Linux
open Image/*.png       # macOS
start Image/*.png      # Windows
```

## Scenes

Each scene lives in `code/Scenes/`. A scene defines its name, version, geometry (walls + spheres), and area light. To add a new scene:

1. Create `code/Scenes/<Name>.h` declaring `Scenes::make<Name>()` and a `<NAME>_VERSION` constexpr.
2. Create `code/Scenes/<Name>.cpp` implementing the factory.
3. Add the `.cpp` to `code/CMakeLists.txt`.
4. Register it in the `sceneRegistry()` map in `code/Main.cpp`.

Bump `<NAME>_VERSION` whenever you make a change to the scene that produces a visibly different render.
