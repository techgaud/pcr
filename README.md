# pcr

A small path tracer in C++20. Ships with a Cornell Box scene; the architecture is designed for adding more scenes.

Cosine-weighted hemisphere sampling for indirect light, explicit area-light sampling with shadow rays for direct light, soft shadows, emissive surfaces, Reinhard tone mapping. Output is lossless PNG with embedded tEXt metadata describing scene name, version, and render parameters.

Three binaries ship from one codebase:

| Binary | Backend | Use |
|--------|---------|-----|
| `frank-based-rendering-cli` | CPU | CLI for headless / scripting / homelab servers |
| `frank-based-rendering` | CPU | Desktop GUI (sliders, render button, preview, history) |
| `physically-cringe-rendering` | GPU | Same GUI, on the GPU. **OpenGL 4.3 compute** on Windows/Linux, **Metal compute** on Apple Silicon Macs. |

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

This produces three executables:

- **`frank-based-rendering-cli`** — CLI, CPU only (headless, scriptable)
- **`frank-based-rendering`** — desktop GUI, CPU
- **`physically-cringe-rendering`** — desktop GUI, GPU. OpenGL 4.3 compute on Windows/Linux, Metal compute on Apple Silicon.

On Windows with Visual Studio they land at `code/Build/Release/<name>.exe`. On Linux and macOS they're at `code/Build/<name>`.

The GUI target uses GLFW (fetched + built statically by CMake on first configure, ~30 sec) and Dear ImGui (vendored under `code/Includes/imgui/`). On Linux you'll need X11 dev headers installed:

```bash
sudo apt install libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev pkg-config
```

Windows and macOS need no additional packages beyond Visual Studio / Xcode CLT.

### Don't want to install a toolchain?

Tagged releases at `github.com/techgaud/pcr/releases` ship pre-built binaries for all three platforms. Each release carries a per-OS bundle (`pcr-vX.Y.Z-<os>-<arch>.zip`) plus individual binary downloads for users who already have the data folders from a previous version.

**Recommended:** download the per-OS bundle ZIP — it's the binary plus the `Scenes/`, `models/`, `spd/`, `luts/`, and `lib/` (OIDN runtime) folders pre-arranged next to it, which the binary needs to find at startup.

#### Windows

1. Download `pcr-vX.Y.Z-windows-x64.zip`, extract anywhere
2. Open the extracted folder, double-click `physically-cringe-rendering.exe` (GPU GUI), `frank-based-rendering.exe` (CPU GUI), or run `frank-based-rendering-cli.exe` from a terminal
3. Windows SmartScreen will warn that the binary is unsigned the first time. Click *More info* → *Run anyway*. (Or right-click the .exe → Properties → check *Unblock* → OK before launching.)

#### macOS (Apple Silicon)

1. Download `pcr-vX.Y.Z-macos-arm64.zip`, extract anywhere
2. macOS Gatekeeper will block downloaded unsigned binaries. Easiest fix from a terminal in the extracted folder:
   ```bash
   xattr -dr com.apple.quarantine .
   ```
   (Or right-click each binary → *Open* → *Open* in the Gatekeeper dialog, individually.)
3. Double-click `physically-cringe-rendering` (GPU GUI, uses Metal) or `frank-based-rendering` (CPU GUI). Or run `./frank-based-rendering-cli` from terminal.

Apple Silicon GPU rendering shipped in v1.4.0; pre-1.4.0 macOS builds are CPU-only.

#### Linux

1. Download `pcr-vX.Y.Z-linux-x64.zip`, extract anywhere
2. From a terminal in the extracted folder:
   ```bash
   chmod +x physically-cringe-rendering frank-based-rendering frank-based-rendering-cli  # if needed
   ./physically-cringe-rendering
   ```
3. The GPU binary needs an OpenGL 4.3 driver (most recent Intel/AMD/NVIDIA setups have this). The CPU binaries have no graphics requirement.

### Building releases yourself

Push a tag (`git tag v1.0.0 && git push origin v1.0.0`) and the workflow auto-publishes a GitHub Release with all three OS bundles + individual binaries attached.

## Run (GUI)

Double-click whichever GUI binary you want, or from a shell:

```bash
./code/Build/frank-based-rendering        # CPU
./code/Build/physically-cringe-rendering  # GPU (OpenGL 4.3 on Win/Linux, Metal on macOS)
```

Both share the same UI; only the path-tracing backend differs. Settings persist to `<binary-name>.json` next to the executable, so the two GUIs keep separate state.

Sliders set every parameter the CLI exposes (depth, samples, shadow, width, square checkbox + height, scene, timezone, output dir). Click **Render** and the path tracer runs on a worker thread; progress is reported as rows-completed-of-total. **Cancel** mid-render works (sub-second responsiveness). When the render finishes the resulting PNG is loaded and shown in the same window. Settings persist to `<binary-name>.json` next to the executable, so each binary keeps its own state.

## Run (CLI)

Run from the repo root so the default `$PWD/Image/` resolves to `pcr/Image/`:

```bash
./code/Build/frank-based-rendering-cli -d 4 -s 16 -S 4
```

All flags are optional — defaults render a 720x720 Cornell Box at decent quality. Pass `-o <dir>` to send renders elsewhere.

### Flags

| Short | Long | Default | Meaning |
|-------|------|---------|---------|
| | `--scene` | `cornell` | Which scene to render |
| `-d` | `--depth` | `4` | Max ray bounces (GUI cap: 8) |
| `-s` | `--samples` | `16` | Indirect-light samples per hit (GUI cap: 4096) |
| `-S` | `--shadow` | `4` | Direct-light shadow rays per hit (GUI cap: 64) |
| `-w` | `--width` | `720` | Output width in pixels |
| | `--height` | (matches `-w`) | Output height; if omitted, output is square |
| | `--tz`, `--timezone` | system local | Timezone for filename, see below |
| `-o` | `--output` | `$PWD/Image` | Output directory |
| | `--scenes-dir` | (none) | Extra directory to search for `*.json` scene files. May be passed multiple times. Searched before the default `$PWD/Scenes` and `<binary-dir>/Scenes`. |
| | `--list-scenes` | | Print all discovered scenes (hardcoded + JSON) with name, version, source, then exit. |
| | `--denoise` | off | 5x5 cross-bilateral filter on output, reduces speckle |
| | `--mis` | off | MIS on direct lighting (partial impl, light-side only) |
| | `--russian` | off | Russian roulette path termination at depth >= 1 |
| | `--stratified` | off | Jittered stratified samples for first indirect bounce |
| `-h` | `--help` | | Print usage and exit |

The four "techniques" flags are off by default to keep behavior matching v1.0.0; flip on to compare. In the GUI they appear as checkboxes under "Techniques" and persist in the per-binary JSON. PNG metadata records which techniques were active for any given render so renders stay diff-able.

### Scenes

Scenes can be hardcoded in C++ (always available, baked into the binary) or authored as JSON files dropped into a `Scenes/` directory. Currently shipped:

| Name | Version | Source |
|------|---------|--------|
| `cornell`             | `1.1.0` | `Scenes/cornell.json` (1.0.0 hardcoded fallback if JSON missing) |
| `cornell-spheres`     | `1.0.0` | `Scenes/cornell-spheres.json` (1.0.0 hardcoded fallback) |
| `cornell-large-light` | `1.0.0` | `Scenes/cornell-large-light.json` (1.0.0 hardcoded fallback) |
| `cornell-bunny`       | `1.0.0` | `Scenes/cornell-bunny.json` (Stanford bunny in cornell box) |

Each scene declares its own version number that bumps when the scene definition changes. The scene name and version are baked into the output filename and PNG metadata, so renders stay traceable to the exact scene revision that produced them.

Run with `--list-scenes` to see what the binary discovers in your environment, including which file (or "hardcoded") backs each one.

### Scene file search path

When the binary starts it merges scenes from:

1. Any directory passed via `--scenes-dir` (CLI; may be passed multiple times)
2. `$PWD/Scenes/`
3. The directory containing the running binary, plus `Scenes/`
4. Hardcoded C++ scenes baked into the binary (fallback for any name not found in steps 1-3)

JSON wins over hardcoded on name collision. Within JSON dirs, earlier dirs in the search order win.

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
| `Software` | `frank-based-rendering-cli` |
| `Scene` | `cornell` |
| `SceneVersion` | `1.0.0` |
| `CreationTime` | `20260506-143234-EDT` |
| `Depth` | `4` |
| `Samples` | `16` |
| `ShadowSamples` | `4` |
| `Width` | `720` |
| `Height` | `720` |
| `RenderTimeMs` | `12345` |
| `Denoise` | `0` or `1` |
| `MIS` | `0` or `1` |
| `Russian` | `0` or `1` |
| `Stratified` | `0` or `1` |

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
./code/Build/frank-based-rendering-cli -d 2 -s 4 -S 2

# Decent quality, visible noise (~4 min at 720, ~10 min at 1080)
./code/Build/frank-based-rendering-cli -d 4 -s 16 -S 4

# Picture-perfect, near-zero noise (~1 hr at 720, ~2-3 hr at 1080)
./code/Build/frank-based-rendering-cli -d 4 -s 256 -S 4 -w 1080 --tz EST -o ~/renders

# Non-square (1920x1080)
./code/Build/frank-based-rendering-cli -w 1920 --height 1080
```

Rule of thumb: doubling `--samples` roughly doubles render time and roughly halves visible noise (sqrt scaling). To go from "decent" to "clean" usually costs 4-16x more time.

View the resulting `.png` with any image viewer:

```bash
xdg-open Image/*.png   # Linux
open Image/*.png       # macOS
start Image/*.png      # Windows
```

## Scenes

A scene defines its name, version, camera, geometry (walls + spheres), and area light. There are two ways to add one.

### JSON (preferred)

Drop a `*.json` file in `Scenes/` at the repo root (or anywhere `--scenes-dir` points). Schema is `1.0`; comments (`//`) are supported. Smallest possible scene:

```jsonc
{
  "schema": "1.0",
  "name": "my-scene",
  "version": "1.0.0",

  "camera": { "position": [0, 0, 0], "fov": 65 },

  "materials": {
    "white": { "albedo": [0.74, 0.74, 0.64] },
    "light": { "emissive": [80, 68, 48] }
  },

  "primitives": [
    { "type": "plane", "name": "ceiling",
      "origin": [-2, 2, -6], "u": [4, 0, 0], "v": [0, 0, 7],
      "material": "white" },

    { "type": "sphere", "center": [0, -1, -4.5], "radius": 0.75,
      "material": "white" },

    // Exactly one primitive must be flagged as the area light.
    { "type": "plane", "light": true,
      "origin": [-0.375, 2, -4.25], "u": [0.75, 0, 0], "v": [0, 0, 0.4],
      "material": "light" }
  ]
}
```

Primitive types: `sphere`, `plane`, `triangle`, `mesh`. Triangles take `v0`/`v1`/`v2` plus optional `n0`/`n1`/`n2` per-vertex normals for smooth shading (omit for flat shading from the geometric normal). Meshes load OBJ files — see the "Mesh import" subsection below.

`light: true` is allowed on planes, triangles, and meshes. Multiple lights coexist; the renderer picks one per shadow sample proportional to surface area. Both CPU and GPU binaries support all three light kinds. Materials are a named registry referenced by name from each primitive.

### Mesh import

To drop a mesh into a scene, point a `mesh` primitive at an OBJ file:

```jsonc
{ "type": "mesh",
  "file": "../models/stanford-bunny.obj",
  "scale": 8,
  "rotation": [0, 180, 0],   // Euler XYZ degrees
  "position": [0, -2, -4.5],
  "material": "bunny-tan"    // optional; if set, overrides MTL materials
}
```

`file` is resolved relative to the JSON scene's directory. `scale` is either a scalar (uniform) or `[sx, sy, sz]`. `rotation` is Euler XYZ in degrees, applied X then Y then Z. `position` is applied after scale + rotation. If the OBJ ships vertex normals, smooth shading is on by default; pass `"smooth": false` to force flat shading.

The OBJ's `mtllib` directive is honored. MTL materials enter the registry under their MTL-given names; the JSON `materials` block can shadow them. A per-mesh `material` override clobbers all MTL materials for that mesh. Texture maps in MTL (`map_Kd` etc.) are ignored with a warning today — texture support is planned.

The repo ships the Stanford bunny (~69k triangles) at `models/stanford-bunny.obj`, demoed by `cornell-bunny.json`. Renders use a BVH for log(N) triangle intersection cost, so even high-poly meshes are tractable. Bump `version` whenever geometry changes meaningfully — it shows up in the output filename and PNG metadata, so renders stay traceable.

The CLI rescans on every invocation; the GUI rescans on combo open. Use `--list-scenes` to see what the binary picks up.

### C++ (only for hardcoded fallbacks)

Use this only if the scene must work without `Scenes/` next to the binary (e.g. cornell, which ships as a 1.0.0 fallback so a standalone binary download still has at least one scene to render). To add one:

1. Create `code/Scenes/<Name>.h` declaring `Scenes::make<Name>()` and a `<NAME>_VERSION` constexpr.
2. Create `code/Scenes/<Name>.cpp` implementing the factory (populate `SceneData` including `camera`).
3. Add the `.cpp` to `PCR_RENDERER_SOURCES` in `code/CMakeLists.txt`.
4. Register it in `kHardcoded[]` inside `code/Scenes/SceneDiscovery.cpp`.

Bump `<NAME>_VERSION` whenever the scene changes visibly. JSON files of the same name override the hardcoded version.
