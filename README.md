# pcr

A small CPU path tracer in C++20. Renders a hardcoded Cornell Box scene to a lossless PNG.

Multi-threaded, cosine-weighted hemisphere sampling for indirect light, explicit area-light sampling with shadow rays for direct light, soft shadows, emissive surfaces, Reinhard tone mapping.

## Requirements

A C++20 compiler and CMake 3.20+:

- **Linux**, GCC 10+ or Clang 14+
- **macOS**, Apple Clang 14+ (Xcode 14, macOS 12 or newer)
- **Windows**, MSVC from Visual Studio 2019 16.10+ (2022 recommended)

## Layout

Source code lives under `code/`. Renders are written to `Image/` next to it (default), so the source tree stays clean while past renders are preserved beside the code that produced them.

## Build

From the repo root:

```bash
cmake -S code -B code/Build
cmake --build code/Build --config Release
```

On Windows with Visual Studio, the binary lands at `code/Build/Release/pcr-cornell.exe`. On Linux and macOS it's `code/Build/pcr-cornell`.

## Run

Run from the repo root so the default `$PWD/Image/` resolves to `pcr/Image/`:

```bash
./code/Build/pcr-cornell -d 4 -s 16 -S 4
```

All flags are optional — defaults render a reasonable image. Pass `-o <dir>` to send renders elsewhere.

### Flags

| Short | Long | Default | Meaning |
|-------|------|---------|---------|
| `-d` | `--depth` | `4` | Max ray bounces |
| `-s` | `--samples` | `16` | Indirect-light samples per hit |
| `-S` | `--shadow` | `4` | Direct-light shadow rays per hit |
| | `--tz`, `--timezone` | system local | Timezone for filename, see below |
| `-o` | `--output` | `$PWD/Image` | Output directory |
| `-h` | `--help` | | Print usage and exit |

### Output

Each render writes to `<output-dir>/<timestamp>-d#-s#-shadow#-t<ms>.png`. Example:

```
20260506-143234-EDT-d2-s4-shadow2-t115.png
```

The timestamp is `YYYYMMDD-HHMMSS-<ZONE>`. Render time in milliseconds is appended as `t<ms>` and also printed to stdout.

### Timezones

`--tz` accepts friendly short names that map to DST-aware POSIX strings:

| You pass | Internally set to | Result label in May | Result label in December |
|----------|-------------------|---------------------|--------------------------|
| `EST` | `EST5EDT` | `EDT` | `EST` |
| `CST` | `CST6CDT` | `CDT` | `CST` |
| `MST` | `MST7MDT` | `MDT` | `MST` |
| `PST` | `PST8PDT` | `PDT` | `PST` |
| `UTC` | `UTC` | `UTC` | `UTC` |

Anything not in the table is passed through verbatim, so power users can do `--tz America/New_York` or `--tz Etc/GMT+5`. Without `--tz`, the binary uses the system's local time.

### Example invocations

```bash
# Quick noisy preview (~1 sec)
./code/Build/pcr-cornell -d 2 -s 4 -S 2

# Reasonable quality (~15 sec)
./code/Build/pcr-cornell -d 4 -s 16 -S 4

# Production-ish (~minutes)
./code/Build/pcr-cornell -d 5 -s 64 -S 8 --tz EST -o ~/renders
```

View the resulting `.png` with any image viewer:

```bash
xdg-open Image/*.png   # Linux
open Image/*.png       # macOS
start Image/*.png      # Windows
```

## Scene

The scene is defined in `code/Main.cpp`. Walls and the area light live in `Renderer::createWalls` and the `lightSource` Plane. To try a different scene, edit those and rebuild.
