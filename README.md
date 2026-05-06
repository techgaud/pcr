# pcr

A small CPU path tracer in C++20. Renders a hardcoded Cornell Box scene to a lossless PNG.

Multi-threaded, cosine-weighted hemisphere sampling for indirect light, explicit area-light sampling with shadow rays for direct light, soft shadows, emissive surfaces, Reinhard tone mapping.

## Requirements

A C++20 compiler and CMake 3.20+:

- **Linux**, GCC 10+ or Clang 14+
- **macOS**, Apple Clang 14+ (Xcode 14, macOS 12 or newer)
- **Windows**, MSVC from Visual Studio 2019 16.10+ (2022 recommended)

## Build

```bash
cmake -S . -B Build
cmake --build Build --config Release
```

On Windows with Visual Studio, the second step produces `Build/Release/pcr.exe`. On Linux and macOS it produces `Build/pcr`.

## Run

The renderer writes its output under `<cwd>/../Image/`. Run from the `Build/` directory so images land at the repo root:

```bash
cd Build
./pcr            # Linux, macOS
.\Release\pcr    # Windows
```

It will prompt for three integers on stdin:

1. `depth`, max ray bounces (try 4)
2. `samples`, indirect-lighting samples per pixel hit (try 16)
3. `shadowSamples`, direct-light shadow rays per pixel hit (try 4)

Higher values give a cleaner image and a slower render.

The output is a lossless 712x712 PNG, named `<timestamp>-d#-s#-shadow#-t<ms>.png`, where the timestamp is local-zone `YYYYMMDD-HHMMSS-<ZONE>` and the trailing `t` is render time in milliseconds. Example: `20260506-143234-EDT-d2-s4-shadow2-t115.png`. The renderer also prints the elapsed time to stdout at the end of the run.

## Scene

The scene is defined in `Main.cpp`. Walls and the area light live in `Renderer::createWalls` and the `lightSource` Plane. To try a different scene, edit those and rebuild.
