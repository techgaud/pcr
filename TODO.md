# TODO

Deferred work, with enough context to pick up cold later.

## Make the GPU binary portable (single-file Windows .exe)

**Status:** not started. Today the binary lives at `Build/frank-based-rendering-cli` and you invoke it with the path. To run it as just `frank-based-rendering-cli` from any directory, add a CMake `install()` rule and `cmake --install`.

### Why deferred

For solo dev on this machine the explicit path (`Build/frank-based-rendering-cli`) is fine and avoids polluting `/usr/local/bin`. Worth doing when:

- You want to type `frank-based-rendering-cli ...` from any cwd without remembering the path
- You distribute releases (Homebrew formula, Debian package, GitHub Releases binaries)
- A second person wants to use it on their machine without learning the source layout

### CMake change

Add to `CMakeLists.txt` after the `add_executable` block:

```cmake
include(GNUInstallDirs)
install(TARGETS frank-based-rendering-cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
```

`GNUInstallDirs` provides `CMAKE_INSTALL_BINDIR`, which is `bin` on Unix. This is what GNU/Debian package conventions expect.

### Usage after that

After building, install with one of:

```bash
# System-wide (Linux/macOS), requires sudo
sudo cmake --install Build

# User-local (Linux/macOS), no sudo
cmake --install Build --prefix ~/.local

# Custom location
cmake --install Build --prefix /opt/pcr
```

Default install paths:

- Linux/macOS without `--prefix`: `/usr/local/bin/frank-based-rendering-cli`
- With `--prefix ~/.local`: `~/.local/bin/frank-based-rendering-cli`
- Windows without `--prefix`: `C:\Program Files\frank-based-rendering-cli\bin\frank-based-rendering-cli.exe`, requires admin

### PATH considerations

After installing, the install dir needs to be on `$PATH` for "just type `frank-based-rendering-cli`" to work.

- **Linux:** `/usr/local/bin` is on `$PATH` by default. `~/.local/bin` may not be. Add to `~/.bashrc` or `~/.zshrc` if needed:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```
- **macOS:** Same as Linux. `~/.local/bin` is not on `$PATH` by default on most setups.
- **Windows:** Default install prefix `Program Files\frank-based-rendering-cli\bin` is *not* on `PATH`. Either edit System Properties → Environment Variables, or install with `--prefix %LOCALAPPDATA%\pcr` and add that bin dir to user `PATH`.

### Uninstall

CMake writes an install manifest at install time:

```bash
xargs rm -v < Build/install_manifest.txt
```

Or just delete the binary by hand. CMake doesn't generate a `cmake --uninstall` by default.

### Testing the install rule

After adding the `install()` line:

```bash
cmake -S . -B Build
cmake --build Build
cmake --install Build --prefix /tmp/pcr-test
ls /tmp/pcr-test/bin/   # should contain frank-based-rendering-cli
/tmp/pcr-test/bin/frank-based-rendering-cli --help
```

If that works, real install with `cmake --install Build` (with appropriate sudo / admin / `--prefix`).

### Related future work

- A `cpack` config to produce `.deb`/`.rpm`/`.pkg` packages instead of just installing files. Useful if pcr ever gets distributed.
- A GitHub Actions workflow that runs `cmake --install` and uploads the binary to GitHub Releases. Useful if pcr ever gets multiple users on different platforms.

## Fourth binary: GPU CLI (`physically-cringe-rendering-cli`)

**Status:** not started. v1.4.0 ships GPU on all three OSes via `physically-cringe-rendering` (the GUI binary), but there's no headless GPU path. Scripted / batch / CI renders that want the GPU still have to launch the GUI.

### Why deferred

The Metal port made GPU compute structurally feasible without a display server (Metal doesn't need a window the way OpenGL does), but the existing `frank-based-rendering-cli` is intentionally CPU-only so it can run on headless homelab servers without dragging GLFW + X11 deps. Adding `--gpu` to the existing CLI would force those deps onto every CLI build, regressing the headless-server fitness.

A fourth binary keeps the separation clean: CPU CLI stays lean, GPU CLI is its own target with its own dep graph.

### Why a separate binary, not just a flag

Two of three options were considered and rejected (see chat history during the v1.4.0 Metal port for the full analysis):

- **`--gpu` flag on existing CLI:** drags GLFW onto Linux/Windows CLI builds even when unused. Bad for headless servers.
- **Mac-only `--gpu` flag:** smaller, but makes the CLI behavior asymmetric across platforms (works on Mac, errors on Win/Linux). Future-confusing.
- **Fourth binary (this one):** symmetric across all three OSes. Win/Linux pull in GLFW as expected for the GPU backend; Mac uses Metal cleanly. CPU CLI stays untouched.

### Implementation sketch

1. New target in `code/CMakeLists.txt` named `physically-cringe-rendering-cli`. Sources: `Cli/Main.cpp` (with one new code path that constructs `GpuRenderer` instead of `Renderer` when invoked) + `${PCR_GPU_BACKEND_SRC}` (already conditionally Opengl/.cpp or Metal/.mm) + the shared renderer sources + the GUI's GLFW dep on non-Apple.
2. Refactor `Cli/Main.cpp` to dispatch to either CPU or GPU renderer based on a build-time `PCR_USE_GPU` define (mirrors what the GUI binaries already do via the same define).
3. On Win/Linux: at startup, `glfwInit()` + `glfwCreateWindow(1, 1, ..., GLFW_VISIBLE=FALSE)` to satisfy the OpenGL backend's shared-context requirement. On Apple: nothing extra; Metal stands alone.
4. CMake conditional: link Foundation + Metal frameworks on Apple (mirror what the GPU GUI does); link GLFW + OpenGL on Win/Linux.
5. CI: stage the new binary in all three OS jobs, plus add it to the release-job rename matrix as `physically-cringe-rendering-cli-vX.Y.Z-<os>-<arch>`. Release goes from 8 individual binary assets to 11.

### Why `physically-cringe-rendering-cli` as the name

Keeps the silly-naming convention from v1.0.0. `frank-based-rendering-cli` is the CPU one; `physically-cringe-rendering-cli` is the GPU one. Symmetric with the GUI pair.

### Use cases this unlocks

- Scripted batch renders on Mac Studio (the obvious motivator: M1 Ultra GPU is much faster than the 20-core CPU for path tracing, and the GUI is overkill for "render this list of scenes overnight")
- CI render-diff goldens regenerated on the GPU, faster turnaround when adding a new render-test tuple
- Homelab GPU rendering jobs without remote-X / VNC into the GUI

## Spatial chopping in the multi-pass dispatch (Apple-only, very-high-res)

**Status:** not started. Multi-pass dispatch handles cleanly up to ~8K square; 16K square (Apple Silicon's MTLTexture hard limit) trips Apple's compute watchdog because per-dispatch ops don't fit in the ~3-sec budget even at the floor of one sample per pixel per pass.

### Why deferred

For the workloads pcr's audience (one) actually runs, 4K-8K is the hero-render sweet spot. 16K is "because we can" rather than "because the output is meaningfully different." Implementing spatial chopping is real work for a corner case nobody's asked for.

### Numbers

Multi-pass per-pass work scales as `pixels * samples_per_pass * shadow * depth`. The watchdog target is ~9B ops per dispatch (calibrated against the M1 Ultra's measured ~3.15 G ops/sec saturated throughput). At Picture-class settings (shadow=32, depth=6, ~192 ops/pixel/sample):

- 1080² (1.17M pixels): samplesPerPass = 9B / (1.17M × 192) = 40, well under the budget
- 4K (16.7M pixels): samplesPerPass = 2, ~12.8B ops/dispatch, still safe
- 8K (67M pixels): samplesPerPass = 1 (clamped), ~12.9B ops/dispatch, ~4 sec — borderline
- 16K (268M pixels): samplesPerPass = 1 (clamped), ~52B ops/dispatch, ~16 sec — very likely killed by the watchdog

The fix is to dispatch a SUBREGION of the image each pass instead of the full image when the full-image dispatch would exceed budget at samplesPerPass=1. Treat the budget as "a slice of work that's both spatially and sample-axis bounded."

### Implementation sketch

1. Compute pixelsPerDispatch as `kTargetOpsPerDispatch / opsPerPixelPerSample`. If pixelsPerDispatch >= total pixels, current full-image multi-pass continues to work.
2. If pixelsPerDispatch < total pixels, also chop the image into spatial regions of approximately pixelsPerDispatch each. Total dispatches = `aaSamples * sampleBatches * spatialTiles`.
3. Per-pass uniforms grow: alongside `aaIdx / sampleStart / sampleCount`, add `xOffset / xEnd / yOffset / yEnd` for the spatial region.
4. The kernel already supports spatial bounds (the strip path uses them). Same Uniforms struct can carry both.
5. Pass 0's "clobber the texture" branch needs to apply only at the per-pixel level (not per-dispatch); easier to just zero-fill the output texture once at render start with a tiny clear kernel (which we'd wanted anyway).

### Risk

Adds ordering complexity to the pass loop. Currently pass index uniquely determines (aaIdx, sampleStart). With spatial chopping, pass index → (aaIdx, sampleStart, spatialRegion). The mapping is straightforward but easy to off-by-one. Test fixture: render Picture at exactly 16K and confirm the output PNG matches (within rounding) a 16K render done by tiling 4× of an 8K render.

### When to revisit

When someone asks for 16K renders. Until then, the slider letting users PICK 16K is the cosmetic accessibility; the cmd-buffer-error logging that already exists will surface the watchdog kill explicitly to stderr if anyone tries it. Not silently broken, just not supported.
