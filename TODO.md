# TODO

Deferred work, with enough context to pick up cold later.

## Make the GPU binary portable (single-file Windows .exe)

**Status:** spiked, paused. The infrastructure is on the `portable-gpu` branch and covers ~80% of the path; the remaining 20% is the hard part (TBB) and is what stopped the spike.

### Goal

Ship `physically-cringe-rendering.exe` as a true single-file Windows binary that runs by double-click without a `lib/` folder beside it. The CPU binaries can stay on the bundled-zip release path; only the GPU binary needs this treatment.

### Why this is hard

OIDN is the dependency that forces the issue. It's distributed only as DLLs, has a plugin architecture that `dlopen`s device DLLs at runtime, and depends transitively on Intel oneTBB which Intel discourages static-linking. Closing the dependency graph for a single-file build means handling all three: OIDN itself, OIDN's device plugin, and TBB.

### What's on the `portable-gpu` branch (overcome)

- `PCR_OIDN_STATIC` cmake option that switches OIDN from the prebuilt-binary FetchContent path to a source-build FetchContent path. Default OFF so the existing flow is unchanged.
- OIDN configured for the static path with `OIDN_STATIC_LIB=ON`, `OIDN_DEVICE_CPU=ON`, all other devices off (CUDA / HIP / SYCL), `OIDN_APPS=OFF`, `OIDN_FILTER_RT=ON`, `OIDN_FILTER_RTLIGHTMAP=OFF`.
- `pcr_link_oidn` CMake helper short-circuits the runtime DLL deploy and the `/DELAYLOAD` linker flag when `PCR_OIDN_STATIC=ON`, since both are meaningless for a static link.
- New `windows-portable` CI job that:
  - Downloads ISPC v1.23.0 (OIDN's SIMD kernel compiler), unpacks it, finds `ispc.exe` recursively (layout-agnostic), prepends to `$GITHUB_PATH`.
  - Configures cmake with `-DPCR_OIDN_STATIC=ON` in a separate `code/Build-Portable` directory so it doesn't conflict with the dynamic-OIDN build.
  - Builds only the `physically-cringe-rendering` target.
  - Stages the .exe as `pcr-windows-x64-portable` artifact.
- Two smoke-test gates before artifact upload:
  - File-based: no DLLs alongside the .exe, binary is at least 30 MB (the static OIDN weights blob is ~50 MB; a normal-sized binary would mean OIDN didn't actually link in).
  - Import-table: `dumpbin /dependents` (located via vswhere, no third-party action needed) checked against a forbidden list of `OpenImageDenoise*`, `tbb*`, `MSVCP*`, `VCRUNTIME*`, `pi_level_zero`, `sycl`. Any match fails the build.
- Release job updated to pick up the portable artifact and rename it `physically-cringe-rendering-${VER}-portable-windows-x64.exe` alongside the bundled variant.

### Where the spike stopped (hurdles to overcome)

1. **`CMAKE_BUILD_TYPE` not in cache under Visual Studio.** OIDN's `oidn_common.cmake` does `set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS …)`, but VS is a multi-config generator and `CMAKE_BUILD_TYPE` is empty / not in cache by default. Fix: in our top-level cmake, when `PCR_OIDN_STATIC=ON`, force `set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)` early so OIDN's `set_property` finds something. One-line fix.
2. **TBB not found.** OIDN expects an external TBB via `find_package(TBB)`. The runner has no system TBB on Windows, no apt package, no easy `vcpkg install`. Two paths:
   - **Easy path**: download a prebuilt oneTBB binary distribution in the CI step (similar to ISPC), set `TBB_ROOT` before cmake configure. TBB stays dynamic — the resulting binary still imports `tbb12.dll`. Not single-file portable, but a "1 .exe + 1 DLL" release which is much closer to the goal.
   - **Hard path**: `FetchContent_Declare` oneTBB from source, build with `BUILD_SHARED_LIBS=OFF`, propagate to OIDN. Static TBB requires:
     - Saving and restoring `BUILD_SHARED_LIBS` around the TBB fetch so it doesn't bleed into other deps (GLFW, OIDN's own internals).
     - Either installing the built TBB to a staging directory and pointing `TBB_ROOT` at the install layout (which OIDN's `FindTBB.cmake` expects), or patching OIDN to use TBB's exported `TBB::tbb` target directly.
     - Accepting the runtime caveats Intel publishes about static TBB (global-object lifetime, atexit ordering, etc.) — for a short-running render-and-exit process these are probably benign, but they're real.
3. **`/MT` static CRT propagation to OIDN's sub-build.** Our top-level cmake forces `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` for MSVC, but FetchContent sub-projects can override that via their own `CACHE FORCE` settings. If OIDN's sub-build ends up `/MD`, our binary would have a hidden `MSVCP140.dll` dependency. The dumpbin smoke test catches this; the fix is to ensure the CRT setting is in the cache before any sub-fetch (already mostly the case) and to re-`set` it after each fetch as a defensive measure.
4. **Binary size budget.** OIDN's pretrained weights blob is ~50 MB and lives inside the static lib. Our binary goes from ~5 MB to ~50–80 MB. The smoke test's "must be ≥ 30 MB" gate is the floor; the upper end depends on TBB-static (adds another ~10 MB) and the C++ runtime statically linked (~1–2 MB).
5. **OIDN Apache-2.0 NOTICE.** Static redistribution requires bundling the upstream NOTICE text. Not a build-system issue but a release-correctness one. Either bake the text into a `--license` flag in the binary, or ship a `NOTICE.txt` next to the .exe (which arguably defeats single-file purity, since now there are two files). Realistic answer: embed the NOTICE as a string in the binary, print on `--about` or in the GUI's About dialog.

### What we'd actually learn by finishing this

The static-linking fight is the actual lesson. Specifically: closing a transitive C++ dependency graph (OIDN → TBB → CRT), how `/MT` vs `/MD` ABI mismatch surfaces in mystery crashes, how to inspect import tables (`dumpbin /dependents`, `objdump -p`, `otool -L`), and why "single-file portable" is genuinely hard for any non-trivial native app. Real software-engineering knowledge, transferable beyond this project.

### Decision needed before resuming

Pick a target on the static-vs-dynamic-TBB axis:

- **Hard target (static TBB, true single-file)**: 2–3 more CI iterations to get green. Real risk of runtime breakage from TBB's static-link caveats.
- **Soft target (dynamic TBB, 1 .exe + tbb12.dll)**: probably one CI iteration. Not portable but much closer.
- **Abandon, ship bundled zip**: the portable spike's infrastructure stays on this branch as documentation; the release goes back to the universal Windows convention (download zip, unzip, run).

The branch is preserved so picking back up doesn't require re-discovery. Just check out `portable-gpu`, decide which target, and continue.

## Add a `cmake --install` flow for system-wide / user-local install

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
