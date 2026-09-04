# Backends: parity, compile locality, and diagnosis

pcr runs the same path tracer on three backends across four binaries. This note
covers the disciplines that keep those backends interchangeable: the parity
rule the code holds to, where each backend actually compiles, how to work on
Metal safely from a machine that cannot build it, and how to diagnose a
brightness or parity mismatch between backends. The mechanics of each backend
(SSBOs, dispatch, wavefront kernels) are in `project-knowledge.md`.

## Where each backend compiles

- **CPU** (`Renderer.cpp`) compiles everywhere: GCC, Clang, and MSVC (Windows
  CI uses clang-cl across the whole matrix because MSVC has an internal-compiler
  bug on the spectral code).
- **OpenGL 4.3 compute** (`code/Gpu/Opengl/`) compiles on Windows and Linux.
  The GLSL kernel is an embedded string compiled at runtime when a real GL 4.3
  context is current. It can be exercised headlessly under Xvfb (for example
  `xvfb-run -a -s "-screen 0 256x256x24" ./code/Build/physically-cringe-rendering-cli ...`)
  on a machine with no display, which compiles the GLSL and runs the full path.
- **Metal compute** (`code/Gpu/Metal/`, `.mm` host plus an embedded MSL kernel)
  compiles only on macOS with the Metal toolchain. A non-Mac build excludes the
  Metal translation units, so a Linux or Windows build gives CPU and OpenGL
  coverage only. The MSL kernel string and the `.mm` host code get a real
  compiler only on the macOS build.

The macOS job in `.github/workflows/build.yml` is the compile gate for the
Metal backend. It runs a full `cmake --build` of the Metal targets on every
push to the Metal branch. When a change touches `.mm` or the MSL kernel string,
that specific check is the one that proves it compiles. A green Linux, Windows,
or tests run says nothing about whether Metal built, so the macOS check has to
be read on its own rather than trusted to the overall commit status.

## The parity invariant

Every feature is reachable from both front-ends and works on both backends:

- exposed in **both the CLI and the GUI**,
- working on **both the CPU and the GPU** backend (GPU is Metal on macOS,
  OpenGL on Windows and Linux),
- on **every supported OS**.

The front-ends share the renderer flag, so GUI parity usually follows once the
backend supports a feature, but the GUI control and its JSON persistence are
part of the feature rather than an afterthought. The two GPU kernels are
line-for-line ports of each other: the MSL kernel in `MetalRenderer.mm` mirrors
the GLSL kernel in `OpenglRenderer.cpp`, and a change to one side is a change to
the other.

Where a backend genuinely cannot support a feature, it is an explicit,
documented gap with a warn-and-fallback, not a silent divergence. Spectral SPPM
on the Metal wavefront is the standing example: it routes to the megakernel for
a structural reason, documented in `docs/photon-mapping.md`.

## Working on Metal from a machine that cannot build it

Because `.mm` does not compile on a non-Mac host, treat Metal edits made there
as potentially not even compilable, not merely untested. Two practices reduce
the risk:

- Self-review for the classes a compiler catches: argument-count and signature
  mismatches, use-before-declaration, and buffer-slot or binding collisions.
- Prefer signature changes that force every call site to update (for example
  changing a `statePack` arity) over additive helpers that a call site can
  silently skip.

The first real build on a Mac should run with Metal's runtime validation on:

```
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING=1 MTL_DEBUG_LAYER=1
```

Validation catches missing buffer bindings, out-of-bounds writes, and many
Metal undefined-behavior classes at roughly 5x render time, and it reports them
in the first lines of the log rather than as corrupted pixels many hours into a
render. A dispatched kernel that declares a buffer but is dispatched without
that binding, for instance, surfaces as a one-line validation error instead of
scale-dependent brightness corruption that only appears at large resolutions.
Reach for validation before bisecting a Metal render for hours.

## Diagnosing a brightness or parity mismatch between backends

When two backends disagree, or a render looks wrongly bright or dark, work in
this order.

1. **Diff the PNG `tEXt` chunks first.** The config keys (`Adaptive`, `OIDN`,
   `Architecture`, `WavefrontMode`, `WavefrontDispersion`, `BsdfMIS`,
   `Spectral`) show whether the two renders even ran the same math. The flag
   most likely to change the math rather than just the quality is `Adaptive`. A
   small Python `tEXt`-chunk dump reads these; exiftool is not required.

2. **Know Metal's two normalization paths.** They must not be confused.
   - Non-adaptive: the kernel writes a running HDR sum, and the CPU divides by
     `aaSamples * samples` at readback to recover the mean.
   - Adaptive: the kernel keeps per-pixel Welford state and writes the running
     mean straight to the output texture, so the CPU readback skips the divide.

   Mixing up which path is active is a common source of a 2x discrepancy.

3. **Compare raw linear values, not the 8-bit PNG.** Instrument the
   pre-tone-map framebuffer or HDR accumulator and compare means. The tone curve
   turns a clean linear 2x into a ratio that varies across the image, which
   hides a constant factor and makes a simple scale look like a complex bug.

4. **Check the direct-light cosine term.** The area-light NEE term uses the
   normalized light direction for `cosLight`, matching the `cosTheta` used one
   line above. Using the unnormalized vector-to-light scales the term by the
   distance to the light, which reads as a distance-weighted over-brightness
   (roughly 2x in a Cornell box). Both the RGB (`castRay`) and spectral
   (`castRaySpectral`) paths compute this term, so a change here touches both.

## Production renders

Production renders run on a separate Apple Silicon machine and land in
`Image/MacStudio/` for review. The first thing to check on any such render is
its PNG `tEXt` metadata, which records the full config that produced it, so a
render can always be traced back to the exact settings and backend behind it.
