# TODO

Deferred work, with enough context to pick up cold later.

## Photon mapping: GPU SPPM ports + spectral integration — COMPLETE

**Status (2026-05-31): full backend + mode parity reached.** Classical
(Jensen 1996), plain progressive (Hachisuka 2008-style ensemble), and true
SPPM (Hachisuka & Jensen 2009, α=2/3 per-pixel radius shrink) all ship on
**all four backends** — CPU, Metal megakernel, Metal wavefront, OpenGL.
Spectral (dispersion) caustics ship on all four too. CLI + GUI both expose
the flags via the shared renderer.

Verification: CPU + OpenGL validated. The Metal commits are written but
**UNVERIFIED** (the Linux dev box can't compile `.mm`) — they gate on the
macOS CI build + a Mac A/B. Hand off Metal work UNVERIFIED and watch the
macOS CI check; first Mac render with `MTL_SHADER_VALIDATION=1
MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING=1 MTL_DEBUG_LAYER=1`.

Key commits: OpenGL SPPM `89a33aa`; spectral CPU `ba79d0d`+`f2be894`;
spectral OpenGL `e5b5c62`; spectral Metal megakernel `7d09cc5`; spectral
Metal wavefront `0593230`. (Wavefront SPPM `75afdc7`, verified on Mac.)

**One deliberate non-parity:** spectral **SPPM** on Metal routes to the
megakernel (a narrow force `--no-wavefront` gate scoped to spectral+SPPM).
Per-pixel SPPM needs one visible point per pixel carrying all four hero
wavelengths together (= the megakernel hero-4 single-path shape); the
wavefront's monochromatic spectral fork splits a glass-piercing ray into
sub-rays that share a pixelIdx (would race the atomic-free `sppmDelta`) and,
in terminate mode, drop 3/4 of the spectrum. Classical/progressive spectral
run natively on the wavefront (forks write their own ray-slot color +
atomic fork-scatter, so no race). See `[[project-photon-mapping]]` memory.

Algorithm code: `code/Photon/` (`PhotonMap`, `PhotonShoot`,
`DensityEstimate.h`, `GpuFlatten.h`, `Sppm.h`). CLI flags: `--photon-map`,
`--photons`, `--photon-radius`, `--photon-progressive`, `--photon-passes`,
`--photon-sppm`. PNG `tEXt`: `PhotonMap`/`PhotonCount`/`PhotonRadius`/
`PhotonProgressive`/`PhotonPasses`/`PhotonSppm` (+ `Spectral`), gated on what
ran. Filename suffix `-photon-sppm` > `-photon-prog` > `-photon`.

**Remaining is validation only** (Mac A/B for the two spectral Metal
commits + the megakernel spectral commit). No code work outstanding on this
feature.

## Indirect dispatch for wavefront shading kernels (Metal)

**Status:** not started. The wavefront shading kernels (wf_shade_emissive, _mirror, _glass, _diffuse) currently dispatch at the worst-case thread count (`baseRayCount` plus the spectral-fork allocation of `3 * baseRayCount`) and bounds-check inside the kernel against the actual queueLen. In spectral-fork mode the worst case is 4x baseRayCount, but actual queue lengths after glass forks dispatch are typically 20-40% of that because forks only spawn on the first dispersive refraction. The wasted threads do bounds-check work and return without contributing.

### Why deferred

Needs A/B vs the current worst-case-with-bounds-check approach. Indirect dispatch reads thread counts from a GPU buffer at dispatch time, eliminating wasted threads but adding an indirect-args buffer barrier between wf_compact_by_material and each shading kernel. M-series barrier cost might eat the savings; the only way to know is measure.

### Implementation outline

1. Allocate a per-pass MTLBuffer of size `4 * sizeof(MTLDispatchThreadgroupsIndirectArguments)` (12 bytes per dispatch: groupsX, groupsY, groupsZ). One slot per material type.
2. New MSL kernel `wf_compute_indirect_args` dispatched once after `wf_compact_by_material`. Reads `queueCounters[matType]`, divides by `linearThreadsPerGroup`, ceiling-up, writes the result to the indirect buffer.
3. Replace `[enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg]` for each shading kernel with `[enc dispatchThreadgroupsWithIndirectBuffer:indirectBuf indirectBufferOffset:matType*12 threadsPerThreadgroup:tpg]`.
4. Add `useIndirectDispatch` to MetalRenderer + GUI Debug menu (matches the threadgroup-debug placement convention) + `--indirect-dispatch` CLI flag.
5. PNG metadata: `IndirectDispatch: 0|1` when wavefront ran.

### Risk

Medium. The indirect-args buffer barrier between the compute kernel and the dispatch is implicit in MTL command-buffer ordering, but the dispatch needs to read the args via the buffer not via host. Easy to get the offset arithmetic wrong. Validation: A/B on cornell-spec spectral wavefront fork at production samples, look for wallclock improvement >5% before promoting.

### When to revisit

Next perf session. Sized correctly this is the highest-impact Mac wavefront perf win on the shelf today.

### Starting points

- MetalRenderer.mm:4034 (current per-pass dispatch loop)
- MetalRenderer.mm:1836 (wf_compact_by_material; needs a sibling kernel right after it)
- MetalRenderer.mm:3334 (shading pipeline build sites; the indirect dispatch use mirrors the existing dispatch call shape)

## BSDF-side MIS in CPU + OpenGL + Metal megakernel

**Status:** wavefront shipped. The wavefront path now ships the full
balance heuristic (commits c151d0f / 885f17b / 6bcf433 on the mac branch).
CPU and OpenGL and Metal megakernel are still light-side-only.

### Why deferred

Only wavefront sees production use on Mac per the v1.5.0 A/B (~25%
faster than megakernel at cornell-class). CPU is for CI tests, OpenGL
is Win/Linux dev only. Bringing them to parity is consistency work,
not perf-impact work.

### Implementation outline

- CPU `castRay` (Renderer.cpp): track `lastBsdfPdf` as a parameter
  through recursion; weight emissive return paths.
- OpenGL GLSL `tracePath` and `tracePathSpectral`: carry one extra
  local across the iterative bounce loop.
- Metal megakernel `tracePath` / `tracePathSpectral` / `tracePathSpectralSingle`:
  same pattern as OpenGL.

The wavefront version (see code/Gpu/Metal/MetalRenderer.mm wf_shade_emissive
around the `w_bsdf` computation) is the working reference.

## OIDN buffer-access bug (denoising silently disabled on Apple Silicon)

**Status:** confirmed bug. Every OIDN-enabled render on Apple Silicon prints `OIDN filter error: image data not accessible by the device, please use OIDNBuffer or device allocator for storage` to stderr and produces output WITHOUT the OIDN denoise applied. The renderer doesn't bail, it just skips the denoise step, so the user gets the raw HDR + tone-map (or the bilateral fallback when `useDenoise` is also on). Visually: more noise than the user expected on every OIDN-enabled render to date.

### Why deferred

Pre-existing bug that's been silently failing. Orthogonal to the wavefront work that surfaced it. Caught when v1.5.0's wavefront fallback path also tripped the same OIDN warning.

### Root cause

`code/OidnDenoise.cpp::denoise()` calls `filter.setImage(...)` with raw `std::vector<Vec3f>::data()` pointers. OIDN 2.x rejects this when the device backend can directly access GPU memory (Metal on Apple Silicon). The fix is to allocate device-side buffers via `device.newBuffer()` and bind those instead of raw CPU pointers; on Apple Silicon's unified memory the buffer is in the same physical RAM as the std::vector so the cost is effectively just a memcpy through the buffer's `write()` / `read()` API.

### Implementation outline

```cpp
size_t bytes = (size_t)width * (size_t)height * 3 * sizeof(float);
oidn::BufferRef colorBuf  = device.newBuffer(bytes);
oidn::BufferRef albedoBuf, normalBuf;
colorBuf.write(0, bytes, color.data());
filter.setImage("color",  colorBuf,  oidn::Format::Float3, w, h);
if (haveAlbedo) {
    albedoBuf = device.newBuffer(bytes);
    albedoBuf.write(0, bytes, albedo.data());
    filter.setImage("albedo", albedoBuf, oidn::Format::Float3, w, h);
}
if (haveNormal) {
    normalBuf = device.newBuffer(bytes);
    normalBuf.write(0, bytes, normal.data());
    filter.setImage("normal", normalBuf, oidn::Format::Float3, w, h);
}
filter.setImage("output", colorBuf, oidn::Format::Float3, w, h);
// ... commit + execute ...
colorBuf.read(0, bytes, color.data());
```

Plus error-check `device.getError()` after each `newBuffer` since out-of-memory or capability-mismatch could surface there.

### Risk

Low. Single-file change in `OidnDenoise.cpp`. Existing CPU-only OIDN device (selectable via env var or explicit device-type construction) would also fix it but loses the Apple Silicon GPU acceleration that OIDN ships with by default; the buffer-route is the right answer.

### Validation

Render `cornell` at d=4 / s=64 (low samples to maximize noise so denoise is visible), with `useOIDN=true` and `useDenoise=false`, compare wallclock + image quality before / after. After the fix the output should be visibly smoother and stderr should show no OIDN error.

## Adaptive sampling in wavefront multi-sample-per-pass mode

**Status:** partial. Adaptive sampling works in `useWavefront + wavefrontMultiSample=false` (1-spp wavefront). The writeback kernel accumulates per-pass contributions into a per-pixel staging slot and, on the last sample of each aaIdx, finalizes the iteration's mean, folds it into a Welford accumulator, checks for convergence, and writes the running mean to the output texture. Done pixels short-circuit in ray-gen and the rest of the pipeline.

`useWavefront + useAdaptive + wavefrontMultiSample=true` still falls back to megakernel with a stderr warning.

### Why deferred for multi-spp

The 1-spp variant works because each pipeline run is exactly one sample for each pixel, so the writeback can sum them across passes into staging and divide by `u.samples` at the aaIdx boundary cleanly. Multi-spp packs `samplesPerPass` samples per pixel per pass, which means the writeback's per-pixel-per-aaIdx sum needs to thread through both a within-pass sample-axis loop AND the cross-pass staging accumulator. Doable, just more bookkeeping.

### Implementation outline (for multi-spp)

The 1-spp writeback already does:
```
sumThisPass = sum of color[s*pixelCount + pixelIdx] for s in 0..sampleCount
staging += sumThisPass
if batchEndOfAa: aaMean = staging / samples; Welford update; reset staging
```

For multi-spp: `sumThisPass` is already a multi-sample sum (sampleCount > 1). The math should already work - `staging` becomes a sum of all samples in the aaIdx across all batches, divide by `u.samples` at boundary. Need to test that the 1-spp writeback handles sampleCount > 1 correctly when fallback gate is lifted. Probably 5-line change.

### When to revisit

When multi-spp wavefront becomes worth tuning. Current numbers (~2-3% win for multi-spp over 1-spp) suggest 1-spp+adaptive may be the natural daily-driver anyway, since adaptive's early-exit savings stack on top of wavefront's divergence elimination and 1-spp+adaptive has the smaller working set.

## A/B test multi-level prefix-sum vs SIMD-group-batched atomic for queue compaction

**Status:** wavefront ships with SIMD-group-batched atomic queue compaction (per AMD GPUOpen "Fast Compaction with mbcnt", adapted to Metal's `simd_prefix_exclusive_sum()` intrinsic). One atomic per SIMD group instead of per thread, ~32x reduction in atomic traffic. Apple's WWDC22 "Scale compute workloads across Apple GPUs" specifically flags global atomics as a bottleneck on multi-core M-series GPUs, which makes the SIMD-group-batched pattern the safe-and-simple default.

### Why deferred

Multi-level prefix-sum compaction (full Blelloch scan, no atomics) is the production-grade choice for large ray counts. But it's ~3x the code (per-SIMD scan + per-threadgroup combine + cross-threadgroup combine + final scatter) and only meaningful when atomic contention actually dominates. At pcr's 1080² scale (~1.17M rays / 32 = ~36K atomics per SIMD-batched queue counter), contention should be bounded. Worth measuring before committing to the larger refactor.

### Implementation outline

1. Add a `--queue-compaction` mode flag to the GPU CLI: `simd-batched` (current default) or `prefix-sum`.
2. Wrap the queue-write path in each shading kernel behind a macro / function pointer so both modes share the same call site.
3. Implement the full Blelloch scan against `simd_prefix_inclusive_sum()` for per-SIMD, threadgroup memory + barrier for per-threadgroup combine, and a second dispatch for cross-threadgroup combine.
4. Render the same scene+settings with both modes via the GUI queue (batching feature from v1.5.0), compare wallclock + GPU power draw.
5. If prefix-sum is consistently faster by >5%, promote it to default. Otherwise document the result and remove the prefix-sum kernels.

### When to revisit

After wavefront ships and we have actual M1 Ultra wallclock numbers for the SIMD-batched baseline. Likely materializes only if pcr starts pushing 4K+ renders where queue scales are 10-100x larger and atomic contention may show up.

## macOS signed + notarized .dmg distribution

**Status:** not started. The current macOS release flow ships a per-OS bundle ZIP. macOS Gatekeeper blocks any binary downloaded from the internet that isn't notarized; users have to `xattr -dr com.apple.quarantine` the extracted folder before launch. A signed and notarized .dmg removes that friction.

### Why deferred

Apple Developer Program membership is $99/year, which is not worth paying when the audience for pcr is one person who can also run `xattr` from a terminal. The first-launch friction is small and explained in the release body's install instructions.

### When to revisit

When pcr has an audience beyond Nate. Or when Nate wants the polished "double-click .dmg, drag to Applications" install flow.

### Implementation outline

1. Apple Developer Program membership.
2. Generate a Developer ID Application certificate, install it on the build machine.
3. CMake `BUNDLE` generator: build `physically-cringe-rendering.app` and `frank-based-rendering.app` as proper `.app` bundles (Info.plist, executable in Contents/MacOS, OIDN dylib in Contents/Frameworks). The CLI stays a plain command-line binary.
4. `codesign --deep --sign "Developer ID Application: <name>" --options runtime <app>` to sign the bundles + nested dylibs.
5. `xcrun notarytool submit <dmg> --apple-id <id> --team-id <id> --password <app-password> --wait` to notarize.
6. `xcrun stapler staple <dmg>` to embed the notarization ticket.
7. Probably easiest as a separate workflow step running on the macos-latest CI runner with the cert + app password stored as repo secrets.

Cost: ~1-2 hours of CMake + GitHub Actions YAML once the Apple Developer setup is done.

## MetalRT hardware ray tracing (M3+ only)

**Status:** not started. M1 Ultra (Nate's current Mac Studio) doesn't expose MetalRT, so this is gated by future hardware. M3 and later have hardware-accelerated ray tracing via `MTLAccelerationStructure` + `MTLIntersectionFunctionTable`.

### Why deferred

Wrong hardware. Even on M3+, this is a pure enhancement: pcr's BVH traversal works fine, MetalRT would just make it faster. Not a correctness fix.

### What it would replace

The MSL kernel's `intersectBvh()` function (stack-based BVH traversal in private memory, ~70 lines of MSL) gets replaced by a `MTLIntersectionQuery` against an `MTLAccelerationStructure` built from the scene's triangles at upload time. The host side builds the acceleration structure with a `MTLAccelerationStructureCommandEncoder`; the kernel does `query.intersect()` and reads back hit info.

### When to revisit

When Nate has an M3-or-later Mac. Until then, M1 Ultra at saturated multi-pass throughput is plenty fast for everything pcr does.

### Risk

MetalRT only accelerates triangles, not spheres or planes. We'd need to keep the existing intersection code for non-triangle primitives and fall back to MetalRT only for the BVH traversal. Mixed dispatch adds kernel complexity. Worth measuring before committing, for cornell-class scenes with few triangles the speedup may not justify the maintenance cost.
