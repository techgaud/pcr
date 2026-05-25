# TODO

Deferred work, with enough context to pick up cold later.

## Photon mapping: GPU SPPM ports + spectral integration

**Status:** Classical photon mapping ships on all four backends (sessions 1-4, commits `3b0209b` / `942887c` / `9371c48` / `add7d0a`). Plain progressive (ensemble averaging across N fresh photon shoots) ships on all four backends (sessions 5-6, commits `814db3c` / `5191178`). True SPPM (Hachisuka & Jensen 2009, per-pixel adaptive radius shrinkage) ships on **CPU + Metal megakernel only** (session 7, commit `b28e7dc`); Metal wavefront and OpenGL warn + fall back to plain progressive when `--photon-sppm` is requested.

Algorithm code lives in `code/Photon/`:
- `PhotonMap.h/.cpp` — Jensen hash-grid spatial index
- `PhotonShoot.h/.cpp` — host-side caustic photon shoot, reuses BVH + `dielectricBounce`
- `DensityEstimate.h` — CPU Jensen-1996 estimator (header-only)
- `GpuFlatten.h` — shared host→GPU table flattener (Metal + OpenGL)
- `Sppm.h` — `SppmPixel` (R, tauRGB, N) + `SppmDelta` (dtauRGB, M) PODs + `sppmUpdatePixel` host helper + `kSppmAlpha = 2/3`

CLI flags: `--photon-map`, `--photons N` (default 1M), `--photon-radius R` (default 0.05), `--photon-progressive`, `--photon-passes N` (default 8), `--photon-sppm`. PNG metadata: `PhotonMap` / `PhotonCount` / `PhotonRadius` / `PhotonProgressive` / `PhotonPasses` / `PhotonSppm` tEXt chunks, gated on what actually ran. Filename suffix `-photon-sppm` > `-photon-prog` > `-photon` for A/B-sort cleanliness.

Three remaining tasks:

### 1. Wavefront SPPM

**Why deferred:** SPPM's per-pixel visible-point semantics need a per-ray `firstDiffuse` bit so `wf_shade_diffuse` knows whether the current ray is at its primary's first diffuse hit (mutate per-pixel state) or a later one (skip photons entirely). Megakernel's `tracePath` tracks this as a local `bool` because its loop is per-pixel-per-pass. Wavefront splits the loop across kernels so the bit must live in the per-ray SoA.

**Implementation outline:**
1. Add bit 9 to the existing `rayState` packing (currently uses bits 0-8 for `depth` + `alive`; the comment at MetalRenderer.mm:284 calls out bits 9..31 as reserved for future per-ray flags). New helpers `stateFirstDiffuse(s)`, `stateClearFirstDiffuse(s)`, extend `statePack(depth, alive)` → `statePack(depth, alive, firstDiffuse)`.
2. `wf_raygen_primary` sets `firstDiffuse=true` on every primary ray it generates.
3. `wf_shade_diffuse` reads + clears the bit. When set AND `u.useCausticPhotonSppm != 0`, route to `sppmContributeAtVisiblePoint` (already exists in MSL); skip the classical density estimate. When clear, no photon contribution.
4. Other shading kernels (`wf_shade_mirror`, `wf_shade_glass`, `wf_shade_emissive`) pass `firstDiffuse` through unchanged via the existing `statePack` calls.
5. Bind `sppm` + `sppmDelta` at slots 11/12 in the wavefront `encodeShading` helper (currently passes nullptr for those fields in `wf_shade_diffuse`'s Scene init). Add `needsSppm` parameter to encodeShading matching the existing needsColor / needsPhotons / needsFullScene knob convention.
6. The host-side `sppm_pass_update` dispatch + final composite already work backend-agnostic (read `sppmBuf` back, apply Hachisuka equations); only the in-kernel write side changes.
7. Drop the `effectiveSppm && effectiveWavefront` warning + fallback in MetalRenderer.mm `render()`.

**Risk:** Low. The rayState packing change touches `statePack` call sites in `wf_raygen_primary` + each `wf_shade_*` kernel (the `statePack(depth, alive)` site near the end of each shading kernel where the per-ray state gets updated for the next bounce), but each touch is mechanical and the existing `stateAlive` / `stateDepth` helper pattern shows how to add `stateFirstDiffuse` cleanly. The Hachisuka math is shared from `Photon::Sppm.h`; the MSL kernels reuse the existing `sppmContributeAtVisiblePoint` helper unchanged.

**Validation:** A/B `--no-wavefront --photon-sppm` (megakernel SPPM, known working) vs `--photon-sppm` (wavefront SPPM) on cornell-glass at production-ish samples. Per-pixel output should match within Monte Carlo noise. PNG metadata `Architecture` differs (megakernel vs wavefront); everything else identical.

### 2. OpenGL SPPM

**Why deferred:** Same shape as Metal SPPM but in GLSL + OpenGL SSBOs. Two new SSBOs (Sppm + SppmDelta), a `sppm_pass_update` compute shader, host-side allocation + binding + dispatch after each progressive pass, final composite identical to Metal's.

**Implementation outline:**
1. Add GLSL `SppmPixel` + `SppmDelta` structs to the OpenglRenderer.cpp kernel string (next to the existing `Photon` + `PhotonCell` structs). SppmPixel is 5 floats (R, tauR, tauG, tauB, N) so it lays out naturally in std430; SppmDelta is 4 floats also fine. No need for the `float[3]`-instead-of-vec3 trick that `Photon` needed.
2. Add `uPhotonSppm` uniform (mirror of MSL `Uniforms.useCausticPhotonSppm`).
3. Add `sppmContributeAtVisiblePoint` GLSL function (line-for-line translation of the MSL version near the top of MetalRenderer.mm's kernel string).
4. Modify the GLSL `tracePath` (around OpenglRenderer.cpp line 858) to track a local `bool firstDiffuseSppm = (uPhotonSppm != 0);`. At the diffuse case, branch on SPPM vs classical the same way the megakernel `tracePath` does (see Renderer.cpp's diffuse case for the reference flow; MSL's is at MetalRenderer.mm tracePath).
5. Add a second compute shader `sppm_pass_update` (its own GL program). One thread per pixel; applies Hachisuka math to the SppmPixel SSBO using the SppmDelta SSBO; zeros the delta.
6. Host: allocate Sppm + SppmDelta SSBOs in `initGL()` (or per-render in `render()` like the photon SSBOs currently do), upload init state (R = photonRadius, everything else 0), bind at chosen SSBO indices (next free after the photon SSBOs at 8/9, so 10/11).
7. Dispatch `sppm_pass_update` after the existing tile-dispatch loop completes each progressive pass.
8. After all progressive passes, read back Sppm SSBO + apply the final composite per pixel (same code shape as Metal's at MetalRenderer.mm's "SPPM final composite" block).
9. Drop the OpenGL "warn + fallback" stub.

**Risk:** Medium. OpenGL needs two compute programs (the existing path-trace shader + a new sppm_pass_update shader), so the host-side single-`_program` assumption gets broken. Easy fix: keep them as two separate `GLuint` members.

### 3. Spectral photon mapping

**Why deferred:** Density estimate (CPU + Metal + OpenGL) currently uses RGB photon power. In `--spectral` mode each eye-path ray carries a hero-wavelength tuple; per-photon RGB power doesn't combine correctly with per-wavelength radiance. All photon-mapping toggles silently disable in spectral mode with a warning.

**Implementation outline:**
1. Per-photon wavelength: add `float lambda` to `Photon::Record` (becomes 40 bytes; update all consumers). Or alternatively: each photon carries 4 stratified wavelengths' powers (`float4 power` + the hero lambda tuple) — matches the eye-path's hero-N=4 convention but burns more memory per photon.
2. Photon shoot: per-emitted-photon, sample a wavelength uniformly in [400, 700] nm. Photon power scales by the emitter's per-wavelength emissive (using `Material::emissiveAt(lambda)`).
3. Density estimate: at a diffuse hit during the eye path, for each hero lambda the eye ray carries, sum photon contributions weighted by spectral reflectance evaluated at the photon's wavelength.
4. PBRT v4 chapter 16.4 has a clean spectral SPPM treatment worth following before implementing.

**Risk:** High. Spectral photon mapping has multiple defensible formulations and the literature is less unanimous than the RGB case. This is a research-flavored implementation, not a port. Worth bisecting first whether the user actually needs spectral caustics or if RGB caustics + spectral direct-lighting is good enough.

### When to revisit

In priority order: **(1) wavefront SPPM** (the user's production rendering path is wavefront, so SPPM is currently unavailable on production renders without `--no-wavefront`), **(2) OpenGL SPPM** (consistency for Linux/Windows users), **(3) spectral photon mapping** (algorithmic depth, lower practical priority since the user's caustic-heavy scenes render fine in RGB mode).

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
