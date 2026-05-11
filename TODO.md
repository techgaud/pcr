# TODO

Deferred work, with enough context to pick up cold later.

## OIDN buffer-access bug (denoising silently disabled on Apple Silicon)

**Status:** confirmed bug. Every OIDN-enabled render on Apple Silicon prints `OIDN filter error: image data not accessible by the device, please use OIDNBuffer or device allocator for storage` to stderr and produces output WITHOUT the OIDN denoise applied. The renderer doesn't bail, it just skips the denoise step, so the user gets the raw HDR + tone-map (or the bilateral fallback when `useDenoise` is also on). Visually: more noise than the user expected on every OIDN-enabled render to date.

### Why deferred

Pre-existing bug that's been silently failing. Orthogonal to the wavefront work that surfaced it. Caught when v1.4.2's wavefront fallback path also tripped the same OIDN warning.

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

## Adaptive sampling in wavefront mode

**Status:** explicit limitation as of the wavefront baseline. `useAdaptive` is silently honored only when `useWavefront=false` (megakernel multi-pass). When `useWavefront=true`, the adaptive flag is ignored and a stderr warning fires.

### Why deferred

Megakernel adaptive keeps per-pixel Welford state in a device buffer, updated at AA-iteration boundaries (the last batch of each aaIdx fires the convergence check). Wavefront has no AA-iteration boundary per-pixel because rays from different pixels interleave through the shading kernels, so the existing Welford-update trigger doesn't map across cleanly.

### Two paths forward (pick one when revisiting)

1. **Run wavefront within each AA iteration.** One full wavefront pipeline run per aaIdx. Welford updates between passes, same per-pixel device buffer pattern as megakernel multi-pass. Preserves the existing adaptive semantics and convergence threshold. Cost: each pipeline run has 1/aaSamples the rays-in-flight that a one-shot wavefront run would have, so some of wavefront's ray-parallelism gain is given back. Probably still net positive for divergence-heavy scenes.

2. **Per-ray Welford in the ray state struct.** Each ray carries running mean/M2/count for its destination pixel. Convergence is checked when a ray terminates (no hit or depth exceeded), at which point a "this pixel is done" flag is set in a per-pixel mask. Subsequent primary-ray generation skips done pixels. More complex (Welford state now lives on the hot ray-state path), but preserves wavefront's one-shot pipeline.

### When to revisit

Right after the wavefront baseline ships and proves itself on non-adaptive renders. Top of the TODO list because Nate uses adaptive often in spectral mode where the convergence early-exit is significant.

## A/B test multi-level prefix-sum vs SIMD-group-batched atomic for queue compaction

**Status:** wavefront ships with SIMD-group-batched atomic queue compaction (per AMD GPUOpen "Fast Compaction with mbcnt", adapted to Metal's `simd_prefix_exclusive_sum()` intrinsic). One atomic per SIMD group instead of per thread, ~32x reduction in atomic traffic. Apple's WWDC22 "Scale compute workloads across Apple GPUs" specifically flags global atomics as a bottleneck on multi-core M-series GPUs, which makes the SIMD-group-batched pattern the safe-and-simple default.

### Why deferred

Multi-level prefix-sum compaction (full Blelloch scan, no atomics) is the production-grade choice for large ray counts. But it's ~3x the code (per-SIMD scan + per-threadgroup combine + cross-threadgroup combine + final scatter) and only meaningful when atomic contention actually dominates. At pcr's 1080² scale (~1.17M rays / 32 = ~36K atomics per SIMD-batched queue counter), contention should be bounded. Worth measuring before committing to the larger refactor.

### Implementation outline

1. Add a `--queue-compaction` mode flag to the GPU CLI: `simd-batched` (current default) or `prefix-sum`.
2. Wrap the queue-write path in each shading kernel behind a macro / function pointer so both modes share the same call site.
3. Implement the full Blelloch scan against `simd_prefix_inclusive_sum()` for per-SIMD, threadgroup memory + barrier for per-threadgroup combine, and a second dispatch for cross-threadgroup combine.
4. Render the same scene+settings with both modes via the GUI queue (batching feature from v1.4.1), compare wallclock + GPU power draw.
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
