#pragma once

// Shared GPU-renderer default values referenced by the renderers, the
// CLI, and the GUI. Centralized so flipping a default is a one-line edit
// instead of a scavenger hunt across header/CLI/Settings/JobConfig.
//
// Numbers here are conclusions from empirical A/B, not arbitrary picks.
// The git history alongside each constant captures when and why it
// changed; the const NAME is stable across reorderings so call sites
// don't churn.

namespace pcr
{

// Metal compute threadgroup shape for the path_trace_pass{,_adaptive}
// kernels. 8x8 = 64 threads = 2 SIMD groups (Apple's SIMD width is 32).
//
// Selected via v1.4.1 A/B across seven shapes (2x2, 4x4, 8x4, 8x8,
// 16x16, 32x8, 32x32) at multiple sample counts in both RGB and
// spectral modes. 8x8 won by 5-7% over the sub-SIMD probes (2x2/4x4/
// 8x4 all tied each other within ~1%, so the "more concurrent groups"
// hypothesis didn't beat 64-thread groups), and by 35-40% over the
// 256-thread and 1024-thread shapes (divergence within larger groups
// dominated). The win was consistent across RGB and spectral, which
// suggests it's a property of the dispatch / occupancy model rather
// than of any particular kernel branch.
//
// The GUI exposes a debug-mode tuning panel for re-running the A/B
// if the kernel shape changes meaningfully (e.g. wavefront refactor).
constexpr int kDefaultThreadgroupX = 8;
constexpr int kDefaultThreadgroupY = 8;

// Architecture default. true = wavefront (rays rebatched per-material
// between bounces, divergence-free shading kernels), false = megakernel
// (single-kernel pipeline, the v1.4.0+ baseline). Set to true based on
// v1.4.2 A/B measurements showing wavefront-1spp beats megakernel by
// ~25% on cornell-spec at 1080^2 Picture, with no quality regression.
// Configurations that wavefront doesn't yet support (spectral, multi-
// sample-per-pass + adaptive) fall back to megakernel automatically
// with a stderr warning, so this default Just Works for the common
// cases without surprising the user on the unsupported edges.
constexpr bool kDefaultUseWavefront = true;

// When useWavefront=true, whether to dispatch multiple samples per
// pipeline run (true) or one sample per pipeline run (false). 1spp
// is the safer default: smaller per-render working set, narrower
// performance variance across scene sizes. Multi-spp won by only
// ~2-3% in the first A/B and has scaling concerns at 4K+ where the
// working set grows linearly with samplesPerPass.
constexpr bool kDefaultWavefrontMultiSample = false;

// When useWavefront=true AND useSpectral=true, controls how the glass
// shading kernel handles a dispersive refraction (cauchyB > 0):
//   false (default): terminate the three secondary hero wavelengths
//     at the first dispersive refraction and continue the hero
//     wavelength scalar, with a 4x energy compensation on the survivor
//     so heroLambdasXYZ's 1/N normalization stays unbiased. Cheaper
//     per-bounce but post-glass paths sample one wavelength instead of
//     four, so dispersive caustics (e.g. cornell-glass) converge slower.
//   true: fork the ray into four monochromatic sub-paths, each carrying
//     its own wavelength's IOR and refraction direction. Matches what
//     megakernel's tracePathSpectral does for cauchyB > 0. 4x more
//     post-glass rays per primary ray, ~4x the SoA buffer footprint,
//     better dispersion convergence at the cost of implementation
//     complexity (variable-output append queue + writeback rework).
// Default off so existing renders keep their wallclock profile until
// the fork mode is dialed in via A/B.
constexpr bool kDefaultSpectralFork = false;

} // namespace pcr
