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

} // namespace pcr
