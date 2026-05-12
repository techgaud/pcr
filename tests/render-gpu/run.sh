#!/usr/bin/env bash
#
# GPU render-diff regression tests.
#
# Parallel suite to tests/render/ but targeting the GPU CLI
# (physically-cringe-rendering-cli). Covers Metal backend coverage on
# macOS: megakernel vs wavefront A/B, fork vs terminate dispersion,
# spectral hero=4 vs hero=1, stratified-on regression marker for the
# wavefront-stratified latent bug.
#
# Goldens are baked locally on the user's M1 Ultra via
# regenerate-golden.sh and committed to the repo. CI on macos-latest
# rebuilds, re-renders, and tolerance-diffs against the goldens.
#
# Tolerance is looser than the CPU suite because Metal atomic-CAS
# float adds (perPixelAccum in spectralFork mode) and threadgroup
# scheduling are not bit-deterministic across runs. The dimming bug
# and FOV bug both blow past --abs-tol=2 / --pct-budget=0.05 trivially.
#
# Usage:
#   tests/render-gpu/run.sh                          # diff against goldens
#   BINARY=/path/to/cli tests/render-gpu/run.sh      # override binary path
#   PCR_TEST_SEED=99 tests/render-gpu/run.sh         # override seed
#   PCR_TEST_SKIP_IF_NO_GOLDENS=1 tests/render-gpu/run.sh  # exit 0 if golden/ empty
#
# Run from anywhere; the script resolves paths relative to itself.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default binary location. Mac builds drop here. Override via $BINARY
# for custom build directories.
DEFAULT_BINARY="$REPO_ROOT/code/Build/physically-cringe-rendering-cli"
BINARY="${BINARY:-$DEFAULT_BINARY}"

OUT_DIR="$SCRIPT_DIR/output"
GOLDEN_DIR="$SCRIPT_DIR/golden"
DIFF_PY="$REPO_ROOT/tests/render/diff.py"

# Golden-dir empty check. When goldens haven't been baked yet (initial
# scaffolding commit, or a CI run that landed before regenerate-golden.sh
# was executed locally), exit 0 with a clear note rather than fail.
# Set PCR_TEST_SKIP_IF_NO_GOLDENS=0 to force-fail in this case.
SKIP_IF_NO_GOLDENS="${PCR_TEST_SKIP_IF_NO_GOLDENS:-1}"
if [[ "$SKIP_IF_NO_GOLDENS" == "1" ]]; then
    if [[ ! -d "$GOLDEN_DIR" ]] || [[ -z "$(ls -A "$GOLDEN_DIR" 2>/dev/null | grep '\.png$' || true)" ]]; then
        echo "tests/render-gpu: no goldens present, skipping."
        echo "  bake goldens with: tests/render-gpu/regenerate-golden.sh"
        exit 0
    fi
fi

if [[ ! -x "$BINARY" ]]; then
    echo "error: GPU CLI not found or not executable: $BINARY" >&2
    echo "build first with:" >&2
    echo "  cmake -S $REPO_ROOT/code -B $REPO_ROOT/code/Build" >&2
    echo "  cmake --build $REPO_ROOT/code/Build -j --target physically-cringe-rendering-cli" >&2
    exit 2
fi

SEED="${PCR_TEST_SEED:-12345}"

mkdir -p "$OUT_DIR"

# Each tuple is "key|scene|mode|extra-flags". Empty extra-flags = none.
# Mode: "rgb" or "spectral" (drives whether --spectral is added).
#
# Coverage rationale:
#   * megakernel vs wavefront A/B for the same scene+mode, to catch
#     either side regressing on the other.
#   * fork vs terminate dispersion on cornell-glass spectral.
#   * hero=1 on glass spectral guards the latent
#     `tan(u.fov * 0.5f)` bug surviving in the megakernel hero=1
#     fallback paths (MetalRenderer.mm:1131, 1289, 1468). If hero=1
#     spectral output ever drifts visibly from hero=4, this catches it.
#   * --stratified on wavefront is the regression marker for the
#     wavefront-stratified latent bug. Today the golden looks
#     identical to non-stratified (kernels ignore the flag). The day
#     the kernels start honoring it, the golden updates and this
#     diff confirms the fix.
#   * cornell-bunny exercises BVH + mesh on both backends.
#   * cornell-spec exercises measured SPDs + Jakob upsampling path.
TUPLES=(
    "bunny-rgb-mega|cornell-bunny|rgb|--no-wavefront"
    "bunny-rgb-wf|cornell-bunny|rgb|--wavefront"
    "bunny-rgb-wf-stratified|cornell-bunny|rgb|--wavefront --stratified"
    "bunny-rgb-wf-mis-russian|cornell-bunny|rgb|--wavefront --mis --russian"
    "bunny-rgb-wf-adaptive|cornell-bunny|rgb|--wavefront --adaptive"

    "glass-rgb-mega|cornell-glass|rgb|--no-wavefront"
    "glass-rgb-wf|cornell-glass|rgb|--wavefront"
    "glass-spectral-hero4-mega|cornell-glass|spectral|--no-wavefront"
    "glass-spectral-hero4-wf-fork|cornell-glass|spectral|--wavefront"
    "glass-spectral-hero4-wf-terminate|cornell-glass|spectral|--wavefront --spectral-terminate"
    "glass-spectral-hero1-mega|cornell-glass|spectral|--no-wavefront --hero-samples 1"
    "glass-spectral-hero1-wf|cornell-glass|spectral|--wavefront --hero-samples 1"
    "glass-spectral-wf-mis-russian|cornell-glass|spectral|--wavefront --mis --russian"

    "spec-rgb-wf|cornell-spec|rgb|--wavefront"
    "spec-spectral-hero4-mega|cornell-spec|spectral|--no-wavefront"
    "spec-spectral-hero4-wf|cornell-spec|spectral|--wavefront"
    "spec-spectral-wf-denoise|cornell-spec|spectral|--wavefront --denoise"
)

# Render settings. Larger than the CPU suite's -w 120 -s 4 because GPU
# renders are cheap on Mac and dispersion features need samples to
# show up above the noise floor. ~7 sec per render on M1 Ultra, so
# the full 17-tuple suite bakes in ~2 min locally.
#
# Adaptive and OIDN deliberately excluded from common flags:
#   * adaptive: per-pixel early exit makes pass count vary per scene,
#     tested explicitly in one tuple.
#   * OIDN: non-deterministic neural denoiser, can't be diffed.
COMMON_FLAGS=(-d 4 -s 64 -S 2 -w 240 --seed "$SEED")

# Tolerance for the GPU diff. Looser than CPU's 1/255-1% defaults:
#   * --abs-tol 2 absorbs Metal's atomic-add-float CAS reordering
#     drift and threadgroup-scheduling FP-sum-order drift, both of
#     which are well below 2/255 in practice.
#   * --pct-budget 0.05 allows 5% of pixels to exceed the per-pixel
#     budget, which covers thin-feature edges (BVH-traced bunny
#     silhouette, dispersion caustic edges) where one-pixel jitter
#     accounts for ~1-3% of the image.
# The dimming bug drops hundreds of pixels by ~75% of full radiance,
# orders of magnitude above this budget. The FOV bug shifts every
# pixel by ~3x scale, catastrophic everywhere.
DIFF_ABS_TOL="${PCR_TEST_ABS_TOL:-2}"
DIFF_PCT_BUDGET="${PCR_TEST_PCT_BUDGET:-0.05}"

render_one() {
    local key="$1" scene="$2" mode="$3" extra="$4"
    local tmpdir
    tmpdir="$(mktemp -d)"

    local mode_flag=()
    if [[ "$mode" == "spectral" ]]; then
        mode_flag=(--spectral)
    fi

    # shellcheck disable=SC2206
    local extra_flags=($extra)

    if ! "$BINARY" "${COMMON_FLAGS[@]}" --scene "$scene" \
            "${mode_flag[@]}" "${extra_flags[@]}" \
            --scenes-dir "$REPO_ROOT/Scenes" \
            -o "$tmpdir" >/dev/null 2>&1; then
        echo "render failed: $key" >&2
        rm -rf "$tmpdir"
        return 1
    fi

    local produced
    produced="$(find "$tmpdir" -maxdepth 1 -name '*.png' | head -1)"
    if [[ -z "$produced" ]]; then
        echo "no PNG produced for $key in $tmpdir" >&2
        rm -rf "$tmpdir"
        return 1
    fi
    cp "$produced" "$OUT_DIR/$key.png"
    rm -rf "$tmpdir"
    return 0
}

PASS=0
FAIL=0
FAILED_KEYS=()

cd "$REPO_ROOT"

for tuple in "${TUPLES[@]}"; do
    IFS='|' read -r key scene mode extra <<< "$tuple"
    printf "[render-gpu] %-40s ... " "$key"
    if render_one "$key" "$scene" "$mode" "$extra"; then
        printf "rendered, "
        if [[ ! -f "$GOLDEN_DIR/$key.png" ]]; then
            printf "FAIL (no golden)\n"
            FAIL=$((FAIL + 1))
            FAILED_KEYS+=("$key (no golden)")
            continue
        fi
        if python3 "$DIFF_PY" "$GOLDEN_DIR/$key.png" "$OUT_DIR/$key.png" \
                --abs-tol "$DIFF_ABS_TOL" --pct-budget "$DIFF_PCT_BUDGET"; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            FAILED_KEYS+=("$key")
        fi
    else
        printf "RENDER ERROR\n"
        FAIL=$((FAIL + 1))
        FAILED_KEYS+=("$key (render error)")
    fi
done

echo
echo "=========================================="
echo "GPU render tests: $PASS passed, $FAIL failed"
if (( FAIL > 0 )); then
    echo "Failed tuples:"
    for k in "${FAILED_KEYS[@]}"; do
        echo "  - $k"
    done
    exit 1
fi
exit 0
