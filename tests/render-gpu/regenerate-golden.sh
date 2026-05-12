#!/usr/bin/env bash
#
# Regenerate every GPU golden PNG from scratch.
#
# Runs the same matrix as run.sh, but copies outputs into
# tests/render-gpu/golden/ instead of output/. Use after an intentional
# GPU-output change (new feature, kernel tweak, bug fix that shifts
# pixels). Then eyeball the new goldens before committing.
#
# Bakes 17 renders at 240x240 -d 4 -s 64 -S 2 --seed 12345. Roughly
# 2 min wall on M1 Ultra, longer on other Apple GPUs.
#
# Usage:
#   tests/render-gpu/regenerate-golden.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_BINARY="$REPO_ROOT/code/Build/physically-cringe-rendering-cli"
BINARY="${BINARY:-$DEFAULT_BINARY}"

if [[ ! -x "$BINARY" ]]; then
    echo "error: GPU CLI not found or not executable: $BINARY" >&2
    echo "build first with:" >&2
    echo "  cmake -S $REPO_ROOT/code -B $REPO_ROOT/code/Build" >&2
    echo "  cmake --build $REPO_ROOT/code/Build -j --target physically-cringe-rendering-cli" >&2
    exit 2
fi

SEED="${PCR_TEST_SEED:-12345}"
GOLDEN_DIR="$SCRIPT_DIR/golden"
mkdir -p "$GOLDEN_DIR"

# Tuple list kept in sync with run.sh manually. If you add a tuple,
# add it both places. Same convention as tests/render/.
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

COMMON_FLAGS=(-d 4 -s 64 -S 2 -w 240 --seed "$SEED")

cd "$REPO_ROOT"

for tuple in "${TUPLES[@]}"; do
    IFS='|' read -r key scene mode extra <<< "$tuple"
    printf "[golden-gpu] %-40s ... " "$key"
    tmpdir="$(mktemp -d)"
    mode_flag=()
    if [[ "$mode" == "spectral" ]]; then
        mode_flag=(--spectral)
    fi
    # shellcheck disable=SC2206
    extra_flags=($extra)
    "$BINARY" "${COMMON_FLAGS[@]}" --scene "$scene" \
        "${mode_flag[@]}" "${extra_flags[@]}" \
        --scenes-dir "$REPO_ROOT/Scenes" \
        -o "$tmpdir" >/dev/null
    produced="$(find "$tmpdir" -maxdepth 1 -name '*.png' | head -1)"
    if [[ -z "$produced" ]]; then
        echo "FAIL (no PNG produced)" >&2
        rm -rf "$tmpdir"
        exit 1
    fi
    cp "$produced" "$GOLDEN_DIR/$key.png"
    rm -rf "$tmpdir"
    echo "regenerated"
done

echo
echo "All GPU goldens regenerated under $GOLDEN_DIR"
echo "Eyeball them, then 'git add tests/render-gpu/golden && git commit'."
