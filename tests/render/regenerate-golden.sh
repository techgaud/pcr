#!/usr/bin/env bash
#
# Regenerate every golden PNG from scratch.
#
# Runs the same render matrix as run.sh, but copies outputs into
# tests/render/golden/ instead of tests/render/output/. Use this after
# an intentional render-output change (new feature, threshold tweak,
# bug fix that shifts pixel values) to refresh the goldens. Then
# eyeball the new goldens before committing.
#
# Usage:
#   tests/render/regenerate-golden.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_BINARY="$REPO_ROOT/code/Build/frank-based-rendering-cli"
BINARY="${BINARY:-$DEFAULT_BINARY}"

if [[ ! -x "$BINARY" ]]; then
    echo "error: binary not found or not executable: $BINARY" >&2
    exit 2
fi

SEED="${PCR_TEST_SEED:-12345}"
GOLDEN_DIR="$SCRIPT_DIR/golden"
mkdir -p "$GOLDEN_DIR"

# Source the tuple list from run.sh by re-defining it here. Kept in
# sync manually; if you add a tuple, add it both places.
TUPLES=(
    "cornell-rgb|cornell|rgb|"
    "cornell-spectral|cornell|spectral|"
    "cornell-spectral-lut|cornell|spectral|--lut"
    "cornell-bunny-rgb|cornell-bunny|rgb|"
    "cornell-bunny-spectral|cornell-bunny|spectral|"
    "cornell-glass-rgb|cornell-glass|rgb|"
    "cornell-glass-spectral|cornell-glass|spectral|"
    "cornell-spheres-rgb|cornell-spheres|rgb|"
    "cornell-large-light-rgb|cornell-large-light|rgb|"
    "cornell-spec-rgb|cornell-spec|rgb|"
    "cornell-rgb-denoise|cornell|rgb|--denoise"
    "cornell-rgb-aces|cornell|rgb|--aces"
    "cornell-rgb-mis-russian-stratified|cornell|rgb|--mis --russian --stratified"
    "cornell-rgb-aa4|cornell|rgb|--aa --aa-samples=4"
    "cornell-rgb-adaptive-aa8|cornell|rgb|--adaptive --aa --aa-samples=8"
)

COMMON_FLAGS=(-d 2 -s 4 -S 2 -w 120 --seed "$SEED")

cd "$REPO_ROOT"

for tuple in "${TUPLES[@]}"; do
    IFS='|' read -r key scene mode extra <<< "$tuple"
    printf "[golden] %-40s ... " "$key"
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
echo "All goldens regenerated under $GOLDEN_DIR"
echo "Eyeball them, then 'git add tests/render/golden && git commit'."
