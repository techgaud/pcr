#!/usr/bin/env bash
#
# Render-diff regression tests.
#
# For each (scene, mode, techniques) tuple in the matrix below, render the
# scene at low quality with --seed for determinism, then PNG-diff against
# the golden image. Aggregate pass/fail counts and exit non-zero if any
# tuple regresses.
#
# Usage:
#   tests/render/run.sh                            # diff against goldens
#   BINARY=/path/to/cli tests/render/run.sh        # override binary path
#   PCR_TEST_SEED=99 tests/render/run.sh           # override seed (rarely needed)
#
# Run from anywhere; the script resolves paths relative to itself.

set -uo pipefail

# Resolve repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default binary location is the standard CMake build output. Override
# via $BINARY for non-standard builds.
DEFAULT_BINARY="$REPO_ROOT/code/Build/frank-based-rendering-cli"
BINARY="${BINARY:-$DEFAULT_BINARY}"

if [[ ! -x "$BINARY" ]]; then
    echo "error: binary not found or not executable: $BINARY" >&2
    echo "build first with:" >&2
    echo "  cmake -S $REPO_ROOT/code -B $REPO_ROOT/code/Build" >&2
    echo "  cmake --build $REPO_ROOT/code/Build -j --target frank-based-rendering-cli" >&2
    exit 2
fi

# Fixed seed across all renders. Goldens are tied to this seed; changing
# it requires regenerating every golden.
SEED="${PCR_TEST_SEED:-12345}"

OUT_DIR="$SCRIPT_DIR/output"
GOLDEN_DIR="$SCRIPT_DIR/golden"
DIFF_PY="$SCRIPT_DIR/diff.py"

mkdir -p "$OUT_DIR"

# Each tuple is "key|scene|mode|extra-flags". Empty extra-flags = none.
# Mode: "rgb" or "spectral" (drives whether --spectral is added).
# Key is the deterministic basename used for output and golden lookup.
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

# Common quality flags. Low enough to run all 15 tuples in well under
# a minute single-threaded; tight enough on noise that 1/255 pixel diff
# is meaningful for regression detection given the fixed seed.
COMMON_FLAGS=(-d 2 -s 4 -S 2 -w 120 --seed "$SEED")

# Render one tuple. Returns the path to the produced PNG copied into
# OUT_DIR with the canonical name. Uses --output to a per-tuple temp
# directory so we can find the binary's auto-named PNG, then copies
# it to OUT_DIR/<key>.png.
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
            -o "$tmpdir" >/dev/null; then
        echo "render failed: $key" >&2
        rm -rf "$tmpdir"
        return 1
    fi

    # Binary writes one .png with an auto-generated name. Pick it up
    # and copy to OUT_DIR with the canonical key.
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

# Run every tuple, collect results.
PASS=0
FAIL=0
FAILED_KEYS=()

# Repo root as cwd so the binary's relative-path scene fallbacks work
# even if we drop --scenes-dir later. tinyobjloader resolves mesh paths
# relative to the JSON's location, not cwd.
cd "$REPO_ROOT"

for tuple in "${TUPLES[@]}"; do
    IFS='|' read -r key scene mode extra <<< "$tuple"
    printf "[render] %-40s ... " "$key"
    if render_one "$key" "$scene" "$mode" "$extra"; then
        printf "rendered, "
        if [[ ! -f "$GOLDEN_DIR/$key.png" ]]; then
            printf "FAIL (no golden)\n"
            FAIL=$((FAIL + 1))
            FAILED_KEYS+=("$key (no golden)")
            continue
        fi
        if python3 "$DIFF_PY" "$GOLDEN_DIR/$key.png" "$OUT_DIR/$key.png"; then
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
echo "Render tests: $PASS passed, $FAIL failed"
if (( FAIL > 0 )); then
    echo "Failed tuples:"
    for k in "${FAILED_KEYS[@]}"; do
        echo "  - $k"
    done
    exit 1
fi
exit 0
