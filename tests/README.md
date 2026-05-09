# tests

Automated regression suite for `pcr`. Two flavors:

- **Unit tests** under `tests/unit/`. Plain `int main()` files registered
  with CTest. Cover Spectrum, CIE color, Optics, RGBToSpectrum (Jakob
  upsampler + LUT), and BVH (build + traversal). Full ctest run is ~4 s
  on Linux; the LUT build in `test_rgbtospec` dominates at ~4 s.
- **Render-diff tests** under `tests/render/`. Drive `frank-based-rendering-cli`
  with a fixed seed across 15 (scene, mode, technique) tuples and
  pixel-compare against golden PNGs.

CI runs both on every push to `main`, `dev`, `gui`, `gpu`, `test-suite`.
The spectral branch skips the tests job (it has its own narrower GPU-only
matrix).

## Running locally

```bash
# Build the CLI and the test binaries together.
cmake -S code -B code/Build -DPCR_BUILD_TESTS=ON
cmake --build code/Build -j

# Unit tests via ctest.
ctest --test-dir code/Build --output-on-failure

# Render-diff tests. Builds nothing extra; expects code/Build/frank-based-rendering-cli.
tests/render/run.sh
```

`tests/render/run.sh` honors two env vars:

- `BINARY=/path/to/frank-based-rendering-cli` to point at a non-default
  build location.
- `PCR_TEST_SEED=42` to use a different seed (rarely useful: the
  goldens are tied to the default seed of 12345).

## How the render-diff tests stay deterministic

The CLI takes `--seed <N>` (added on the test-suite branch). When set,
`NumGen` initializes its per-thread PRNG from the seed and Renderer drops
to single-threaded execution. Same machine + same seed = byte-identical
PNG output across runs.

Goldens are generated on Linux (GCC + libstdc++) and CI also runs the
diffs on Linux. Cross-OS bit-exact reproduction is **not** expected —
floating-point rounding differs subtly across compilers and CPUs. If we
ever want render-diff coverage on Windows or macOS, we'll need per-OS
golden subdirectories; for now Linux-only is the cheap and effective
choice.

## Updating goldens

Render-diff goldens need refreshing whenever a deliberate change shifts
pixel values: a new feature, a bug fix that affects shading, a tone-map
parameter tweak, etc.

```bash
# Build the CLI first if you haven't.
cmake --build code/Build -j --target frank-based-rendering-cli

# Re-render every tuple, overwriting tests/render/golden/*.png.
tests/render/regenerate-golden.sh

# Eyeball the new goldens before committing. Cornell box should still
# look like Cornell box: red wall, green wall, light, geometry. If
# anything looks wrong, fix the underlying change before regenerating.
git add tests/render/golden
git diff --stat tests/render/golden
git commit
```

If only some tuples changed, you can run the regen script and then
`git add` only those. The script always rewrites all 15.

## Render matrix

| Key | Scene | Mode | Extra flags |
|---|---|---|---|
| cornell-rgb | cornell | rgb | (none) |
| cornell-spectral | cornell | spectral | (none) |
| cornell-spectral-lut | cornell | spectral | --lut |
| cornell-bunny-rgb | cornell-bunny | rgb | (none) |
| cornell-bunny-spectral | cornell-bunny | spectral | (none) |
| cornell-glass-rgb | cornell-glass | rgb | (none) |
| cornell-glass-spectral | cornell-glass | spectral | rainbow caustic safety net |
| cornell-spheres-rgb | cornell-spheres | rgb | (none) |
| cornell-large-light-rgb | cornell-large-light | rgb | (none) |
| cornell-spec-rgb | cornell-spec | rgb | (none) |
| cornell-rgb-denoise | cornell | rgb | --denoise |
| cornell-rgb-aces | cornell | rgb | --aces |
| cornell-rgb-mis-russian-stratified | cornell | rgb | --mis --russian --stratified |
| cornell-rgb-aa4 | cornell | rgb | --aa --aa-samples=4 |
| cornell-rgb-adaptive-aa8 | cornell | rgb | --adaptive --aa --aa-samples=8 |

`--oidn` is deliberately omitted: Intel Open Image Denoise is
non-deterministic across hardware and driver versions. The manual
visual-pass step in the pre-merge checklist covers it.

All renders use `-d 2 -s 4 -S 2 -w 120` for speed. Increasing quality
would tighten the noise floor but stretch CI runtime; the current
combination is fast (~10 s for all 15 tuples single-threaded) and
sensitive enough to catch any real regression because `--seed` makes
the result bit-exact.

## Diff threshold

`diff.py` defaults to "max per-channel absolute difference > 1/255 in
more than 1% of pixels" = fail. Change with `--abs-tol` and
`--pct-budget`. The default is tight because `--seed` produces
bit-exact output on the same machine; any drift past 1/255 is a real
shift, not Monte Carlo noise.

When CI fails, the `render-test-output` artifact contains the produced
PNGs from that run. Download, eyeball next to `tests/render/golden/`,
and either fix the regression or regenerate the goldens (in that order).

## Adding a new test

**Unit test.** Add a `.cpp` file under `tests/unit/`. Add its base name
to the `foreach(t IN ITEMS ...)` list in `tests/CMakeLists.txt`. Reuse
the PASS/FAIL print + `g_failed` pattern from the existing files for
consistent CI logs.

**Render-diff test.** Add a tuple to the `TUPLES=()` array in **both**
`tests/render/run.sh` and `tests/render/regenerate-golden.sh` (the array
is duplicated to keep each script self-contained). Then run
`tests/render/regenerate-golden.sh` to produce the new golden, eyeball,
and commit.

## File layout

```
tests/
├── CMakeLists.txt               # ctest registration
├── README.md                    # this file
├── unit/
│   ├── colorspace.cpp           # CIE matrices, yBarIntegral, singleLambdaXYZ
│   ├── rgbtospec.cpp            # Jakob fits, LUT, Material accessors
│   ├── spectrum.cpp             # 61-sample container, interp, arithmetic
│   ├── bvh.cpp                  # build + traversal vs brute-force
│   └── optics.cpp               # Schlick, Cauchy IOR, dielectric bounce
└── render/
    ├── run.sh                   # diff against goldens
    ├── regenerate-golden.sh     # rebuild goldens
    ├── diff.py                  # PNG pixel diff (PIL)
    ├── golden/                  # reference images, committed
    └── output/                  # run.sh writes here, gitignored
```
