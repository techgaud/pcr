# GPU render-diff goldens

This directory holds reference PNGs produced by `physically-cringe-rendering-cli`
for the tuple matrix defined in `../run.sh`.

Goldens are baked locally on an Apple Silicon Mac (M1+) by running
`../regenerate-golden.sh`. They are committed to the repo. CI on
macos-latest re-renders the same matrix and tolerance-diffs against
these files.

The matrix targets:

- megakernel vs wavefront A/B on the same scene+mode
- fork vs terminate spectral dispersion on cornell-glass
- spectral hero=1 vs hero=4 (hero=1 guards a latent FOV bug in the
  megakernel hero=1 fallback path)
- `--stratified` on wavefront, a regression marker for the
  wavefront-stratified latent (today the kernels ignore the flag and
  the golden looks identical to non-stratified; when the kernels
  start honoring it the golden will need to update)
- cornell-bunny for BVH + mesh coverage
- cornell-spec for measured-SPD + Jakob upsampling coverage

If the golden set is empty, `../run.sh` exits 0 with a skip notice so
CI doesn't fail before the first bake. Override with
`PCR_TEST_SKIP_IF_NO_GOLDENS=0` if you want to force-fail.

## Regen flow

```
cmake -S code -B code/Build
cmake --build code/Build -j --target physically-cringe-rendering-cli
tests/render-gpu/regenerate-golden.sh
# eyeball the PNGs
git add tests/render-gpu/golden
git commit -m "tests: bake GPU render-diff goldens"
```

Bump every golden whenever a tuple's underlying render changes by
design (kernel rewrite, scene version bump, flag semantics change).
