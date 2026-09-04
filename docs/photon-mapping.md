# Caustic photon mapping

pcr renders caustics with photon mapping layered on top of the eye-side path
tracer. It is caustic-only by design: photon mapping handles light paths that
touch at least one specular surface (mirror or glass) before landing on a
diffuse surface. Direct lighting and diffuse-only indirect are still produced
by the ordinary path tracer, unchanged. The two contributions compose at the
diffuse hit.

## Module

The algorithm lives in `code/Photon/`:

- `PhotonMap.{h,cpp}`: the photon store, a Jensen-style spatial hash grid.
- `PhotonShoot.{h,cpp}`: host-side photon tracing. Walks the scene BVH and
  reuses the renderer's `dielectricBounce` so photon refraction matches eye-ray
  refraction exactly.
- `DensityEstimate.h`: header-only CPU density estimator at a visible point.
- `GpuFlatten.h`: flattens the host photon store into the flat table the GPU
  backends consume, shared by OpenGL and Metal.
- `Sppm.h`: the per-pixel SPPM state struct and the Hachisuka radius-update
  equations, with `alpha = 2/3`.

## Three modes

All three estimate caustic radiance by gathering photons near a diffuse
visible point, and differ in how they trade bias for noise across passes.

- **Classical** (Jensen 1996): one photon shoot, then a fixed-radius density
  estimate at each diffuse hit.
- **Progressive** (Hachisuka 2008, plain): ensemble averaging across N fresh
  photon shoots. Each pass shoots a new photon set and averages the estimates.
- **SPPM** (Hachisuka and Jensen 2009): per-pixel adaptive radius shrinkage.
  Each pass shrinks the per-pixel gather radius toward the true density,
  driving bias down as passes accumulate.

Note that this codebase's SPPM is progressive-based: it re-renders the eye
image each pass rather than caching visible points across passes. SPPM is
therefore not faster than plain progressive here, and it does not clean
monotonically with more passes the way plain Monte Carlo does. Shrinking the
radius over many passes concentrates sparse-pixel flux, which can raise firefly
variance rather than lower it.

## Controls, metadata, and output

- CLI: `--photon-map`, `--photons N` (default 1M), `--photon-radius R`
  (default 0.05), `--photon-progressive`, `--photon-passes N` (default 8),
  `--photon-sppm`.
- GUI: nested Techniques rows under a caustic photon mapping group (photons,
  radius, progressive, passes, SPPM), all JSON-persisted.
- PNG `tEXt` chunks (`PhotonMap`, `PhotonCount`, `PhotonRadius`,
  `PhotonProgressive`, `PhotonPasses`, `PhotonSppm`) are gated on what actually
  ran, so metadata reflects the real render, not the requested flags.
- Filename suffix precedence is `-photon-sppm` over `-photon-prog` over
  `-photon`, so A/B renders sort cleanly.

## Backend routing

Classical, progressive, and SPPM all run on the four backends: CPU, Metal
megakernel, Metal wavefront, and OpenGL. A few backend-specific mechanics:

- **Wavefront SPPM.** The split-kernel pipeline carries visible-point
  semantics with a `firstDiffuse` bit in the `rayState` packing, set by
  ray-gen, passed through the mirror and glass kernels, and consumed by the
  diffuse kernel at the first diffuse hit. The `sppm` and `sppmDelta` buffers
  bind at wavefront slots 21 and 22. Wavefront SPPM forces one sample per pass,
  because a multi-sample pass would race the atomic-free per-pixel delta
  deposit.
- **OpenGL SPPM.** A GLSL twin of the megakernel update: a second compute
  program runs the per-pass Hachisuka update, with `SppmPixel` and `SppmDelta`
  SSBOs, depositing at the first diffuse hit. The host-side composite is
  identical to the CPU and Metal paths.

## Spectral (dispersion) caustics

Spectral photon mapping produces chromatic dispersion in caustics: a glass
surface with `cauchyB > 0` splits white light into a colored caustic. The
formulation carries per-wavelength photon power (call it formulation B):

- Each photon carries a hero-4 `specPower`. At a dispersive glass surface
  (`cauchyB > 0`) the photon stochastically collapses to a single wavelength
  and refracts at that wavelength's Cauchy IOR. That single-wavelength
  refraction is the dispersion.
- The photon shoot and the eye paths share **per-pass hero wavelengths**,
  rotated across passes to cover 400 to 700 nm, so that `specPower[k]` aligns
  with the eye path's `lambda[k]` by index.
- In SPPM, the per-wavelength caustic flux is converted to RGB (via CIE) at the
  visible point and folded into the existing RGB `SppmPixel`, so the SPPM state
  layout does not change.

### GPU representation

The GPU keeps a **parallel** spectral-power buffer rather than growing the base
photon record. `GpuFlatten` emits a `GpuSpectralPower` (float4) array parallel
to the 36-byte RGB `GpuRecord`, plus the per-pass lambdas, so the RGB GPU path
is untouched by the spectral addition. OpenGL binds the spectral powers as the
`SpectralPowers` SSBO at binding 12. Metal binds them as a spectral-power
buffer at `buffer(13)` on the megakernel kernels and at `buffer(23)` in the
wavefront diffuse kernel.

### Spectral SPPM on the Metal wavefront routes to the megakernel

Spectral SPPM specifically falls back to the megakernel on Metal. This is a
structural constraint, not a shortcut. Per-pixel SPPM needs one visible point
per pixel carrying all four hero wavelengths together, which is the megakernel
hero-4 single-path shape. The wavefront's monochromatic spectral fork splits a
glass-piercing camera ray into sub-rays that share the parent's `pixelIdx` and
would race the atomic-free `sppmDelta` deposit, and in terminate mode drop
three quarters of the spectrum. Classical and progressive have no such race:
their forks write their own ray-slot color, and the atomic fork-scatter
combines them afterward, so both run on the wavefront. The gate is narrow:
spectral and SPPM and wavefront together route to the megakernel, and every
other combination stays on the wavefront.

## Performance characteristics

- Spectral photon mapping (progressive and SPPM) re-renders the fork-heavy
  spectral eye image once per pass, so on the CPU it runs on the order of
  minutes per pass. The CPU spectral photon path is hard-capped at 16 passes
  with a warning (`kSpectralPhotonPassCap` in `Renderer.cpp`). The CPU is the
  reference path; the GPU is the practical one.
- SPPM is firefly-prone at low sample counts with many passes. A config like
  `-s 1` with several hundred passes is an A/B comparison config, not a way to
  render a final image: `-s 1` leaves eye-path indirect noisy, and many passes
  shrink the SPPM radius so `tau / (pi R^2 N)` spikes at sparse pixels. For a
  clean image, use fewer passes (around 128), `--mis --russian`, and optionally
  `--oidn`.
