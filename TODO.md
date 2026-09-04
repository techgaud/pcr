# TODO

Deferred work for pcr is tracked in Vikunja, not here. State lives in git and
Vikunja, never in a durable doc.

Open items live in the **Techgaud** Vikunja project, each title prefixed
`PCR:`. As of the 2026-09-04 migration out of this file:

- Validate spectral caustic photon mapping on Metal (Mac A/B) — the last
  outstanding spectral-photon work
- Fix the OIDN buffer-access bug (denoise silently disabled on Apple Silicon)
- Bring BSDF-side MIS to CPU + OpenGL + Metal megakernel (parity)
- Indirect dispatch for the wavefront shading kernels (Metal)
- Adaptive sampling in wavefront multi-sample-per-pass mode
- A/B multi-level prefix-sum vs SIMD-batched atomic queue compaction (Metal)
- macOS signed + notarized .dmg distribution
- MetalRT hardware ray tracing (M3+ only, hardware-gated)

Each task carries its own implementation outline, risk notes, and validation
steps. The full prior prose (with line-number starting points and code
snippets) is recoverable from this file's git history before the migration
commit.

Durable design and architecture stay in the docs: `docs/photon-mapping.md`,
`docs/rendering-backends.md`, and `project-knowledge.md`.
