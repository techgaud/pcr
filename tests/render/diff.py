#!/usr/bin/env python3
"""PNG diff for render regression tests.

Compares two PNG files at the pixel level. Reports the fraction of
pixels whose maximum per-channel absolute difference exceeds a
threshold; passes if that fraction is below a budget.

With --seed-deterministic renders, expect bit-exact output on the
same machine. The default threshold (1/255 in 99% of pixels) is tight
enough to catch real regressions but allows for tiny FP-rounding
noise in case anyone re-runs without --seed by mistake. Tweak via
flags.

Usage:
    diff.py reference.png candidate.png [--abs-tol 1] [--pct-budget 0.01]

Exit codes:
    0  - within budget
    1  - exceeded budget (regression)
    2  - usage error or missing file
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("error: PIL/Pillow required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(2)


def diff_images(ref_path: Path, cand_path: Path, abs_tol: int, pct_budget: float) -> int:
    if not ref_path.exists():
        print(f"error: reference image missing: {ref_path}", file=sys.stderr)
        return 2
    if not cand_path.exists():
        print(f"error: candidate image missing: {cand_path}", file=sys.stderr)
        return 2

    ref = Image.open(ref_path).convert("RGB")
    cand = Image.open(cand_path).convert("RGB")
    if ref.size != cand.size:
        print(f"FAIL: size mismatch ref={ref.size} cand={cand.size}", file=sys.stderr)
        return 1

    rb = ref.tobytes()
    cb = cand.tobytes()
    n_pixels = ref.size[0] * ref.size[1]
    bad = 0
    max_diff = 0
    for i in range(n_pixels):
        rR, rG, rB = rb[3*i], rb[3*i+1], rb[3*i+2]
        cR, cG, cB = cb[3*i], cb[3*i+1], cb[3*i+2]
        d = max(abs(rR - cR), abs(rG - cG), abs(rB - cB))
        if d > max_diff:
            max_diff = d
        if d > abs_tol:
            bad += 1

    bad_frac = bad / n_pixels
    label = ref_path.name
    print(f"{label}: bad={bad}/{n_pixels} ({bad_frac:.4%}) max_diff={max_diff}/255 "
          f"budget={pct_budget:.2%} tol={abs_tol}/255")
    return 0 if bad_frac <= pct_budget else 1


def main() -> int:
    p = argparse.ArgumentParser(description="PNG diff for render tests")
    p.add_argument("reference", type=Path, help="Golden image path")
    p.add_argument("candidate", type=Path, help="Output image to check")
    p.add_argument("--abs-tol", type=int, default=1,
                   help="Per-channel absolute difference threshold (0-255). Default 1.")
    p.add_argument("--pct-budget", type=float, default=0.01,
                   help="Fraction of pixels allowed to exceed --abs-tol. Default 0.01 (1%%).")
    args = p.parse_args()
    return diff_images(args.reference, args.candidate, args.abs_tol, args.pct_budget)


if __name__ == "__main__":
    sys.exit(main())
