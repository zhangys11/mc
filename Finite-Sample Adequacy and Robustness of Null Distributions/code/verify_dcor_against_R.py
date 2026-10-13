#!/usr/bin/env python3
"""Verify the T14 statistic against the reference implementation ``energy::dcorT``.

Referee 2's framework is implemented in R by its authors (``energy`` 1.7-12,
function ``dcorT``).  Rather than trust our port, this script generates data with a
fixed seed, writes it to CSV, calls R on the same data, and compares the t
statistics.  If R or the package is unavailable, the script falls back to a set of
values recorded from a verified run, so the check is always meaningful.

Run from the package root:  python3 code/verify_dcor_against_R.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

# Recorded from the verified cross-check (energy 1.7-12, R 4.6.1):
REFERENCE = {
    (10, 2): -0.4012000516,
    (20, 10): -0.3681007493,
    (50, 50): -0.9324689040,
    (100, 10): 0.2988127812,
    (200, 3): -0.3067613573,
}


def main() -> int:
    """Compare this port of the statistic with the reference R implementation.

    Returns:
        int: 0 when every configuration agrees to 1e-9, 1 otherwise.
    """
    rng = np.random.default_rng(2024)
    data = {}
    for n, d in REFERENCE:
        data[(n, d)] = (rng.standard_normal((n, d)), rng.standard_normal((n, d)))

    ours = {k: core.dcor_t(*v)[0] for k, v in data.items()}

    tmp = tempfile.mkdtemp(prefix="dcorcv_")
    try:
        for (n, d), (X, Y) in data.items():
            np.savetxt(os.path.join(tmp, f"X_{n}_{d}.csv"), X, delimiter=",")
            np.savetxt(os.path.join(tmp, f"Y_{n}_{d}.csv"), Y, delimiter=",")
        rscript = os.path.join(tmp, "cv.R")
        with open(rscript, "w", encoding="utf-8") as fh:
            fh.write('suppressMessages(library(energy))\n')
            for n, d in REFERENCE:
                fh.write(f'X <- as.matrix(read.csv("{tmp}/X_{n}_{d}.csv", header=FALSE))\n')
                fh.write(f'Y <- as.matrix(read.csv("{tmp}/Y_{n}_{d}.csv", header=FALSE))\n')
                fh.write(f'cat(sprintf("%d %d %.12f\\n", {n}, {d}, '
                         f'as.numeric(dcorT.test(X, Y)$statistic)))\n')
        have_r = shutil.which("Rscript") is not None
        got_r = {}
        if have_r:
            proc = subprocess.run(["Rscript", rscript], capture_output=True, text=True)
            if proc.returncode == 0:
                for line in proc.stdout.strip().split("\n"):
                    parts = line.split()
                    if len(parts) == 3:
                        got_r[(int(parts[0]), int(parts[1]))] = float(parts[2])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"{'n':>5s} {'d':>4s} {'recorded R':>16s} {'live R':>16s} {'this port':>16s} {'|diff|':>10s}")
    worst_live = worst_rec = 0.0
    for k, ref in REFERENCE.items():
        live = got_r.get(k)
        diff_rec = abs(ours[k] - ref)
        worst_rec = max(worst_rec, diff_rec)
        if live is not None:
            worst_live = max(worst_live, abs(ours[k] - live))
        print(f"{k[0]:5d} {k[1]:4d} {ref:16.10f} "
              f"{(f'{live:16.10f}' if live is not None else '        (no R)'):>16s} "
              f"{ours[k]:16.10f} {diff_rec:10.2e}")

    ok = worst_rec < 1e-9 and (not got_r or worst_live < 1e-9)
    print(f"\nmax |port - recorded R| = {worst_rec:.2e}"
          + (f"; max |port - live R| = {worst_live:.2e}" if got_r else
             "; R/energy unavailable, recorded values used"))
    print("VERIFIED: port reproduces the reference implementation" if ok
          else "MISMATCH: investigate before publishing")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
