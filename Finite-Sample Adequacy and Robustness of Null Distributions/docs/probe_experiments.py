#!/usr/bin/env python3
"""Determinism probe for ``experiments.py``.

Run from the repository root::

    python3 docs/probe_experiments.py

The archive script draws every statistic from one global ``numpy.random`` stream
that is seeded once at import (``np.random.seed(2024)``).  This probe replays a
fixed prefix of that stream through all ten archived generators and prints a
SHA-256 over the resulting float64 bytes, so that a change to the file can be
shown to be behavior-neutral (identical digest) or not.

Used by t4-docs to verify that adding docstrings to ``experiments.py`` does not
alter a single drawn number.

Author: t4-docs (task-4).
"""

from __future__ import annotations

import hashlib
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np  # noqa: E402

try:  # the archive module is experiments.py in the submission layout, archive_experiments.py in the repository
    import experiments as EX  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - layout-dependent
    import archive_experiments as EX  # noqa: E402

N = 200  # small replication count: the probe checks the RNG stream, not precision


def main() -> int:
    """Draw one fixed batch from each archived generator and hash the result."""
    np.random.seed(2024)
    arrays = {
        "T1_t_test": EX.mc_t_test(50, N),
        "T2_anova": EX.mc_anova(3, 50, N)[0],
        "T3_kruskal_wallis": EX.mc_kruskal_wallis(3, 50, N),
        "T4_chisq_gof": EX.mc_chisq_gof(6, 250, N),
        "T5_bartlett": EX.mc_bartlett(3, 50, N),
        "T6_fligner_killeen": EX.mc_fligner_killeen(3, 50, N),
        "T7_sign_test": EX.mc_sign_test(50, N),
        "T8_cochran_q": EX.mc_cochran_q(3, 50, N),
        "T9_median_test": EX.mc_median_test(3, 50, N),
        "T10_hotelling": EX.mc_hotelling_t2(2, 50, N)[0],
    }
    h = hashlib.sha256()
    print(f"numpy {np.__version__}   seed 2024   N = {N}")
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=np.float64)
        h.update(arr.tobytes())
        print(f"  {name:<20} shape={str(arr.shape):<12} "
              f"first={arr.flat[0]: .10f}  last={arr.flat[-1]: .10f}")
    print(f"\ncombined SHA-256 of all ten arrays : {h.hexdigest()}")
    _arch = next((os.path.join(ROOT, "code", n)
                  for n in ("experiments.py", "archive_experiments.py")
                  if os.path.exists(os.path.join(ROOT, "code", n))), None)
    print(f"sha256({os.path.basename(_arch)})".ljust(35) + ": "
          f"{hashlib.sha256(open(_arch, 'rb').read()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
