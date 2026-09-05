#!/usr/bin/env python3
"""Chunk 6: plumb the validation split through the direct CLI.

Project:     Juniper
Sub-Project: juniper-cascor
Author:      Paul Calnon
Status:      ad-hoc, single-use (partition arc, Chunk 6)

Two things, one of which is a live regression.

**The regression.** cascor#620 widened ``SpiralDatasetTuple`` from three pairs to
four. ``SpiralProblem.solve_n_spiral_problem`` still unpacks three, so the direct
CLI raises ``ValueError: too many values to unpack (expected 3)`` against a real
service. The unit tests miss it because they ``patch.object(sp,
"generate_n_spiral_dataset")`` with a hand-built three-tuple: the STUB, not the
provider, defines the shape they see. Proved by
``2026-09-05_probe_cli_unpack_arity.py``, which drives the real conversion.

**Chunk 6 proper (E-11).** ``fit()`` has accepted ``x_val`` / ``y_val`` all along --
nothing in the CLI ever passed them, so the CLI's early stopping had no validation
signal and only the service tier got one. That is design decision 5's tier
asymmetry. The CLI now carves a val split and hands it to ``fit()``.

Sizing: train stays at 0.8 and val is taken from test's share (0.2 -> 0.1 test +
0.1 val), matching the ecosystem default. Section 6.3's principle is that train
does not move; taking val out of train would move it and invalidate every CLI
baseline for a reason the design explicitly rejects.
"""

from __future__ import annotations

import pathlib
import sys

WT = pathlib.Path("/home/pcalnon/Development/python/Juniper/worktrees/juniper-cascor--feature--cli-val-split--20260905-1015--4ee5d94a")

EDITS: list[tuple[str, str, str, int]] = [
    # ---------------------------------------------------------------- constants
    (
        "src/cascor_constants/constants_problem/constants_problem.py",
        """# Define constants for the Two Spiral Problem Dataset, train and test ratios used for splitting the dataset
_SPIRAL_PROBLEM_TRAIN_RATIO = 0.8
_SPIRAL_PROBLEM_TEST_RATIO = 0.2
""",
        """# Define constants for the Two Spiral Problem Dataset, train / val / test ratios used for splitting the dataset.
#
# ``val`` is the IN-LOOP split -- early stopping reads it and no reported metric may.
# Its share comes out of TEST (0.2 -> 0.1), never out of train: design section 6.3 is
# explicit that the training count does not move, and moving it would invalidate every
# CLI baseline for a reason the design rejects. The three sum to 1.0; the generator
# refuses them otherwise.
_SPIRAL_PROBLEM_TRAIN_RATIO = 0.8
_SPIRAL_PROBLEM_VAL_RATIO = 0.1
_SPIRAL_PROBLEM_TEST_RATIO = 0.1
""",
        1,
    ),
    (
        "src/cascor_constants/constants.py",
        """_CASCOR_TEST_RATIO = _SPIRAL_PROBLEM_TEST_RATIO
_CASCOR_TRAIN_RATIO = _SPIRAL_PROBLEM_TRAIN_RATIO
""",
        """_CASCOR_TEST_RATIO = _SPIRAL_PROBLEM_TEST_RATIO
_CASCOR_TRAIN_RATIO = _SPIRAL_PROBLEM_TRAIN_RATIO
_CASCOR_VAL_RATIO = _SPIRAL_PROBLEM_VAL_RATIO
""",
        1,
    ),
    (
        "src/cascor_constants/constants.py",
        """_SPIRAL_PROBLEM_TEST_RATIO = _CASCOR_TEST_RATIO
_SPIRAL_PROBLEM_TRAIN_RATIO = _CASCOR_TRAIN_RATIO
""",
        """_SPIRAL_PROBLEM_TEST_RATIO = _CASCOR_TEST_RATIO
_SPIRAL_PROBLEM_TRAIN_RATIO = _CASCOR_TRAIN_RATIO
_SPIRAL_PROBLEM_VAL_RATIO = _CASCOR_VAL_RATIO
""",
        1,
    ),
]


def replace(rel: str, old: str, new: str, expected: int) -> bool:
    path = WT / rel
    src = path.read_text()
    found = src.count(old)
    if found != expected:
        print(f"{rel}: matched {found}x, expected {expected} -- refusing", file=sys.stderr)
        return False
    path.write_text(src.replace(old, new))
    print(f"{rel}: ok")
    return True


def main() -> int:
    ok = True
    for rel, old, new, expected in EDITS:
        ok &= replace(rel, old, new, expected)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
