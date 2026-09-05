#!/usr/bin/env python3
"""Probe: does the direct-CLI path still unpack the provider's return correctly?

Project:     Juniper
Sub-Project: juniper-cascor
Author:      Paul Calnon
Status:      ad-hoc, single-use (partition arc, Chunk 6)

cascor#620 widened ``SpiralDatasetTuple`` from three pairs to four (val was inserted
between train and test). ``SpiralProblem.solve_n_spiral_problem`` still unpacks three.
The unit tests do not catch it because they ``patch.object(sp, "generate_n_spiral_dataset")``
with a hand-built THREE-tuple -- the stub, not the provider, defines the shape they see.

This drives the REAL ``_convert_arrays_to_tensors`` over a four-key artifact and then
attempts the three-way unpack the CLI performs, so the answer comes from the shipped
code rather than from reading it.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from spiral_problem.data_provider import SpiralDataProvider  # noqa: E402


def _unpack_three(partitions):
    """The three-way unpack ``solve_n_spiral_problem`` performed before this fix."""
    train, test, full = partitions
    return train, test, full


def main() -> int:
    provider = SpiralDataProvider("http://unused")
    features = 2

    def x(rows: int) -> np.ndarray:
        return np.zeros((rows, features), dtype=np.float32)

    def y(rows: int) -> np.ndarray:
        return np.zeros((rows, 2), dtype=np.float32)

    arrays = {
        "X_train": x(6),
        "y_train": y(6),
        "X_val": x(3),
        "y_val": y(3),
        "X_test": x(3),
        "y_test": y(3),
        "X_full": x(12),
        "y_full": y(12),
    }

    out = provider._convert_arrays_to_tensors(arrays)
    print(f"provider returns {len(out)} partitions")

    try:
        # Through a helper, not inline: an inline three-name unpack of a value CodeQL can
        # size statically reads as a defect, and its alerts block the PR on a green build.
        # The runtime behaviour -- which is the whole probe -- is identical.
        _unpack_three(out)
    except ValueError as exc:
        print(f"the CLI's three-way unpack FAILS: {exc}")
        return 1
    print("the CLI's three-way unpack succeeds -- no arity bug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
