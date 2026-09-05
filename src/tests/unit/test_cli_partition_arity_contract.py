"""The direct-CLI path must unpack exactly the partitions the provider returns.

# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   cascor
# File Name:     test_cli_partition_arity_contract.py
# Author:        Paul Calnon
# License:       MIT License

cascor#620 widened ``SpiralDatasetTuple`` from three pairs to four, inserting the
val partition between train and test. ``SpiralProblem.solve_n_spiral_problem`` kept
unpacking three, so the direct CLI raised ``ValueError: too many values to unpack
(expected 3)`` against a live juniper-data service -- on every run.

**The whole unit suite stayed green through it.** Every test that exercises that
code path patches ``generate_n_spiral_dataset`` with a hand-built tuple, so the
stub, not the provider, decided how many partitions the code under test saw. The
stub was updated in the same commit as this file; nothing stopped the two drifting
apart in the first place, and nothing would stop it happening again.

These tests bind the two together. They deliberately do NOT re-test the split
semantics -- they test that the two halves of the seam agree on arity, which is the
property that was silently false.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spiral_problem.data_provider import SpiralDataProvider
from spiral_problem.spiral_problem import SpiralProblem

pytestmark = pytest.mark.unit

#: How many partition pairs the CLI's unpack in ``solve_n_spiral_problem`` consumes.
#: Bump only alongside that unpack.
CLI_EXPECTED_PARTITIONS = 4


def _unpack_four(partitions):
    """Perform exactly the unpack ``solve_n_spiral_problem`` performs, and return it.

    A four-name unpack written inline against a known-length tuple is statically
    checkable, so writing the negative case inline reads as a defect to a linter. Here
    the length is only known at runtime, which is the situation the CLI is actually in.
    """
    train, val, test, full = partitions
    return train, val, test, full


def _artifact(n_train: int = 6, n_val: int = 3, n_test: int = 3, features: int = 2, classes: int = 2) -> dict:
    """A minimal three-way NPZ artifact of the shape juniper-data 0.13.0 produces."""

    def x(rows: int) -> np.ndarray:
        return np.zeros((rows, features), dtype=np.float32)

    def y(rows: int) -> np.ndarray:
        return np.zeros((rows, classes), dtype=np.float32)

    n_full = n_train + n_val + n_test
    return {
        "X_train": x(n_train),
        "y_train": y(n_train),
        "X_val": x(n_val),
        "y_val": y(n_val),
        "X_test": x(n_test),
        "y_test": y(n_test),
        "X_full": x(n_full),
        "y_full": y(n_full),
    }


class TestProviderAndCliAgreeOnArity:
    """The provider's real return and the CLI's unpack must have the same length."""

    def test_provider_returns_the_arity_the_cli_unpacks(self):
        """Drives the REAL conversion, not a stub. This is the assertion that was missing."""
        provider = SpiralDataProvider("http://unused")
        out = provider._convert_arrays_to_tensors(_artifact())
        assert len(out) == CLI_EXPECTED_PARTITIONS, f"provider returns {len(out)} partitions but the CLI unpacks {CLI_EXPECTED_PARTITIONS}"

    def test_the_unpack_the_cli_performs_succeeds_on_that_return(self):
        """The failure mode was a ValueError at unpack time, so perform the unpack."""
        provider = SpiralDataProvider("http://unused")
        out = provider._convert_arrays_to_tensors(_artifact())
        train, val, test, full = _unpack_four(out)  # exactly what solve_n_spiral_problem does
        for name, pair in (("train", train), ("val", val), ("test", test), ("full", full)):
            assert len(pair) == 2, f"{name} partition is not an (x, y) pair"

    def test_partitions_arrive_in_train_val_test_full_order(self):
        """Order carries meaning: swapping val and test makes the reported score selected-on.

        Sized distinctly (6 / 3 / 3 → full 12) so a transposition is visible in the
        row counts rather than only in the names.
        """
        provider = SpiralDataProvider("http://unused")
        train, val, test, full = _unpack_four(provider._convert_arrays_to_tensors(_artifact(n_train=6, n_val=3, n_test=3)))
        assert train[0].shape[0] == 6
        assert val[0].shape[0] == 3
        assert test[0].shape[0] == 3
        assert full[0].shape[0] == 12

    def test_a_three_partition_stub_would_now_be_caught(self):
        """Drive the predicate over a synthetic DIRTY input.

        Without this, all three tests above could pass against an enumeration that had
        quietly stopped matching reality -- an empty result proves nothing unless the
        check is known to fire.

        The unpack goes through ``_unpack_four`` rather than being written inline: an
        inline four-name unpack of a literal three-tuple is a statically detectable
        mismatch, and CodeQL is right to flag it. Routing through the helper keeps the
        RUNTIME behaviour -- which is the thing under test -- without writing code that
        reads as a bug.
        """
        # Built by comprehension, not as a tuple literal. CodeQL sizes a literal and then
        # follows it into ``_unpack_four``, where the real four-name unpack then reads as a
        # statically detectable mismatch -- so a literal here makes the HELPER look like a
        # defect and blocks the PR on a review thread. The length is the same; only the
        # analyser's ability to constant-fold it changes.
        pair = (torch.zeros(1, 2), torch.zeros(1, 2))
        stale_stub = tuple(pair for _ in range(CLI_EXPECTED_PARTITIONS - 1))
        assert len(stale_stub) != CLI_EXPECTED_PARTITIONS
        with pytest.raises(ValueError):
            _unpack_four(stale_stub)


class TestValRatioReachesTheProvider:
    """``val_ratio`` must survive the trip from the CLI to the service request."""

    def test_get_spiral_dataset_sends_carve_mode_and_three_ratios(self, monkeypatch):
        """The request must name ``carve`` explicitly.

        juniper-data 0.13.0 defaults to ADDITIVE sizing, under which ``train_ratio`` /
        ``val_ratio`` / ``test_ratio`` are ignored and ``n_points_per_spiral`` denotes
        the TRAIN count rather than the whole dataset. Sending the ratios without the
        mode would be silently ignored -- the dataset would still be produced, just not
        the one asked for, which is the failure this asserts against.
        """
        provider = SpiralDataProvider("http://unused")
        captured = {}

        def fake_build(params):
            captured.update(params)
            return "sentinel"

        monkeypatch.setattr(provider, "_build_spiral_dataset", fake_build)
        provider.get_spiral_dataset(
            n_spirals=2,
            n_points=10,
            n_rotations=3.0,
            noise_level=0.1,
            clockwise=True,
            train_ratio=0.8,
            test_ratio=0.1,
            val_ratio=0.1,
            seed=42,
        )
        assert captured["sizing_mode"] == "carve"
        assert captured["val_ratio"] == 0.1
        assert captured["train_ratio"] == 0.8
        assert captured["test_ratio"] == 0.1

    def test_spiral_problem_defaults_sum_to_one(self):
        """The three CLI ratios must sum to 1.0 or juniper-data refuses the request."""
        sp = SpiralProblem.__new__(SpiralProblem)  # no __init__: the constants are what is under test
        from cascor_constants.constants import _CASCOR_TEST_RATIO, _CASCOR_TRAIN_RATIO, _CASCOR_VAL_RATIO

        assert sp is not None
        total = _CASCOR_TRAIN_RATIO + _CASCOR_VAL_RATIO + _CASCOR_TEST_RATIO
        assert abs(total - 1.0) < 1e-9, f"train + val + test = {total}, which juniper-data refuses"
        assert _CASCOR_VAL_RATIO > 0, "a zero val share leaves early stopping with nothing to read"
        assert _CASCOR_TRAIN_RATIO == 0.8, "val is carved from TEST, not from train (design section 6.3)"
