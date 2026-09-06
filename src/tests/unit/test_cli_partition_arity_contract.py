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

Decision 11 has since narrowed it back to three by removing the ``full`` pair, which
is the same seam moving in the opposite direction -- and the same failure mode was
available: an unpack left at four against a three-tuple raises ``ValueError: not
enough values to unpack (expected 4, got 3)``. The arity constant below is the single
place that records which it is.

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
CLI_EXPECTED_PARTITIONS = 3


def _unpack_three(partitions):
    """Perform exactly the unpack ``solve_n_spiral_problem`` performs, and return it.

    A three-name unpack written inline against a known-length tuple is statically
    checkable, so writing the negative case inline reads as a defect to a linter. Here
    the length is only known at runtime, which is the situation the CLI is actually in.
    """
    train, val, test = partitions
    return train, val, test


def _artifact(n_train: int = 6, n_val: int = 3, n_test: int = 3, features: int = 2, classes: int = 2, with_full: bool = False) -> dict:
    """A minimal three-way NPZ artifact.

    ``with_full=False`` is the shape juniper-data produces after decision 11
    (juniper-data#369). ``with_full=True`` is the legacy shape every artifact minted
    before it carries, and which design §9.5.4 obliges this consumer to keep loading --
    see ``TestTheRetiredFullFamilyIsToleratedNotRequired``.
    """

    def x(rows: int) -> np.ndarray:
        return np.zeros((rows, features), dtype=np.float32)

    def y(rows: int) -> np.ndarray:
        return np.zeros((rows, classes), dtype=np.float32)

    artifact = {
        "X_train": x(n_train),
        "y_train": y(n_train),
        "X_val": x(n_val),
        "y_val": y(n_val),
        "X_test": x(n_test),
        "y_test": y(n_test),
    }
    if with_full:
        n_full = n_train + n_val + n_test
        artifact["X_full"] = x(n_full)
        artifact["y_full"] = y(n_full)
    return artifact


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
        train, val, test = _unpack_three(out)  # exactly what solve_n_spiral_problem does
        for name, pair in (("train", train), ("val", val), ("test", test)):
            assert len(pair) == 2, f"{name} partition is not an (x, y) pair"

    def test_partitions_arrive_in_train_val_test_order(self):
        """Order carries meaning: swapping val and test makes the reported score selected-on.

        Sized distinctly (7 / 3 / 5) so a transposition is visible in the row counts
        rather than only in the names. The three differ from each other AND from every
        pairwise sum, so no permutation of them reproduces this triple.
        """
        provider = SpiralDataProvider("http://unused")
        train, val, test = _unpack_three(provider._convert_arrays_to_tensors(_artifact(n_train=7, n_val=3, n_test=5)))
        assert train[0].shape[0] == 7
        assert val[0].shape[0] == 3
        assert test[0].shape[0] == 5

    def test_a_stale_four_partition_stub_would_now_be_caught(self):
        """Drive the predicate over a synthetic DIRTY input.

        Without this, all three tests above could pass against an enumeration that had
        quietly stopped matching reality -- an empty result proves nothing unless the
        check is known to fire.

        The unpack goes through ``_unpack_three`` rather than being written inline: an
        inline three-name unpack of a literal four-tuple is a statically detectable
        mismatch, and CodeQL is right to flag it. Routing through the helper keeps the
        RUNTIME behaviour -- which is the thing under test -- without writing code that
        reads as a bug.

        The stale stub is now one pair LONGER than expected rather than one shorter,
        because decision 11 narrowed the tuple: the drift this guards against is a
        caller still unpacking the retired ``full`` pair.
        """
        # Built by comprehension, not as a tuple literal. CodeQL sizes a literal and then
        # follows it into ``_unpack_three``, where the real three-name unpack then reads as
        # a statically detectable mismatch -- so a literal here makes the HELPER look like a
        # defect and blocks the PR on a review thread. The length is the same; only the
        # analyser's ability to constant-fold it changes.
        pair = (torch.zeros(1, 2), torch.zeros(1, 2))
        stale_stub = tuple(pair for _ in range(CLI_EXPECTED_PARTITIONS + 1))
        assert len(stale_stub) != CLI_EXPECTED_PARTITIONS
        with pytest.raises(ValueError):
            _unpack_three(stale_stub)


class TestTheRetiredFullFamilyIsToleratedNotRequired:
    """Decision 11 makes ``*_full`` neither required nor forbidden. Both halves matter.

    Requiring it rejects every artifact juniper-data produces after #369. FORBIDDING it
    rejects every artifact minted before #369 -- and design §9.5.4 obliges consumers to
    keep tolerating those. A test that asserts absence turns "not required" into a
    requirement pointing the other way, which is why there is no such assertion here.
    """

    def test_an_artifact_without_the_full_family_loads(self):
        """The post-#369 shape. This is the case the old ``required_keys`` rejected."""
        provider = SpiralDataProvider("http://unused")
        out = provider._convert_arrays_to_tensors(_artifact(with_full=False))
        assert len(out) == CLI_EXPECTED_PARTITIONS

    def test_an_artifact_still_carrying_the_full_family_loads(self):
        """The legacy shape. Tolerated: extra keys are ignored, not rejected."""
        provider = SpiralDataProvider("http://unused")
        out = provider._convert_arrays_to_tensors(_artifact(with_full=True))
        assert len(out) == CLI_EXPECTED_PARTITIONS

    def test_both_shapes_yield_identical_partitions(self):
        """Vintage must not change what the consumer sees.

        The failure this guards against is the ``_full``-bearing artifact taking a
        different code path from the ``_full``-less one and producing different tensors
        -- the class that makes the same logical dataset render two ways depending on
        when it was minted.
        """
        provider = SpiralDataProvider("http://unused")
        legacy = provider._convert_arrays_to_tensors(_artifact(with_full=True))
        current = provider._convert_arrays_to_tensors(_artifact(with_full=False))
        for (lx, ly), (cx, cy) in zip(legacy, current):
            assert torch.equal(lx, cx)
            assert torch.equal(ly, cy)

    def test_a_malformed_full_family_cannot_fail_the_load(self):
        """Tolerating means NOT validating. A junk ``X_full`` must not reach any check.

        Under the old contract ``X_full`` was shape-validated in the same loops as the
        real partitions, so a 3-column or 1-D ``X_full`` raised. Now that nothing reads
        it, a legacy artifact carrying a malformed one must still load -- otherwise the
        retired key retains veto power over an artifact whose live partitions are fine.
        """
        provider = SpiralDataProvider("http://unused")
        artifact = _artifact(with_full=True)
        artifact["X_full"] = np.zeros((99, 7), dtype=np.float32)  # wrong rows AND wrong columns
        artifact["y_full"] = np.zeros(3, dtype=np.float32)  # wrong ndim
        out = provider._convert_arrays_to_tensors(artifact)
        assert len(out) == CLI_EXPECTED_PARTITIONS


class TestTheDerivedWholeDatasetMatchesTheRetiredKey:
    """``solve_n_spiral_problem`` derives ``x_full`` by concatenation. Prove it is the same array.

    juniper-data built the key as ``np.vstack([X_train, X_val, X_test])``
    (``juniper_data/core/split.py``) over contiguous slices. If that identity does not
    hold, the plot the CLI draws from the derived array is not the plot it used to draw.
    """

    def test_concatenating_the_three_partitions_reproduces_the_legacy_full_array(self):
        provider = SpiralDataProvider("http://unused")
        artifact = _artifact(n_train=7, n_val=3, n_test=5, with_full=True)
        # Make the rows distinguishable: a constant per partition, laid down in the same
        # train | val | test order juniper-data used, so a mis-ordered concatenation shows.
        artifact["X_train"][:] = 1.0
        artifact["X_val"][:] = 2.0
        artifact["X_test"][:] = 3.0
        artifact["X_full"] = np.vstack([artifact["X_train"], artifact["X_val"], artifact["X_test"]])

        (x_train, _), (x_val, _), (x_test, _) = provider._convert_arrays_to_tensors(artifact)
        derived = torch.cat([x_train, x_val, x_test], dim=0)

        assert derived.shape[0] == 15
        assert torch.equal(derived, torch.tensor(artifact["X_full"], dtype=torch.float32))

    def test_a_legacy_two_way_artifact_derives_without_the_val_pair(self):
        """No val split: the concatenation is train | test, matching the old two-way ``X_full``."""
        provider = SpiralDataProvider("http://unused")
        artifact = _artifact(n_train=7, n_test=5, with_full=False)
        del artifact["X_val"], artifact["y_val"]

        (x_train, _), (x_val, y_val), (x_test, _) = provider._convert_arrays_to_tensors(artifact)
        assert x_val is None and y_val is None, "a two-way artifact must yield an EMPTY val pair, not a fabricated one"

        # Exactly the guard solve_n_spiral_problem applies before concatenating.
        parts = [x_train] + ([x_val] if x_val is not None else []) + [x_test]
        assert torch.cat(parts, dim=0).shape[0] == 12


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
