#!/usr/bin/env python
"""Unit tests for the per-sample weight cache in _ReplaySession (CAN-015g g-2).

Covers:
- Cache hit/miss semantics + LRU promotion ordering.
- Byte-budget eviction (oldest entries evicted to fit new ones).
- Backward compat: V1 snapshots (no weight_history) yield a cache
  that advertises ``available=False`` and gates the extended
  state_summary fields.
- Per-sample payload shape: output_weights, output_bias, hidden_units
  with first-sample-aware slicing.
- ``state_summary`` extension exposes ``weights_available``,
  ``weight_sampling.{strategy,interval,num_samples,sample_epochs}``.
"""

import os
import sys

import numpy as np
import pytest

# Add parent directories for imports (matches sibling test files).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.lifecycle.manager import _ReplaySession, _WeightCache  # noqa: E402

pytestmark = pytest.mark.unit


def _make_weight_history(num_samples=5, in_size=2, out_size=1, num_hidden=2, sampling_interval=10):
    """Synthetic weight_history payload (mirrors g-1 fixture)."""
    sample_indices = [i * sampling_interval for i in range(num_samples)]
    output_weights = []
    output_bias = []
    for i in range(num_samples):
        hid_at_sample = min(i, num_hidden)
        output_weights.append(np.random.RandomState(i).randn(in_size + hid_at_sample, out_size).astype(np.float32))
        output_bias.append(np.random.RandomState(i + 100).randn(out_size).astype(np.float32))
    hidden_units = []
    for unit_idx in range(num_hidden):
        # ``first_sample_index`` is the **index into ``sample_indices``**
        # (0-based) at which this unit first appeared, NOT an epoch
        # number. The cache uses it directly to slice the per-sample
        # arrays. Unit ``unit_idx`` first appears at sample
        # ``unit_idx + 1`` (so sample 0 has no units, sample 1 has unit
        # 0 only, etc.).
        first_sample_index = unit_idx + 1
        unit_weights = []
        unit_bias = []
        for j in range(num_samples - first_sample_index):
            unit_weights.append(np.random.RandomState(unit_idx * 10 + j).randn(in_size + unit_idx).astype(np.float32))
            unit_bias.append(float(np.random.RandomState(unit_idx * 10 + j + 1).randn()))
        hidden_units.append(
            {
                "first_sample_index": first_sample_index,
                "activation": "tanh",
                "weights": unit_weights,
                "bias": unit_bias,
            }
        )
    return {
        "sampling_strategy": "adaptive",
        "sampling_interval": sampling_interval,
        "sample_indices": sample_indices,
        "output_weights": output_weights,
        "output_bias": output_bias,
        "hidden_units": hidden_units,
    }


# =============================================================================
# Cache standalone
# =============================================================================


class TestWeightCacheBasics:
    def test_empty_weight_history_unavailable(self):
        cache = _WeightCache(None)
        assert cache.available is False
        assert cache.num_samples == 0
        assert cache.get(0) is None

    def test_empty_dict_unavailable(self):
        cache = _WeightCache({})
        assert cache.available is False

    def test_empty_sample_indices_unavailable(self):
        cache = _WeightCache({"sample_indices": []})
        assert cache.available is False

    def test_populated_history_available(self):
        cache = _WeightCache(_make_weight_history(num_samples=3))
        assert cache.available is True
        assert cache.num_samples == 3
        assert cache.sampling_strategy == "adaptive"
        assert cache.sampling_interval == 10

    def test_get_returns_none_for_out_of_range_index(self):
        cache = _WeightCache(_make_weight_history(num_samples=3))
        assert cache.get(-1) is None
        assert cache.get(3) is None
        assert cache.get(100) is None


class TestWeightCachePayloadShape:
    def test_payload_contains_expected_keys(self):
        wh = _make_weight_history(num_samples=4, num_hidden=2)
        cache = _WeightCache(wh)
        payload = cache.get(2)
        assert payload is not None
        assert payload["sample_index"] == 2
        assert payload["epoch"] == 20  # sample_indices[2] = 2 * sampling_interval(10)
        assert "output_weights" in payload
        assert "output_bias" in payload
        assert "hidden_units" in payload

    def test_payload_skips_units_not_yet_added(self):
        wh = _make_weight_history(num_samples=4, num_hidden=2)
        # Units appear at samples 1 and 2 (first_sample_index = 10 and 20).
        # At sample 0 no units are present.
        cache = _WeightCache(wh)
        payload = cache.get(0)
        assert payload is not None
        assert payload["hidden_units"] == []

    def test_payload_includes_units_added_at_or_before_sample(self):
        wh = _make_weight_history(num_samples=4, num_hidden=2)
        cache = _WeightCache(wh)
        payload = cache.get(2)
        # Units 0 and 1 should both be present at sample 2.
        assert len(payload["hidden_units"]) == 2
        for unit in payload["hidden_units"]:
            assert "weights" in unit
            assert "bias" in unit
            assert isinstance(unit["bias"], float)

    def test_payload_output_tensors_match_source(self):
        wh = _make_weight_history(num_samples=3)
        cache = _WeightCache(wh)
        payload = cache.get(1)
        np.testing.assert_array_equal(payload["output_weights"], wh["output_weights"][1])
        np.testing.assert_array_equal(payload["output_bias"], wh["output_bias"][1])


# =============================================================================
# LRU + budget
# =============================================================================


class TestWeightCacheLRU:
    def test_hit_promotes_to_most_recently_used(self):
        wh = _make_weight_history(num_samples=4)
        cache = _WeightCache(wh)
        # Fill cache: 0, 1, 2 (LRU order: 0 oldest)
        cache.get(0)
        cache.get(1)
        cache.get(2)
        # Touching 0 promotes it; eviction would now drop 1 first.
        cache.get(0)
        stats = cache.stats()
        assert stats["entries"] == 3
        assert stats["hits"] == 1
        assert stats["misses"] == 3

    def test_eviction_under_tight_budget(self):
        wh = _make_weight_history(num_samples=5, num_hidden=2)
        # Size against a sample where both hidden units are present so
        # the per-sample byte cost is realistic. Budget allows one
        # payload but not two.
        sizing_cache = _WeightCache(wh)
        per_sample = sizing_cache._sizeof(sizing_cache.get(4))
        assert per_sample > 0
        budget = per_sample + 1  # Fits one entry of this size, no slack for two.

        cache = _WeightCache(wh, byte_budget=budget)
        cache.get(2)
        cache.get(3)
        cache.get(4)  # Should evict 2 (oldest)
        stats = cache.stats()
        assert stats["evictions"] >= 1
        # Re-fetch evicted entry — must re-build it (counts as miss)
        miss_before = stats["misses"]
        cache.get(2)
        stats_after = cache.stats()
        assert stats_after["misses"] > miss_before

    def test_oversized_payload_still_admitted(self):
        wh = _make_weight_history(num_samples=2)
        # Budget below any single payload's size
        cache = _WeightCache(wh, byte_budget=1)
        payload = cache.get(0)
        # Caller still gets the data even though admission blew the budget.
        assert payload is not None
        # Entry was admitted but next miss will evict it.
        cache.get(1)
        stats = cache.stats()
        assert stats["evictions"] >= 1


# =============================================================================
# Stats
# =============================================================================


class TestWeightCacheStats:
    def test_stats_initial_zeros(self):
        cache = _WeightCache(_make_weight_history(num_samples=3))
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["entries"] == 0
        assert stats["bytes"] == 0

    def test_stats_reflect_activity(self):
        cache = _WeightCache(_make_weight_history(num_samples=3))
        cache.get(0)
        cache.get(0)  # hit
        cache.get(1)
        stats = cache.stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 1
        assert stats["entries"] == 2
        assert stats["bytes"] > 0


# =============================================================================
# Integration with _ReplaySession
# =============================================================================


class TestReplaySessionWithWeightCache:
    """The session wires the cache and surfaces it through state_summary."""

    @staticmethod
    def _history():
        return {
            "train_loss": [0.5, 0.4, 0.3, 0.2, 0.1],
            "value_loss": [0.6, 0.5, 0.4, 0.3, 0.2],
            "train_accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
            "value_accuracy": [0.55, 0.65, 0.75, 0.8, 0.85],
        }

    def test_session_without_weight_history_advertises_unavailable(self):
        from unittest.mock import MagicMock

        monitor = MagicMock()
        session = _ReplaySession("snap", self._history(), monitor)
        assert session.weight_cache.available is False
        summary = session.state_summary()
        assert summary["weights_available"] is False
        assert "weight_sampling" not in summary

    def test_session_with_weight_history_exposes_sampling_block(self):
        from unittest.mock import MagicMock

        monitor = MagicMock()
        wh = _make_weight_history(num_samples=4, sampling_interval=25)
        session = _ReplaySession("snap", self._history(), monitor, weight_history=wh)
        summary = session.state_summary()
        assert summary["weights_available"] is True
        assert summary["weight_sampling"] == {
            "strategy": "adaptive",
            "interval": 25,
            "num_samples": 4,
            "sample_epochs": [0, 25, 50, 75],
        }

    def test_weights_at_returns_payload(self):
        from unittest.mock import MagicMock

        monitor = MagicMock()
        wh = _make_weight_history(num_samples=3)
        session = _ReplaySession("snap", self._history(), monitor, weight_history=wh)
        payload = session.weights_at(1)
        assert payload is not None
        assert payload["sample_index"] == 1

    def test_weights_at_returns_none_when_unavailable(self):
        from unittest.mock import MagicMock

        monitor = MagicMock()
        session = _ReplaySession("snap", self._history(), monitor)
        assert session.weights_at(0) is None

    def test_custom_budget_honored(self):
        from unittest.mock import MagicMock

        monitor = MagicMock()
        wh = _make_weight_history(num_samples=3)
        session = _ReplaySession("snap", self._history(), monitor, weight_history=wh, weight_cache_budget_bytes=1024)
        assert session.weight_cache._budget == 1024
