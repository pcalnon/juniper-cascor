"""P2-2 (Issue #3) — ``CascadeCorrelationNetwork.record_dataset_swap_event``.

Covers the network-level history-recording surface defined in
``notes/PHASE_2_P2_1D_DESIGN_2026-05-13.md`` and pinned by the §3.9 parent-
spec contract. The recording is fire-and-forget from the lifecycle's
perspective — these tests verify the payload schema, the snapshot-ID
placeholder semantics, and that mutating the caller's input after
recording does not affect the persisted event (deep-copy guarantee).
"""

from __future__ import annotations

import pytest

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

pytestmark = pytest.mark.unit


def _make_network() -> CascadeCorrelationNetwork:
    cfg = CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=42)
    return CascadeCorrelationNetwork(cfg)


def _sample_arch_changes() -> dict:
    """Reference §3.3 arch_changes payload as written by P2-1d's
    ``swap_dataset_live`` step 16. Mirroring the live shape here keeps the
    network method's contract pinned against the lifecycle's expectations."""
    return {
        "input_delta": 2,
        "output_delta": 0,
        "hidden_preserved": 3,
        "appended_nodes": {"input": 2, "output": 0},
        "prepended_layers": [],
        "abandoned_candidate_pool_size": 5,
        "active_output_dim": 2,
    }


class TestRecordDatasetSwapEvent:
    def test_initial_history_has_empty_dataset_swaps_list(self):
        """Sibling of ``hidden_units_added``: present at construction
        with empty list. Without this, downstream code would have to
        defensively ``setdefault`` everywhere."""
        net = _make_network()
        assert "dataset_swaps" in net.history
        assert net.history["dataset_swaps"] == []

    def test_records_event_with_schema_fields(self):
        net = _make_network()
        before = {"dataset_type": "spirals", "n_spirals": 2}
        after = {"dataset_type": "moons", "noise": 0.2}
        result = net.record_dataset_swap_event(
            before_cfg=before,
            after_cfg=after,
            arch_changes=_sample_arch_changes(),
        )
        assert isinstance(result, dict)
        # Schema: all six required keys present.
        assert set(result.keys()) == {
            "timestamp",
            "before_cfg",
            "after_cfg",
            "arch_changes",
            "pre_swap_snapshot_id",
            "post_swap_snapshot_id",
        }
        # Timestamp is ISO-8601 UTC (ends in +00:00).
        assert isinstance(result["timestamp"], str)
        assert "T" in result["timestamp"]  # date/time separator
        # before_cfg / after_cfg copy the input (equal but distinct object).
        assert result["before_cfg"] == before
        assert result["after_cfg"] == after
        # arch_changes deep-copied (top-level + nested ``appended_nodes``).
        assert result["arch_changes"] == _sample_arch_changes()
        # Snapshot IDs default to None (P2-3 backfill).
        assert result["pre_swap_snapshot_id"] is None
        assert result["post_swap_snapshot_id"] is None
        # Appended to history.
        assert len(net.history["dataset_swaps"]) == 1
        assert net.history["dataset_swaps"][0] is result

    def test_records_multiple_events_in_order(self):
        """Each call appends; chronological order preserved."""
        net = _make_network()
        for i in range(3):
            net.record_dataset_swap_event(
                before_cfg={"step": i},
                after_cfg={"step": i + 1},
                arch_changes=_sample_arch_changes(),
            )
        assert len(net.history["dataset_swaps"]) == 3
        assert [e["before_cfg"]["step"] for e in net.history["dataset_swaps"]] == [0, 1, 2]

    def test_caller_mutation_after_record_does_not_affect_event(self):
        """The deep-copy guarantee: mutating the caller's dicts after the
        record call must NOT ripple into the persisted event. Without this,
        a callsite that reuses a dict across swaps would silently corrupt
        the history list."""
        net = _make_network()
        before = {"dataset_type": "spirals", "n_spirals": 2}
        arch = _sample_arch_changes()
        net.record_dataset_swap_event(before_cfg=before, after_cfg=None, arch_changes=arch)
        # Mutate the caller's dicts post-record.
        before["dataset_type"] = "MUTATED"
        arch["input_delta"] = 999
        arch["appended_nodes"]["input"] = 999  # nested mutation
        # Persisted event unchanged.
        ev = net.history["dataset_swaps"][0]
        assert ev["before_cfg"]["dataset_type"] == "spirals"
        assert ev["arch_changes"]["input_delta"] == 2
        assert ev["arch_changes"]["appended_nodes"]["input"] == 2

    def test_records_none_dataset_configs(self):
        """``before_cfg`` / ``after_cfg`` can be None (the lifecycle passes
        None when ``_current_dataset_config`` is unset, e.g. on first swap
        before any dataset metadata is tracked)."""
        net = _make_network()
        ev = net.record_dataset_swap_event(
            before_cfg=None,
            after_cfg=None,
            arch_changes=_sample_arch_changes(),
        )
        assert ev["before_cfg"] is None
        assert ev["after_cfg"] is None

    def test_records_with_explicit_snapshot_ids(self):
        """Snapshot ID fields can be populated by callers (P2-3 will do this
        when auto-snap-pre/post-swap is wired). Tests the path that P2-2
        leaves dormant but doesn't disable."""
        net = _make_network()
        ev = net.record_dataset_swap_event(
            before_cfg=None,
            after_cfg=None,
            arch_changes=_sample_arch_changes(),
            pre_swap_snapshot_id="snap-pre-123",
            post_swap_snapshot_id="snap-post-456",
        )
        assert ev["pre_swap_snapshot_id"] == "snap-pre-123"
        assert ev["post_swap_snapshot_id"] == "snap-post-456"
