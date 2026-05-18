#!/usr/bin/env python
"""P2-2 (Issue #3): HDF5 round-trip for ``dataset_swap`` history events.

Covers:

* Round-trip of a synthetic ``network.history["dataset_swaps"]`` payload
  through ``CascadeHDF5Serializer.save_network`` + ``load_network``.
* Empty-list case: a network with no swaps still produces a valid snapshot
  that loads with an empty ``dataset_swaps`` list (the construction-time
  default).
* Backward compat: snapshots written before P2-2 (no ``dataset_swaps``
  HDF5 group) load successfully and initialise the list as empty.
* Chronological order preserved across save/load.
* Malformed JSON in one event does NOT kill the whole history load —
  the bad event degrades to schema defaults with a warning, other
  events restore cleanly.
* Mixed populated + None ``before_cfg`` / ``after_cfg`` / snapshot ID
  fields round-trip with the schema-faithful None on the missing sides.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import h5py
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_serializer import CascadeHDF5Serializer

pytestmark = pytest.mark.unit


@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def serializer():
    return CascadeHDF5Serializer()


@pytest.fixture
def simple_network():
    config = CascadeCorrelationConfig.create_simple_config(input_size=2, output_size=2, learning_rate=0.1, max_hidden_units=3, random_seed=42)
    return CascadeCorrelationNetwork(config=config)


def _sample_event(i: int = 0) -> dict:
    """Reference §3.3-shaped event; ``i`` lets a test build a small ordered
    sequence to verify chronology survives the round-trip."""
    return {
        "timestamp": f"2026-05-14T00:0{i}:00+00:00",
        "before_cfg": {"dataset_type": "spirals", "n_spirals": 2 + i},
        "after_cfg": {"dataset_type": "moons", "noise": 0.1 * i},
        "arch_changes": {
            "input_delta": i,
            "output_delta": 0,
            "hidden_preserved": 3,
            "appended_nodes": {"input": i, "output": 0},
            "prepended_layers": [],
            "abandoned_candidate_pool_size": 5,
            "active_output_dim": 2,
        },
        "pre_swap_snapshot_id": None,
        "post_swap_snapshot_id": None,
    }


class TestDatasetSwapsRoundTrip:
    def test_empty_dataset_swaps_round_trips_as_empty_list(self, serializer, simple_network, temp_file):
        """Default construction has an empty list; snapshot-then-load must
        not invent events or drop the key."""
        assert simple_network.history["dataset_swaps"] == []
        assert serializer.save_network(simple_network, temp_file, include_training_state=True) is True
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert loaded.history["dataset_swaps"] == []

    def test_single_event_round_trips(self, serializer, simple_network, temp_file):
        simple_network.history["dataset_swaps"] = [_sample_event(0)]
        assert serializer.save_network(simple_network, temp_file, include_training_state=True) is True
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert len(loaded.history["dataset_swaps"]) == 1
        # Compare via JSON (handles nested dict equality + key ordering).
        assert json.dumps(loaded.history["dataset_swaps"][0], sort_keys=True) == json.dumps(_sample_event(0), sort_keys=True)

    def test_multiple_events_preserve_chronological_order(self, serializer, simple_network, temp_file):
        """The save writes ``event_{i}`` subgroups; the load sorts by the
        numeric suffix so the on-disk dict iteration order can't scramble
        chronology. Pinned here because HDF5 group iteration order is
        not guaranteed otherwise."""
        events = [_sample_event(i) for i in range(5)]
        simple_network.history["dataset_swaps"] = events
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert [ev["arch_changes"]["input_delta"] for ev in loaded.history["dataset_swaps"]] == [0, 1, 2, 3, 4]

    def test_none_dataset_configs_round_trip(self, serializer, simple_network, temp_file):
        """``before_cfg`` / ``after_cfg`` are encoded via ``json.dumps`` so
        ``None`` becomes the literal string ``"null"`` on disk; the load
        path must decode that back to Python ``None`` (not the string
        ``"null"``)."""
        ev = _sample_event(0)
        ev["before_cfg"] = None
        ev["after_cfg"] = None
        simple_network.history["dataset_swaps"] = [ev]
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert loaded.history["dataset_swaps"][0]["before_cfg"] is None
        assert loaded.history["dataset_swaps"][0]["after_cfg"] is None

    def test_populated_snapshot_ids_round_trip(self, serializer, simple_network, temp_file):
        """P2-3 will populate snapshot IDs; P2-2 leaves the path live but
        defaulted to None. This test exercises the populated branch so the
        attr is written-and-read symmetrically."""
        ev = _sample_event(0)
        ev["pre_swap_snapshot_id"] = "snap-pre-abc"
        ev["post_swap_snapshot_id"] = "snap-post-def"
        simple_network.history["dataset_swaps"] = [ev]
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert loaded.history["dataset_swaps"][0]["pre_swap_snapshot_id"] == "snap-pre-abc"
        assert loaded.history["dataset_swaps"][0]["post_swap_snapshot_id"] == "snap-post-def"


class TestDatasetSwapsBackwardCompat:
    def test_pre_p2_2_snapshot_loads_with_empty_list(self, serializer, simple_network, temp_file):
        """Simulate a snapshot from before P2-2 by saving normally, then
        deleting the ``dataset_swaps`` HDF5 group. The loader must yield
        an empty list, not raise or skip the ``dataset_swaps`` key."""
        # Save once with a default empty list.
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        # Remove the dataset_swaps group (simulating a snapshot vintage
        # before P2-2 was written). Note: an EMPTY ``dataset_swaps`` list
        # produces NO group on disk (the save path skips empties), so this
        # is also the case for the "saved with P2-2 code but no swaps".
        with h5py.File(temp_file, "a") as f:
            if "history/dataset_swaps" in f:
                del f["history/dataset_swaps"]
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        # Schema invariant: the key exists with an empty list, not absent.
        assert loaded.history["dataset_swaps"] == []


class TestDatasetSwapsCorruptionTolerance:
    def test_malformed_arch_changes_json_degrades_to_default(self, serializer, simple_network, temp_file):
        """A single corrupt event must not break the whole history load.
        Pin the graceful-degrade contract from the load path's
        try/except around the JSON decode."""
        simple_network.history["dataset_swaps"] = [_sample_event(0), _sample_event(1)]
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        # Corrupt the ``arch_changes`` JSON for the first event.
        with h5py.File(temp_file, "a") as f:
            ev_group = f["history/dataset_swaps/event_0"]
            ev_group.attrs["arch_changes"] = "{not valid json"
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        # Both events present; corrupt one has default-empty arch_changes;
        # other event intact.
        assert len(loaded.history["dataset_swaps"]) == 2
        assert loaded.history["dataset_swaps"][0]["arch_changes"] == {}
        assert loaded.history["dataset_swaps"][1]["arch_changes"]["input_delta"] == 1


class TestReadDatasetSwapEvents:
    """P2-7 follow-up: ``read_dataset_swap_events`` reader used by
    ``GET /v1/snapshots/{id}/history/dataset_swaps``."""

    def test_reads_persisted_events_without_full_load(self, serializer, simple_network, temp_file):
        """The reader returns the same schema the full-network loader does,
        without restoring the network. Verifies the route can skip the
        expensive load path for marker rendering."""
        simple_network.history["dataset_swaps"] = [_sample_event(0), _sample_event(1), _sample_event(2)]
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        events = serializer.read_dataset_swap_events(temp_file)
        assert len(events) == 3
        assert [e["arch_changes"]["input_delta"] for e in events] == [0, 1, 2]
        for e in events:
            assert set(e.keys()) == {"timestamp", "before_cfg", "after_cfg", "arch_changes", "pre_swap_snapshot_id", "post_swap_snapshot_id"}

    def test_empty_list_for_snapshot_with_no_swaps(self, serializer, simple_network, temp_file):
        """A snapshot saved with no dataset_swaps (the writer skips empty
        lists) returns ``[]`` rather than raising — pre-P2-2 snapshots
        reach this branch too."""
        simple_network.history["dataset_swaps"] = []
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        events = serializer.read_dataset_swap_events(temp_file)
        assert events == []

    def test_empty_list_when_history_group_absent(self, serializer, simple_network, temp_file):
        """A snapshot saved without training_state has no ``history``
        group at all — must still return ``[]`` rather than raising
        KeyError."""
        serializer.save_network(simple_network, temp_file, include_training_state=False)
        events = serializer.read_dataset_swap_events(temp_file)
        assert events == []

    def test_chronological_order_matches_loader(self, serializer, simple_network, temp_file):
        """Reader sort key must match the full-network loader so canopy's
        timeline draws markers in the same order the snapshot's network
        history would surface them."""
        simple_network.history["dataset_swaps"] = [_sample_event(0), _sample_event(1), _sample_event(2)]
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        events = serializer.read_dataset_swap_events(temp_file)
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert [e["timestamp"] for e in events] == [e["timestamp"] for e in loaded.history["dataset_swaps"]]

    def test_corrupt_event_degrades_consistently_with_loader(self, serializer, simple_network, temp_file):
        """Corruption tolerance must be identical between the reader and
        the full-network loader so a snapshot rendering its own markers
        doesn't disagree with a freshly-loaded network's history."""
        simple_network.history["dataset_swaps"] = [_sample_event(0), _sample_event(1)]
        serializer.save_network(simple_network, temp_file, include_training_state=True)
        with h5py.File(temp_file, "a") as f:
            f["history/dataset_swaps/event_0"].attrs["arch_changes"] = "{not valid json"
        events = serializer.read_dataset_swap_events(temp_file)
        loaded = serializer.load_network(temp_file)
        assert loaded is not None
        assert events[0]["arch_changes"] == loaded.history["dataset_swaps"][0]["arch_changes"] == {}
        assert events[1]["arch_changes"]["input_delta"] == 1
