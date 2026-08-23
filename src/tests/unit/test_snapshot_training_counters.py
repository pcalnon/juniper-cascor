"""
A training counter must be a MEASUREMENT or absent — never a fabricated default.

``_save_metadata`` wrote four counters with ``getattr(network, name, default)``. Three of the
four attributes are never assigned anywhere in ``CascadeCorrelationNetwork``, so the defaults
were written on every save. Measured over the live archive (juniper-ml#1254), across **all
27,908** snapshots:

    current_epoch     0     <- self.current_epoch is never assigned
    patience_counter  0     <- self.patience_counter is never assigned
    best_value_loss   inf   <- self.best_value_loss is never assigned
    snapshot_counter  LIVE  <- this one IS assigned, and increments correctly (0 -> 109
                               across one network's growth curve)

WHY THAT IS WORSE THAN MISSING DATA
    A fabricated default is INDISTINGUISHABLE from a real measurement. A reader cannot tell
    "training never improved" from "nobody ever wrote this down". Reading the archive
    literally says every snapshot sits at epoch 0 with no best loss -- that nothing ever
    trained -- and that reading nearly justified deleting 27,005 real models. It only came
    apart on checking a network known to have grown to 260 hidden units.

    A missing key is a question a reader can ask. A fabricated default is an answer they
    cannot check.

THE FIX HAS TWO HALVES, AND BOTH ARE NEEDED
    1. ``grow_network`` publishes ``best_value_loss`` / ``patience_counter`` onto the instance
       (it already computed them, as locals, to drive early stopping).
    2. ``_save_metadata`` writes a counter ONLY when the network actually carries it, so an
       attribute nobody maintains shows up as absent instead of as a plausible zero.

    Half 1 alone would fix these two fields and leave the next un-maintained counter to
    fabricate silently. Half 2 alone would make the archive honest but still empty.
    ``TestAbsentCountersAreOmitted`` is the load-bearing class.
"""

import os
import sys
import tempfile

import h5py
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _scratch_snapshot_dir(monkeypatch, tmp_path):
    """``train_output_layer`` calls ``create_snapshot()`` unconditionally, so anything that
    trains here would otherwise write into the shared archive."""
    monkeypatch.setenv("JUNIPER_CASCOR_SNAPSHOTS_DIR", str(tmp_path / "scratch"))


def _network(**overrides):
    torch.manual_seed(5)
    settings = {"input_size": 2, "output_size": 2, "random_seed": 5}
    settings.update(overrides)
    return CascadeCorrelationNetwork(config=CascadeCorrelationConfig(**settings))


def _meta(path):
    with h5py.File(path, "r") as handle:
        return dict(handle["meta"].attrs)


def _save(network, tmp_path, name="snap"):
    path = tmp_path / f"{name}.h5"
    assert CascadeHDF5Serializer().save_network(network, str(path))
    return path


class TestAbsentCountersAreOmitted:
    """THE load-bearing class: absence must look like absence.

    Without this, the next counter nobody maintains silently becomes a plausible zero, and the
    whole failure class returns under a different field name.
    """

    def test_an_attribute_the_model_never_sets_is_not_written(self, tmp_path):
        """``current_epoch`` is never assigned in CascadeCorrelationNetwork. It must be ABSENT
        from the snapshot, not written as 0."""
        network = _network()
        assert not hasattr(network, "current_epoch"), "precondition: the model does not maintain current_epoch"
        meta = _meta(_save(network, tmp_path))
        assert "current_epoch" not in meta, "an attribute nobody maintains must not be fabricated as a measurement"

    def test_a_counter_the_model_does_set_is_written(self, tmp_path):
        """The omission must be driven by absence, not by a hard-coded exclusion list."""
        network = _network()
        assert hasattr(network, "snapshot_counter")
        assert "snapshot_counter" in _meta(_save(network, tmp_path))

    def test_setting_the_attribute_makes_it_appear(self, tmp_path):
        """Proves the rule is 'write what exists' rather than 'never write this field'."""
        network = _network()
        network.current_epoch = 17
        meta = _meta(_save(network, tmp_path))
        assert meta.get("current_epoch") == 17

    def test_a_snapshot_without_the_key_still_loads(self, tmp_path):
        """Omitting a key must not break the loader -- every future snapshot omits some."""
        path = _save(_network(), tmp_path)
        assert "current_epoch" not in _meta(path)
        assert CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False) is not None


class TestBestValueLossIsMeasured:
    """The reported ask: ``best_value_loss`` must carry a real number after training."""

    def test_it_is_no_longer_inf_after_training(self, tmp_path):
        module = __import__("juniper_data.generators.xor", fromlist=["x"]) if _has_juniper_data() else None
        x, y = _training_data(module)
        network = _network(max_hidden_units=2, candidate_pool_size=3, candidate_epochs=40, output_epochs=60, max_iterations=2)
        network.fit(x, y, max_epochs=60, max_iterations=2)
        assert hasattr(network, "best_value_loss"), "grow_network must publish the value it already computes"
        meta = _meta(_save(network, tmp_path))
        assert "best_value_loss" in meta
        assert meta["best_value_loss"] != float("inf"), "a trained network must not report an untouched best loss"

    def test_patience_counter_is_published_too(self, tmp_path):
        """Same line, same cause -- it was a local beside best_value_loss."""
        x, y = _training_data(None)
        network = _network(max_hidden_units=2, candidate_pool_size=3, candidate_epochs=40, output_epochs=60, max_iterations=2)
        network.fit(x, y, max_epochs=60, max_iterations=2)
        assert hasattr(network, "patience_counter")
        assert "patience_counter" in _meta(_save(network, tmp_path))

    def test_an_untrained_network_reports_no_best_loss_at_all(self, tmp_path):
        """The distinction the archive lost: 'never trained' must not look like 'trained and
        never improved'. Both used to serialize as inf."""
        meta = _meta(_save(_network(), tmp_path))
        assert "best_value_loss" not in meta

    def test_the_value_round_trips(self, tmp_path):
        x, y = _training_data(None)
        network = _network(max_hidden_units=2, candidate_pool_size=3, candidate_epochs=40, output_epochs=60, max_iterations=2)
        network.fit(x, y, max_epochs=60, max_iterations=2)
        path = _save(network, tmp_path)
        restored = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        assert restored.best_value_loss == pytest.approx(network.best_value_loss)


def _has_juniper_data() -> bool:
    return True


def _training_data(_module):
    """A tiny XOR-shaped problem. Kept local so the suite needs no juniper-data tree."""
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 25, dtype=torch.float32)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]] * 25, dtype=torch.float32)
    return x, y
