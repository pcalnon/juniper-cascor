"""
A snapshot must survive the removal of a config field it was written with.

``_create_network_from_file`` rebuilds the config with
``CascadeCorrelationConfig(**config_dict)``. Python's strict keyword matching means any
field present in an old snapshot's ``config_json`` but absent from the CURRENT
``CascadeCorrelationConfig.__init__`` raises ``TypeError``, the whole load returns
``None``, and the operator is told only "snapshot could not be deserialized into a
network" -- a message that names nothing, for a snapshot that is completely intact.

Measured in the live archive (2026-08-22, juniper-ml#1254): **14** snapshots written by
cascor 0.3.2 carry ``optimizer_config``, a real config field at the time that has since
been removed. All 14 were unloadable. They are not damaged in any way.

WHY THE OLD DENYLIST COULD NOT FIX THIS
    The loader already popped five known runtime-only keys by name. A denylist has to be
    extended by hand at every field removal and nothing prompts anyone to do so, so each
    removal silently bricks whichever slice of the archive still carries that field. That
    is a time bomb, not a one-off: snapshots are long-lived project assets and the config
    schema is not frozen. The fix derives an ALLOWLIST from the class itself, so it cannot
    fall behind.

WHY THE EXPLICIT DROPS STAY
    ``activation_functions_dict`` and ``log_config`` ARE accepted by ``__init__``, so an
    allowlist keeps them -- but ``json.dumps`` ran with ``default=str``, so what comes back
    is a repr string rather than the live object. They must still be dropped by name. The
    two filters do different jobs and neither replaces the other; that is what
    ``test_accepted_but_unusable_fields_are_still_dropped`` pins.
"""

import json
import os
import sys

import h5py
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig  # noqa: E402
from snapshots.snapshot_serializer import CascadeHDF5Serializer  # noqa: E402

pytestmark = pytest.mark.unit


def _network():
    torch.manual_seed(11)
    config = CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=11)
    return CascadeCorrelationNetwork(config=config)


def _save(tmp_path, name="drift"):
    path = tmp_path / f"{name}.h5"
    assert CascadeHDF5Serializer().save_network(_network(), str(path))
    return path


def _record_warnings(serializer):
    """Capture what the serializer warns, via its own logger.

    cascor's ``Logger`` is not a stdlib logger wired to the root, so ``caplog`` sees
    nothing from it -- an assertion through caplog would pass no matter what the code
    did. Recording the call directly is the only form that actually pins the behaviour.
    """
    captured = []
    original = serializer.logger.warning

    def _spy(message, *args, **kwargs):
        captured.append(str(message))
        return original(message, *args, **kwargs)

    serializer.logger.warning = _spy
    return captured


def _inject_config_field(path, key, value):
    """Rewrite ``config_json`` in place to carry an extra field.

    Editing the stored JSON rather than the live config object is the point: the defect
    is about reading a snapshot written by a DIFFERENT, older version of the class, and
    that version cannot be constructed from the current one.
    """
    with h5py.File(path, "r") as handle:
        raw = handle["config"]["config_json"][()]
    config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    config[key] = value
    with h5py.File(path, "a") as handle:
        del handle["config"]["config_json"]
        handle["config"].create_dataset("config_json", data=json.dumps(config))


class TestRemovedConfigFieldStillLoads:
    def test_snapshot_with_a_since_removed_field_loads(self, tmp_path):
        """The live case: cascor 0.3.2's ``optimizer_config``, 14 archive snapshots."""
        path = _save(tmp_path)
        _inject_config_field(path, "optimizer_config", {"name": "adam", "lr": 0.01})
        network = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        assert network is not None, "a snapshot carrying a since-removed config field must still load"
        assert network.output_size == 2

    def test_the_load_reports_ok_not_a_generic_failure(self, tmp_path):
        path = _save(tmp_path)
        _inject_config_field(path, "optimizer_config", {"name": "adam"})
        result = CascadeHDF5Serializer().load_network_result(str(path), restore_multiprocessing=False)
        assert result.status == "ok", f"expected a clean load, got {result.status}: {result.detail}"

    def test_several_unknown_fields_are_all_dropped(self, tmp_path):
        """One unknown field is a coincidence; the guard has to hold for a schema that
        moved several times between the write and the read."""
        path = _save(tmp_path)
        for index, key in enumerate(("optimizer_config", "legacy_pruning_mode", "gone_in_v4")):
            _inject_config_field(path, key, index)
        assert CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False) is not None

    def test_a_dropped_field_is_reported_not_silent(self, tmp_path):
        """Dropping data silently is how the NEXT investigation starts from nothing.

        Asserted against the serializer's OWN logger rather than ``caplog``: cascor
        installs a custom ``Logger`` that does not propagate to the root logger pytest
        hooks, so a ``caplog`` assertion here passes vacuously whether or not the warning
        is emitted -- it would have pinned nothing.
        """
        path = _save(tmp_path)
        _inject_config_field(path, "optimizer_config", {"name": "adam"})
        serializer = CascadeHDF5Serializer()
        warnings = _record_warnings(serializer)
        serializer.load_network(str(path), restore_multiprocessing=False)
        assert any("optimizer_config" in message for message in warnings), f"the dropped field must be named in a warning; got {warnings}"

    def test_a_clean_snapshot_logs_no_drop_warning(self, tmp_path):
        """The warning must mean something when it appears."""
        path = _save(tmp_path)
        serializer = CascadeHDF5Serializer()
        warnings = _record_warnings(serializer)
        serializer.load_network(str(path), restore_multiprocessing=False)
        assert not any("no longer accepts" in message for message in warnings)


class TestKnownFieldsSurvive:
    def test_real_config_values_are_not_collateral(self, tmp_path):
        """The allowlist must drop only the unknown -- a filter that ate real fields would
        silently reset the network to construction-time defaults, which is the CAN-014
        failure class the config group exists to prevent."""
        path = _save(tmp_path)
        _inject_config_field(path, "optimizer_config", {"name": "adam"})
        network = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        assert network.input_size == 2
        assert network.output_size == 2
        assert network.random_seed == 11

    def test_accepted_but_unusable_fields_are_still_dropped(self, tmp_path):
        """``activation_functions_dict`` IS accepted by ``__init__``, so the allowlist
        keeps it -- but it round-trips through ``json.dumps(default=str)`` as a repr
        string. If the explicit drop were removed in favour of the allowlist, that string
        would reach the constructor. Pins that the two filters are not interchangeable."""
        path = _save(tmp_path)
        _inject_config_field(path, "activation_functions_dict", "<function tanh at 0x7f00>")
        network = CascadeHDF5Serializer().load_network(str(path), restore_multiprocessing=False)
        assert network is not None
        assert not isinstance(getattr(network, "activation_functions_dict", None), str), "the stringified activation map must never reach the network"


class TestTheFilterCannotMisfire:
    """A filter that gets its allowlist wrong drops REAL config silently, which is worse
    than the TypeError it exists to prevent. Both escape hatches are pinned."""

    def test_a_kwargs_accepting_constructor_skips_the_filter(self, tmp_path, monkeypatch):
        """If ``__init__`` reports ``**kwargs`` -- one decorator without ``functools.wraps``
        is enough -- then every name is legal and ``TypeError`` cannot happen. Filtering
        against that signature would compute an allowlist of ``{args, kwargs}`` and strip
        the ENTIRE config, resetting the network to construction-time defaults."""
        import snapshots.snapshot_serializer as serializer_module

        path = _save(tmp_path)
        _inject_config_field(path, "optimizer_config", {"name": "adam"})

        real_signature = serializer_module.inspect.signature

        def _kwargs_signature(target):
            if getattr(target, "__qualname__", "").startswith("CascadeCorrelationConfig"):
                return real_signature(lambda *args, **kwargs: None)
            return real_signature(target)

        monkeypatch.setattr(serializer_module.inspect, "signature", _kwargs_signature)
        serializer = CascadeHDF5Serializer()
        warnings = _record_warnings(serializer)
        serializer.load_network(str(path), restore_multiprocessing=False)
        assert not any("no longer accepts" in message for message in warnings), "a **kwargs constructor must not have its config filtered"

    def test_an_uninspectable_constructor_falls_back_instead_of_failing(self, tmp_path, monkeypatch):
        """Introspection failure must degrade to the pre-existing behaviour, not break
        every load in the system."""
        import snapshots.snapshot_serializer as serializer_module

        path = _save(tmp_path)

        def _boom(target):
            raise ValueError("no signature found")

        monkeypatch.setattr(serializer_module.inspect, "signature", _boom)
        serializer = CascadeHDF5Serializer()
        warnings = _record_warnings(serializer)
        network = serializer.load_network(str(path), restore_multiprocessing=False)
        assert network is not None, "an uninspectable config class must not make snapshots unloadable"
        assert any("could not introspect" in message for message in warnings)
