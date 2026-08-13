#!/usr/bin/env python
"""The RC-4 candidate pool is released when the *run* ends (issue #509).

RC-4 keeps candidate workers alive **across growth rounds** — that is the
optimization, and these tests do not challenge it. What was missing is the other
end of that lifetime: ``_shutdown_worker_pool``'s only production caller was
``_ensure_worker_pool`` recycling a stale pool, so a *healthy* pool outlived
every run that created it. The orphaned forkserver children keep a CUDA context
(~116 MiB each), reparent to ``systemd --user``, and accumulate until the card is
full — at which point every candidate dies with ``AcceleratorError`` and runs
report plausible results computed from nothing.

These are fast unit tests driven by mock seams: ``grow_network`` is patched out
and ``_shutdown_worker_pool`` is a mock, so no worker process is ever spawned.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

import cascade_correlation.cascade_correlation as cc_mod
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

pytestmark = pytest.mark.unit


def _make_network():
    config = CascadeCorrelationConfig(input_size=2, output_size=2, candidate_pool_size=2, candidate_epochs=1)
    return CascadeCorrelationNetwork(config=config)


def _xy():
    torch.manual_seed(0)
    return torch.randn(8, 2), torch.randn(8, 2)


class TestFitReleasesPool:
    """``fit`` ends the pool's persistence, on every exit path."""

    def test_successful_fit_releases_the_pool(self):
        """The growth rounds are over, so the pool's reason to persist is too."""
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "grow_network", return_value=None), patch.object(net, "_shutdown_worker_pool") as shutdown:
            net.fit(x_train=x, y_train=y, max_epochs=1, max_iterations=1)
        shutdown.assert_called_once()

    def test_failed_fit_still_releases_the_pool(self):
        """The ``finally`` arm — a failed run is when leaked children do the most damage.

        Issue #509's honest-outcome half made this path live: a round in which
        every candidate errors now raises out of ``grow_network``, and that is
        precisely the GPU-pressure situation where the next run must not inherit
        more orphans.
        """
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "grow_network", side_effect=RuntimeError("training blew up")), patch.object(net, "_shutdown_worker_pool") as shutdown:
            with pytest.raises(RuntimeError, match="training blew up"):
                net.fit(x_train=x, y_train=y, max_epochs=1, max_iterations=1)
        shutdown.assert_called_once()

    def test_release_failure_does_not_mask_the_training_error(self):
        """Cleanup must never overwrite the outcome of the work it follows."""
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "grow_network", side_effect=RuntimeError("the real failure")), patch.object(net, "_shutdown_worker_pool", side_effect=OSError("teardown also failed")):
            with pytest.raises(RuntimeError, match="the real failure"):
                net.fit(x_train=x, y_train=y, max_epochs=1, max_iterations=1)

    def test_release_failure_does_not_fail_a_good_run(self):
        """A teardown problem is logged, not raised — the training result stands."""
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "grow_network", return_value=None), patch.object(net, "_shutdown_worker_pool", side_effect=OSError("teardown failed")):
            history = net.fit(x_train=x, y_train=y, max_epochs=1, max_iterations=1)
        assert history is net.history


class TestReleaseHelper:
    """``_release_candidate_worker_pool`` is the guarded wrapper."""

    def test_delegates_to_shutdown_worker_pool(self):
        net = _make_network()
        with patch.object(net, "_shutdown_worker_pool") as shutdown:
            net._release_candidate_worker_pool()
        shutdown.assert_called_once()

    def test_swallows_and_logs_shutdown_failure(self):
        net = _make_network()
        net.logger = MagicMock()
        with patch.object(net, "_shutdown_worker_pool", side_effect=OSError("boom")):
            net._release_candidate_worker_pool()  # must not raise
        assert net.logger.warning.called

    def test_no_pool_is_a_noop(self):
        """A network that never trained has nothing to release."""
        net = _make_network()
        net._persistent_workers = []
        net._release_candidate_worker_pool()  # must not raise


class TestAtexitRegistration:
    """The belt-and-braces arm for pools created outside a fit."""

    def test_release_is_registered_at_construction(self):
        with patch.object(cc_mod.atexit, "register") as register:
            net = _make_network()
        registered = {getattr(call.args[0], "__name__", None) for call in register.call_args_list if call.args}
        assert "_release_candidate_worker_pool" in registered
        # The pre-existing OPT-5 shared-memory registration must survive alongside it.
        assert "_cleanup_shared_memory" in registered
        assert net is not None
