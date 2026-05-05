"""Fast-fit unit test exercising the complete training pipeline (CR-071).

Existing coverage tests bypass fit()/grow_network() to avoid timeouts,
leaving the most critical code path untested in standard CI runs. This
module uses ultra-minimal parameters to exercise the full path in seconds:

    fit() → train_output_layer() → grow_network() → train_candidates()
          → _execute_candidate_training() → add_unit() → train_output_layer()
"""

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig


@pytest.fixture
def ultrafast_network():
    """Network with ultra-minimal params for fast full-path testing."""
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        learning_rate=0.1,
        candidate_learning_rate=0.1,
        candidate_pool_size=1,
        candidate_epochs=1,
        output_epochs=1,
        max_hidden_units=1,
        patience=100,
        correlation_threshold=0.0,
    )
    return CascadeCorrelationNetwork(config=config)


@pytest.fixture
def tiny_data():
    """Minimal 2-class dataset (8 samples)."""
    torch.manual_seed(42)
    x = torch.randn(8, 2)
    y = torch.zeros(8, 2)
    y[:4, 0] = 1.0
    y[4:, 1] = 1.0
    return x, y


# V38a/V38c — flipped back to ``@pytest.mark.unit`` after the P-1 RC-4
# fix landed. The three formerly-flaky tests (test_fit_executes_full_training_path,
# test_fit_output_weights_grow_after_unit_addition, test_fit_multiple_iterations)
# tripped a multiprocessing-timing heisenbug in
# ``_collect_worker_results`` whose qsize-based early-exit raced ahead
# of in-flight ``multiprocessing.Queue`` feeder writes under sub-second
# per-candidate budgets. The fix dropped the qsize-poll wait loop in
# favour of letting ``_collect_training_results`` block on the queue's
# own semaphore, which is correctly synchronized with worker put()s.
# See ``notes/P1_RC4_INVESTIGATION_PLAN_2026-05-03.md`` (juniper-ml) §4
# for the fix-shape rationale and §3.2 for the ring-buffer instrumentation
# that's still in place to catch any residual race signature.
@pytest.mark.timeout(30)
class TestFastFit:
    """Tests that exercise the complete fit() training pipeline."""

    @pytest.mark.unit
    def test_fit_executes_full_training_path(self, ultrafast_network, tiny_data):
        """fit() must execute: output training → grow_network → candidate training → add_unit."""
        x, y = tiny_data
        history = ultrafast_network.fit(x, y, max_epochs=2, max_iterations=1)

        assert history is not None
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
        assert len(ultrafast_network.hidden_units) == 1, "grow_network should have added exactly 1 hidden unit"

    @pytest.mark.unit
    def test_fit_with_validation_data(self, ultrafast_network, tiny_data):
        """fit() with validation data exercises the validation early stopping path."""
        x, y = tiny_data
        x_val = torch.randn(4, 2)
        y_val = torch.zeros(4, 2)
        y_val[:2, 0] = 1.0
        y_val[2:, 1] = 1.0

        history = ultrafast_network.fit(x, y, x_val=x_val, y_val=y_val, max_epochs=2, max_iterations=1)

        assert "value_loss" in history
        assert len(history["value_loss"]) > 0

    @pytest.mark.unit
    def test_fit_output_weights_grow_after_unit_addition(self, ultrafast_network, tiny_data):
        """Output weight matrix must grow by 1 row when a hidden unit is added."""
        x, y = tiny_data
        initial_rows = ultrafast_network.output_weights.shape[0]

        ultrafast_network.fit(x, y, max_epochs=2, max_iterations=1)

        assert ultrafast_network.output_weights.shape[0] == initial_rows + 1

    @pytest.mark.unit
    def test_fit_history_tracks_accuracy(self, ultrafast_network, tiny_data):
        """fit() must record train_accuracy in history."""
        x, y = tiny_data
        history = ultrafast_network.fit(x, y, max_epochs=2, max_iterations=1)

        assert "train_accuracy" in history
        assert len(history["train_accuracy"]) > 0

    @pytest.mark.unit
    def test_fit_multiple_iterations(self, tiny_data):
        """fit() with max_iterations=2 and early_stopping=False should add 2 hidden units."""
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=2,
            learning_rate=0.1,
            candidate_learning_rate=0.1,
            candidate_pool_size=1,
            candidate_epochs=1,
            output_epochs=1,
            max_hidden_units=2,
            patience=100,
            correlation_threshold=0.0,
        )
        network = CascadeCorrelationNetwork(config=config)
        x, y = tiny_data

        network.fit(x, y, max_epochs=2, max_iterations=2, early_stopping=False)

        assert len(network.hidden_units) == 2
