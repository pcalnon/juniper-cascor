"""C7 (U-4) phase 1 — unit tests for the torch-native scalar classification
metrics (``api.lifecycle.classification_metrics``).

Correctness is pinned against hand-computed confusion matrices and a hand-ranked
ROC-AUC (no scikit-learn dependency — the module is verified against first
principles), plus the graceful-degradation contract (null + reason) for the
undefined-metric cases the C7 task calls out: single-class batches, empty
batches, and NaN/Inf output.
"""

import math

import pytest
import torch

from api.lifecycle.classification_metrics import (
    METRIC_KEYS,
    compute_classification_accuracy,
    compute_scalar_classification_metrics,
)


def _approx(value, expected, tol=1e-6):
    return value is not None and math.isclose(value, expected, rel_tol=0, abs_tol=tol)


@pytest.mark.unit
class TestBinaryScalarMetrics:
    """Binary (single-logit) head: positive-class P/R/F1 + raw-score ROC-AUC."""

    def test_known_confusion_matrix_exact(self):
        """Hand-computed 2x2 confusion matrix -> exact P/R/F1.

        y_true = [1,1,0,0,1], raw scores = [.9,.4,.2,.6,.8], threshold 0.5
        -> y_pred = [1,0,0,1,1]; positive class: TP=2, FP=1, FN=1
        -> precision = recall = f1 = 2/3.
        """
        output = torch.tensor([[0.9], [0.4], [0.2], [0.6], [0.8]])
        target = torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]])
        result = compute_scalar_classification_metrics(output, target)
        assert result["average"] == "binary"
        assert result["n_classes"] == 2
        assert _approx(result["precision"], 2.0 / 3.0)
        assert _approx(result["recall"], 2.0 / 3.0)
        assert _approx(result["f1"], 2.0 / 3.0)

    def test_known_score_distribution_auc_exact(self):
        """Hand-ranked ROC-AUC on a known score distribution.

        pos scores {.9,.4,.8}, neg scores {.2,.6}; sorted ranks give
        sum_ranks_pos = 2+4+5 = 11; AUC = (11 - 3*4/2)/(3*2) = 5/6.
        """
        output = torch.tensor([[0.9], [0.4], [0.2], [0.6], [0.8]])
        target = torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]])
        result = compute_scalar_classification_metrics(output, target)
        assert _approx(result["roc_auc"], 5.0 / 6.0)

    def test_auc_tie_handling_average_ranks(self):
        """Tied scores across classes use average ranks (Mann-Whitney identity).

        scores = [.5(pos), .5(neg)], one of each -> the tie splits the ordering
        symmetrically, giving AUC = 0.5 exactly.
        """
        output = torch.tensor([[0.5], [0.5]])
        target = torch.tensor([[1.0], [0.0]])
        result = compute_scalar_classification_metrics(output, target)
        assert _approx(result["roc_auc"], 0.5)

    def test_perfect_separation(self):
        """Perfectly separable -> all metrics 1.0."""
        output = torch.tensor([[0.9], [0.8], [0.1], [0.2]])
        target = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        result = compute_scalar_classification_metrics(output, target)
        for key in METRIC_KEYS:
            assert _approx(result[key], 1.0), key

    def test_one_d_shapes_accepted(self):
        """1-D output/target are normalized to (N, 1)."""
        output = torch.tensor([0.9, 0.4, 0.2, 0.6, 0.8])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
        result = compute_scalar_classification_metrics(output, target)
        assert _approx(result["roc_auc"], 5.0 / 6.0)


@pytest.mark.unit
class TestMulticlassScalarMetrics:
    """Multi-class (one-hot) head: argmax decode + macro/weighted aggregation."""

    # y_pred = [0,1,2,0,1,2], y_true = [0,1,2,1,1,2]
    _OUTPUT = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
    _TARGET = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_macro_average_exact(self):
        """Hand-computed macro P/R/F1 over 3 classes.

        per-class precision = [.5, 1., 1.] -> macro 2.5/3
        per-class recall    = [1., 2/3, 1.] -> macro (2+2/3)/3
        per-class f1        = [2/3, .8, 1.] -> macro (2/3+.8+1)/3
        """
        result = compute_scalar_classification_metrics(self._OUTPUT, self._TARGET, average="macro")
        assert result["average"] == "macro"
        assert result["n_classes"] == 3
        assert _approx(result["precision"], (0.5 + 1.0 + 1.0) / 3.0)
        assert _approx(result["recall"], (1.0 + 2.0 / 3.0 + 1.0) / 3.0)
        assert _approx(result["f1"], (2.0 / 3.0 + 0.8 + 1.0) / 3.0)

    def test_weighted_average_exact(self):
        """Hand-computed support-weighted P/R/F1 (support = [1,3,2], total 6)."""
        result = compute_scalar_classification_metrics(self._OUTPUT, self._TARGET, average="weighted")
        assert result["average"] == "weighted"
        assert _approx(result["precision"], (0.5 * 1 + 1.0 * 3 + 1.0 * 2) / 6.0)
        assert _approx(result["recall"], (1.0 * 1 + (2.0 / 3.0) * 3 + 1.0 * 2) / 6.0)
        assert _approx(result["f1"], ((2.0 / 3.0) * 1 + 0.8 * 3 + 1.0 * 2) / 6.0)

    def test_multiclass_auc_perfectly_separable(self):
        """OvR macro AUC on a cleanly separable set is 1.0."""
        result = compute_scalar_classification_metrics(self._OUTPUT, self._TARGET)
        assert _approx(result["roc_auc"], 1.0)

    def test_two_column_onehot_uses_multiclass_path(self):
        """A 2-column one-hot target uses argmax (not the single-logit path)."""
        output = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        result = compute_scalar_classification_metrics(output, target)
        assert result["average"] == "macro"  # not "binary"
        assert result["n_classes"] == 2
        for key in METRIC_KEYS:
            assert _approx(result[key], 1.0), key


@pytest.mark.unit
class TestGracefulDegradation:
    """Undefined-metric cases degrade to null + a machine-readable reason."""

    def test_single_class_binary(self):
        """All-positive target -> no contrast -> every scalar None + reason."""
        result = compute_scalar_classification_metrics(torch.tensor([[0.9], [0.8]]), torch.tensor([[1.0], [1.0]]))
        for key in METRIC_KEYS:
            assert result[key] is None, key
            assert result["undefined"][key] == "single_class"

    def test_single_class_multiclass(self):
        """One-hot target with a single distinct class -> null + reason."""
        output = torch.tensor([[3.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = compute_scalar_classification_metrics(output, target)
        for key in METRIC_KEYS:
            assert result[key] is None, key
            assert result["undefined"][key] == "single_class"

    def test_empty_batch(self):
        result = compute_scalar_classification_metrics(torch.empty(0, 1), torch.empty(0, 1))
        for key in METRIC_KEYS:
            assert result[key] is None, key
            assert result["undefined"][key] == "empty_batch"
        assert result["n_samples"] == 0

    def test_nan_output(self):
        output = torch.tensor([[float("nan")], [0.2]])
        target = torch.tensor([[1.0], [0.0]])
        result = compute_scalar_classification_metrics(output, target)
        for key in METRIC_KEYS:
            assert result[key] is None, key
            assert result["undefined"][key] == "invalid_output"

    def test_inf_output(self):
        output = torch.tensor([[float("inf")], [0.2]])
        target = torch.tensor([[1.0], [0.0]])
        result = compute_scalar_classification_metrics(output, target)
        assert result["roc_auc"] is None
        assert result["undefined"]["roc_auc"] == "invalid_output"

    def test_inf_in_target(self):
        """Inf in the *target* tensor must degrade the same as Inf in output.

        Production checks ``isfinite`` on both tensors; only Inf-in-output was
        pinned previously — a NaN/Inf label from a corrupted eval split must not
        emit a misleading numeric F1/AUC.
        """
        output = torch.tensor([[0.9], [0.2]])
        target = torch.tensor([[float("inf")], [0.0]])
        result = compute_scalar_classification_metrics(output, target)
        for key in METRIC_KEYS:
            assert result[key] is None, key
            assert result["undefined"][key] == "invalid_output"

    def test_nan_target(self):
        """NaN in the *target* (not just output) is invalid_output for every scalar."""
        output = torch.tensor([[0.9], [0.2]])
        target = torch.tensor([[float("nan")], [0.0]])
        result = compute_scalar_classification_metrics(output, target)
        for key in METRIC_KEYS:
            assert result[key] is None, key
            assert result["undefined"][key] == "invalid_output"

    def test_never_predicted_class_does_not_undefine_aggregates(self):
        """A class present in y_true but never predicted contributes 0 via zero_division=0.

        Macro aggregates stay defined; the undefined map must not treat this as
        single_class / invalid_output (the C7 contract for per-class gaps).
        """
        # y_pred = [0,1,0,1]; y_true = [0,1,2,1] — class 2 never predicted.
        output = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        target = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        result = compute_scalar_classification_metrics(output, target, average="macro")
        # per-class P = [0.5, 1.0, 0.0] -> macro 0.5
        # per-class R = [1.0, 1.0, 0.0] -> macro 2/3
        # per-class F = [2/3, 1.0, 0.0] -> macro (2/3+1)/3
        assert _approx(result["precision"], 0.5)
        assert _approx(result["recall"], 2.0 / 3.0)
        assert _approx(result["f1"], (2.0 / 3.0 + 1.0 + 0.0) / 3.0)
        assert result["undefined"] == {}


@pytest.mark.unit
class TestWeightedNeverTrueClass:
    """Weighted average when a column has zero support (never present in target)."""

    def test_weighted_ignores_zero_support_class(self):
        """A never-true class (support=0) must not poison the weighted aggregate.

        3-class one-hot where class 2 never appears in ``y_true``. Macro still
        averages over all 3 columns (zero_division=0 for that class); weighted
        must divide only by the positive support total.
        """
        # rows: class0, class1, class0, class1 — class2 never true
        output = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 1.0],
            ]
        )
        target = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        result = compute_scalar_classification_metrics(output, target, average="weighted")
        assert result["average"] == "weighted"
        # Perfect on the two present classes; class2 support=0 contributes nothing.
        assert _approx(result["precision"], 1.0)
        assert _approx(result["recall"], 1.0)
        assert _approx(result["f1"], 1.0)
        assert result["n_classes"] == 3


@pytest.mark.unit
class TestContract:
    """Input-validation and result-shape guarantees."""

    def test_invalid_average_raises(self):
        with pytest.raises(ValueError):
            compute_scalar_classification_metrics(torch.tensor([[0.9], [0.1]]), torch.tensor([[1.0], [0.0]]), average="micro")

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError):
            compute_scalar_classification_metrics([[0.9]], torch.tensor([[1.0]]))

    def test_result_shape_has_all_keys(self):
        result = compute_scalar_classification_metrics(torch.tensor([[0.9], [0.1]]), torch.tensor([[1.0], [0.0]]))
        for key in ("f1", "precision", "recall", "roc_auc", "average", "n_samples", "n_classes", "undefined"):
            assert key in result, key

    def test_input_tensors_not_mutated(self):
        output = torch.tensor([[0.9], [0.4], [0.2]])
        target = torch.tensor([[1.0], [1.0], [0.0]])
        out_clone, tgt_clone = output.clone(), target.clone()
        compute_scalar_classification_metrics(output, target)
        assert torch.equal(output, out_clone)
        assert torch.equal(target, tgt_clone)


@pytest.mark.unit
class TestClassificationAccuracy:
    """``compute_classification_accuracy``, the final evaluation's accuracy (juniper-ml APD-ML-002).

    Pinned to hand-counted cases, plus the guards its sibling metrics apply to the
    same tensors in the same ``eval_metrics.final`` block.
    """

    def test_binary_exact(self):
        """scores [.9,.4,.2,.6,.8] -> y_pred [1,0,0,1,1]; y_true [1,1,0,0,1]; rows 0, 2, 4 match."""
        output = torch.tensor([[0.9], [0.4], [0.2], [0.6], [0.8]])
        target = torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]])
        accuracy, reason = compute_classification_accuracy(output, target)
        assert reason is None
        assert accuracy == 3 / 5

    def test_binary_threshold_is_strictly_greater_on_both_sides(self):
        """An output or a target of exactly 0.5 decodes to class 0, as in the network's ``_accuracy``.

        y_pred = [0, 1, 1, 1, 0] (0.5 is not > 0.5); y_true = [0, 0, 0, 1, 0] (nor is
        a 0.5 target); rows 0, 3 and 4 match. The rows are chosen so every ``>=``
        variant moves the count: on the output side 2/5, on the target side 5/5,
        on both 4/5.
        """
        output = torch.tensor([[0.5], [0.51], [0.9], [0.51], [0.2]])
        target = torch.tensor([[0.0], [0.5], [0.5], [1.0], [0.0]])
        accuracy, reason = compute_classification_accuracy(output, target)
        assert reason is None
        assert accuracy == 3 / 5

    def test_multiclass_exact(self):
        """y_pred = [0,1,2,0,1,2], y_true = [0,1,2,1,1,2]: 5 of 6 rows match."""
        accuracy, reason = compute_classification_accuracy(TestMulticlassScalarMetrics._OUTPUT, TestMulticlassScalarMetrics._TARGET)
        assert reason is None
        assert accuracy == 5 / 6

    def test_two_column_head_is_argmax_decoded_not_thresholded(self):
        """Every output is below 0.5, yet argmax gets both rows right.

        A single-column threshold applied to the first column would mark row 0
        wrong (0.4 is not > 0.5 while its target 1.0 is) and score 0.5.
        """
        output = torch.tensor([[0.4, 0.1], [0.2, 0.3]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        accuracy, reason = compute_classification_accuracy(output, target)
        assert reason is None
        assert accuracy == 1.0

    def test_single_class_binary_target_is_still_scored(self):
        """Unlike F1 / precision / recall / ROC-AUC, accuracy needs no positive/negative contrast."""
        output = torch.tensor([[0.9], [0.2], [0.8], [0.7]])
        target = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
        accuracy, reason = compute_classification_accuracy(output, target)
        assert reason is None
        assert accuracy == 3 / 4
        assert compute_scalar_classification_metrics(output, target)["undefined"]["f1"] == "single_class", "the siblings degrade on this batch; accuracy does not"

    def test_single_class_multiclass_target_is_still_scored(self):
        output = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        accuracy, reason = compute_classification_accuracy(output, target)
        assert reason is None
        assert accuracy == 1 / 2

    @pytest.mark.parametrize("n_cols", [1, 3])
    def test_empty_batch(self, n_cols):
        """``_accuracy`` returns NaN here; the final block gets None plus a reason, as the siblings do."""
        accuracy, reason = compute_classification_accuracy(torch.empty(0, n_cols), torch.empty(0, n_cols))
        assert accuracy is None
        assert reason == "empty_batch"

    def test_nan_output(self):
        accuracy, reason = compute_classification_accuracy(torch.tensor([[float("nan")], [0.2]]), torch.tensor([[1.0], [0.0]]))
        assert accuracy is None
        assert reason == "invalid_output"

    def test_inf_in_target(self):
        accuracy, reason = compute_classification_accuracy(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), torch.tensor([[float("inf"), 0.0], [0.0, 1.0]]))
        assert accuracy is None
        assert reason == "invalid_output"

    def test_one_d_shapes_accepted(self):
        accuracy, reason = compute_classification_accuracy(torch.tensor([0.9, 0.4, 0.2, 0.6, 0.8]), torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]))
        assert reason is None
        assert accuracy == 3 / 5

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError):
            compute_classification_accuracy([[0.9]], torch.tensor([[1.0]]))

    def test_input_tensors_not_mutated(self):
        output = torch.tensor([[0.9], [0.4], [0.2]])
        target = torch.tensor([[1.0], [1.0], [0.0]])
        out_clone, tgt_clone = output.clone(), target.clone()
        compute_classification_accuracy(output, target)
        assert torch.equal(output, out_clone)
        assert torch.equal(target, tgt_clone)

    def test_live_metric_set_is_unchanged(self):
        """Accuracy is final-evaluation only: the live per-step block keeps exactly its four scalars."""
        assert METRIC_KEYS == ("f1", "precision", "recall", "roc_auc")
        result = compute_scalar_classification_metrics(torch.tensor([[0.9], [0.1]]), torch.tensor([[1.0], [0.0]]))
        assert "accuracy" not in result


@pytest.mark.unit
class TestAccuracyMatchesTrainingHistoryDefinition:
    """Differential pin against ``CascadeCorrelationNetwork._accuracy``.

    That method produces the history's ``train_accuracy`` / ``value_accuracy``, which
    ``/v1/metrics`` serves as ``train_accuracy`` / ``val_accuracy``. The final accuracy
    must be the same quantity, or the three numbers stop being comparable. The inputs
    are quantized so exact-0.5 values (binary) and argmax ties (multi-class) occur.
    """

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("n_cols", [1, 2, 3])
    def test_agrees_with_the_network_accuracy(self, simple_network, seed, n_cols):
        generator = torch.Generator().manual_seed(seed)
        n_rows = 37
        if n_cols == 1:
            output = torch.round(torch.rand(n_rows, 1, generator=generator) * 10) / 10
            target = torch.round(torch.rand(n_rows, 1, generator=generator) * 2) / 2
        else:
            output = torch.round(torch.rand(n_rows, n_cols, generator=generator) * 3) / 3
            labels = torch.randint(0, n_cols, (n_rows,), generator=generator)
            target = torch.nn.functional.one_hot(labels, n_cols).to(torch.float32)
        expected = simple_network._accuracy(y=target, output=output)
        accuracy, reason = compute_classification_accuracy(output, target)
        assert reason is None
        assert accuracy == expected
