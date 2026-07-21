"""Torch-native scalar classification metrics (C7 phase 1 / U-4).

Computes **F1, precision, recall, and ROC-AUC** from a network's raw output
tensor and its (one-hot / single-column) target tensor WITHOUT adding a
scikit-learn dependency. cascor already depends on ``torch`` (>=2.10) and on
nothing in the sklearn stack, so these are implemented from first principles —
consistent with the project's "algorithms from first principles" research
philosophy — and kept dependency-free.

The binary-vs-multiclass decode mirrors the network's own ``_accuracy``
(``cascade_correlation.py``): a single output column is thresholded at 0.5; two
or more columns are argmax-decoded against a one-hot target. This keeps the new
scalars consistent with the accuracy already reported on the same rows.

Averaging (multi-class) — **macro** by default. Every class contributes equally
to the reported precision/recall/F1 regardless of its support. Macro was chosen
deliberately over ``weighted`` because the near-term research goal driving U-4
is multi-dataset continual training (juniper-ml
``notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md``
§4-U U-4 / §12 Q2), where class balance varies run-to-run and a support-weighted
average would mask minority-class collapse. ``weighted`` is implemented and
selectable via ``average=`` for callers that want it. Binary problems report the
positive-class scores directly (``average`` is reported as ``"binary"``).

ROC-AUC is rank-based — the Mann-Whitney U identity with average ranks for ties
— so it is exact and matches ``sklearn.metrics.roc_auc_score`` without the
dependency. Binary AUC uses the raw single-column score; multi-class AUC is
one-vs-rest over softmax probabilities, macro-averaged across the classes that
are defined (both present) on the batch.

Graceful degradation. Every metric is returned as ``None`` with a
machine-readable reason (in the ``undefined`` map) when it is undefined for the
batch, rather than raising or emitting a misleading number:

- ``empty_batch``    — zero samples.
- ``single_class``   — the target contains a single distinct class, so there is
  no positive/negative contrast (F1 / precision / recall / AUC are all
  ill-defined).
- ``invalid_output`` — the output or target contained NaN / Inf.

Within an otherwise-defined macro/weighted average, a per-class zero denominator
(e.g. a class that was never predicted) contributes ``0.0`` for that class (the
``zero_division=0`` convention) so the aggregate stays defined — this is a
per-class artifact, not a whole-metric degradation, and is NOT flagged in
``undefined``.

Phase-2 seam (deliberately NOT built here). The curve / explainability artifacts
in U-4 — confusion matrix, SHAP, permutation / feature importance, PDP, and the
calibration / ROC / PR / lift / gain curves — are a later unit (C7 phase 2 +
canopy N9). They belong beside this module (they need the same
``(output, target)`` inputs and the same binary-vs-multiclass decode) but are
out of scope for phase 1; keep this module scalar-only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

#: The scalar metric keys this module produces (flat, JSON-friendly).
METRIC_KEYS: Tuple[str, ...] = ("f1", "precision", "recall", "roc_auc")

#: Supported multi-class averaging strategies (binary problems ignore this and
#: report the positive class directly).
VALID_AVERAGES: Tuple[str, ...] = ("macro", "weighted")


def _null_result(reason: str, *, average: str, n_samples: int, n_classes: int) -> Dict[str, Any]:
    """Build a fully-degraded result: every scalar ``None`` with ``reason``."""
    return {
        "f1": None,
        "precision": None,
        "recall": None,
        "roc_auc": None,
        "average": average,
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "undefined": {key: reason for key in METRIC_KEYS},
    }


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1 for one class from its confusion counts.

    Zero denominators yield ``0.0`` (the ``zero_division=0`` convention) so a
    per-class gap never poisons the aggregate.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _aggregate(values: List[float], support: List[int], average: str) -> float:
    """Macro (equal-weight) or support-weighted mean of per-class values."""
    if not values:
        return 0.0
    if average == "weighted":
        total = sum(support)
        if total == 0:
            return 0.0
        return sum(value * count for value, count in zip(values, support)) / total
    return sum(values) / len(values)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """1-based ranks of ``values`` with tied entries sharing their mean rank.

    Vectorized (no per-element Python loop) so it stays cheap on large eval
    splits: sort once, group ties with ``unique_consecutive``, and assign each
    group the midpoint of the 1-based positions it spans.
    """
    n = values.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.float64)
    order = torch.argsort(values)
    sorted_vals = values[order].to(torch.float64)
    _uniq, inverse, counts = torch.unique_consecutive(sorted_vals, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, dim=0).to(torch.float64)  # 1-based end index per group
    starts = ends - counts.to(torch.float64) + 1.0  # 1-based start index per group
    group_avg = (starts + ends) / 2.0
    ranks_sorted = group_avg[inverse]
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = ranks_sorted
    return ranks


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    """Rank-based ROC-AUC for a binary problem (Mann-Whitney U identity).

    ``scores`` are float ranking scores; ``labels`` are ``{0, 1}``. Returns
    ``None`` when either class is absent (AUC is undefined without contrast).
    Handles ties exactly via average ranks, so this equals
    ``sklearn.metrics.roc_auc_score``.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores.to(torch.float64))
    sum_ranks_pos = float(ranks[labels == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_metrics(output: torch.Tensor, target: torch.Tensor, *, average: str, n_samples: int) -> Dict[str, Any]:
    """Positive-class precision/recall/F1 (threshold 0.5) + raw-score ROC-AUC."""
    scores = output.reshape(-1).to(torch.float64)
    y_true = (target.reshape(-1) > 0.5).to(torch.int64)
    y_pred = (scores > 0.5).to(torch.int64)

    if int(torch.unique(y_true).numel()) < 2:
        return _null_result("single_class", average=average, n_samples=n_samples, n_classes=2)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision, recall, f1 = _prf(tp, fp, fn)
    roc_auc = _binary_auc(scores, y_true)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        # Binary problems report the positive class, not a macro/weighted mean.
        "average": "binary",
        "n_samples": int(n_samples),
        "n_classes": 2,
        "undefined": {} if roc_auc is not None else {"roc_auc": "single_class"},
    }


def _multiclass_auc(output: torch.Tensor, y_true: torch.Tensor, n_classes: int) -> Tuple[Optional[float], Optional[str]]:
    """One-vs-rest macro ROC-AUC over softmax probabilities.

    Averaged across the classes that are defined (both present) on the batch.
    Returns ``(None, "single_class")`` when no class is defined.
    """
    probs = torch.softmax(output.to(torch.float64), dim=1)
    aucs: List[float] = []
    for cls in range(n_classes):
        labels_c = (y_true == cls).to(torch.int64)
        auc = _binary_auc(probs[:, cls], labels_c)
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return None, "single_class"
    return float(sum(aucs) / len(aucs)), None


def _multiclass_metrics(output: torch.Tensor, target: torch.Tensor, *, average: str, n_samples: int, n_classes: int) -> Dict[str, Any]:
    """Argmax-decoded per-class confusion counts, aggregated per ``average``."""
    y_pred = torch.argmax(output, dim=1)
    y_true = torch.argmax(target, dim=1)

    if int(torch.unique(y_true).numel()) < 2:
        return _null_result("single_class", average=average, n_samples=n_samples, n_classes=n_classes)

    per_class_p: List[float] = []
    per_class_r: List[float] = []
    per_class_f: List[float] = []
    support: List[int] = []
    for cls in range(n_classes):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision, recall, f1 = _prf(tp, fp, fn)
        per_class_p.append(precision)
        per_class_r.append(recall)
        per_class_f.append(f1)
        support.append(int((y_true == cls).sum()))

    roc_auc, auc_reason = _multiclass_auc(output, y_true, n_classes)
    undefined: Dict[str, str] = {}
    if roc_auc is None and auc_reason is not None:
        undefined["roc_auc"] = auc_reason

    return {
        "f1": _aggregate(per_class_f, support, average),
        "precision": _aggregate(per_class_p, support, average),
        "recall": _aggregate(per_class_r, support, average),
        "roc_auc": roc_auc,
        "average": average,
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "undefined": undefined,
    }


def compute_scalar_classification_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    average: str = "macro",
) -> Dict[str, Any]:
    """Compute F1 / precision / recall / ROC-AUC for one evaluation batch.

    Args:
        output: Raw network output, shape ``(N, C)`` (or ``(N,)`` / ``(N, 1)``
            for a single-logit binary head). Not assumed to be normalized —
            ROC-AUC is rank-based and the 0.5 threshold mirrors ``_accuracy``.
        target: Target tensor, one-hot ``(N, C)`` for multi-class or
            ``(N,)`` / ``(N, 1)`` for binary.
        average: ``"macro"`` (default) or ``"weighted"`` for the multi-class
            aggregation. Ignored for binary problems (positive class reported).

    Returns:
        A dict with flat scalar fields ``f1`` / ``precision`` / ``recall`` /
        ``roc_auc`` (each a ``float`` or ``None``), the ``average`` actually
        applied, ``n_samples``, ``n_classes``, and an ``undefined`` map of
        ``metric -> reason`` for any ``None`` scalar. Never raises for a
        degenerate batch — it degrades to ``None`` with a reason instead.

    Raises:
        TypeError: if ``output`` / ``target`` are not tensors.
        ValueError: if ``average`` is not one of :data:`VALID_AVERAGES`.
    """
    if average not in VALID_AVERAGES:
        raise ValueError(f"average must be one of {VALID_AVERAGES}, got {average!r}")
    if not isinstance(output, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("output and target must be torch.Tensor")

    output = output.detach()
    target = target.detach()
    if output.ndim == 1:
        output = output.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    n_samples = int(output.shape[0])
    n_cols = int(output.shape[1])
    is_binary = n_cols == 1
    n_classes = 2 if is_binary else n_cols

    if n_samples == 0:
        return _null_result("empty_batch", average=average, n_samples=0, n_classes=n_classes)
    if not (bool(torch.isfinite(output).all()) and bool(torch.isfinite(target).all())):
        return _null_result("invalid_output", average=average, n_samples=n_samples, n_classes=n_classes)

    if is_binary:
        return _binary_metrics(output, target, average=average, n_samples=n_samples)
    return _multiclass_metrics(output, target, average=average, n_samples=n_samples, n_classes=n_classes)
