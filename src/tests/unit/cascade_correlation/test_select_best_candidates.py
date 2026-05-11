"""§1.5 C2 (Issue #1, PR-4b) — _select_best_candidates strategy paths.

Three strategies in scope:

  - "top": pick the N highest-correlation candidates (legacy behavior).
  - "random": pick N uniformly at random from the eligible pool.
  - "mixed": pick T highest-correlation + R random from the remainder
    (the §1.5 C2.1 invariant guarantees T+R==S when both nonzero).

Threshold filtering (``correlation_threshold``) applies to every
strategy — random/mixed picks never include below-threshold candidates.

PR-4a shipped the schema + invariant validator. PR-4b wires those
fields into selection here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from candidate_unit.candidate_unit import CandidateTrainingResult
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork


def _make_pool(correlations: list[float]) -> list[CandidateTrainingResult]:
    """Build a list of CandidateTrainingResult fakes with the given correlations."""
    pool = []
    for i, corr in enumerate(correlations):
        result = CandidateTrainingResult(
            candidate_id=i,
            correlation=corr,
            candidate=MagicMock(name=f"candidate-{i}"),
            success=True,
        )
        pool.append(result)
    return pool


def _make_network(**attrs):
    """Bare CascadeCorrelationNetwork — only the attrs the selector reads."""
    net = CascadeCorrelationNetwork.__new__(CascadeCorrelationNetwork)
    import logging

    net.logger = logging.getLogger("test_select")
    net.correlation_threshold = attrs.pop("correlation_threshold", 0.0)
    net.candidate_selection = attrs.pop("candidate_selection", "top")
    net.top_candidates = attrs.pop("top_candidates", 1)
    net.random_candidates = attrs.pop("random_candidates", 0)
    # random_seed=42 makes random/mixed picks deterministic across runs.
    net.random_seed = attrs.pop("random_seed", 42)
    return net


# ---------------------------------------------------------------------------
# top strategy (legacy behavior)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTopStrategy:
    def test_top_picks_highest_correlations_in_order(self):
        net = _make_network()
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3])
        selected = net._select_best_candidates(pool, num_candidates=3, strategy="top")
        # Sorted descending by abs(correlation): 0.9, 0.7, 0.5
        assert [r.correlation for r in selected] == [0.9, 0.7, 0.5]

    def test_top_handles_negative_correlations_via_abs(self):
        net = _make_network()
        pool = _make_pool([-0.95, 0.1, -0.4, 0.2])
        selected = net._select_best_candidates(pool, num_candidates=2, strategy="top")
        # |-0.95| > |0.2|; -0.95 wins; next is |-0.4| > |0.1|
        assert [r.correlation for r in selected] == [-0.95, -0.4]

    def test_top_strategy_default_when_unspecified(self):
        net = _make_network(candidate_selection="top")
        pool = _make_pool([0.1, 0.9, 0.5])
        # No explicit strategy kwarg — falls back to self.candidate_selection.
        selected = net._select_best_candidates(pool, num_candidates=2)
        assert [r.correlation for r in selected] == [0.9, 0.5]


# ---------------------------------------------------------------------------
# Threshold filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThresholdFilter:
    def test_threshold_excludes_below_min_correlation(self):
        net = _make_network(correlation_threshold=0.5)
        pool = _make_pool([0.1, 0.9, 0.3, 0.6, 0.49])
        # Only 0.9 and 0.6 are >= 0.5.
        selected = net._select_best_candidates(pool, num_candidates=5, strategy="top")
        assert sorted(r.correlation for r in selected) == [0.6, 0.9]

    def test_empty_when_nothing_above_threshold(self):
        net = _make_network(correlation_threshold=0.99)
        pool = _make_pool([0.1, 0.5, 0.9])
        selected = net._select_best_candidates(pool, num_candidates=3, strategy="top")
        assert selected == []


# ---------------------------------------------------------------------------
# random strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRandomStrategy:
    def test_random_returns_n_from_eligible_pool(self):
        net = _make_network(random_seed=42)
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3, 0.4])
        selected = net._select_best_candidates(pool, num_candidates=3, strategy="random")
        assert len(selected) == 3
        # All picks come from the original pool (no fabricated entries).
        assert {id(r) for r in selected} <= {id(r) for r in pool}

    def test_random_with_seed_is_deterministic(self):
        net1 = _make_network(random_seed=7)
        net2 = _make_network(random_seed=7)
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3, 0.4])
        s1 = net1._select_best_candidates(pool, num_candidates=3, strategy="random")
        s2 = net2._select_best_candidates(pool, num_candidates=3, strategy="random")
        assert [r.candidate_id for r in s1] == [r.candidate_id for r in s2]

    def test_random_does_not_always_pick_top_correlations(self):
        """Distinguishes random from top — the union of picks across many
        calls should contain at least one non-top candidate."""
        # Different seed each call to span the sample space.
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3, 0.2])
        all_picks: set[int] = set()
        for seed in range(20):
            net = _make_network(random_seed=seed)
            sel = net._select_best_candidates(pool, num_candidates=2, strategy="random")
            all_picks.update(r.candidate_id for r in sel)
        # Across 20 seeds × 2 picks we should see candidates that aren't the top-2.
        # Top-2 by correlation: id=1 (0.9), id=3 (0.7). Anything else proves randomness.
        non_top_seen = all_picks - {1, 3}
        assert non_top_seen, f"random strategy never picked a non-top candidate across 20 seeds: {all_picks}"

    def test_random_honors_threshold(self):
        net = _make_network(correlation_threshold=0.5, random_seed=1)
        pool = _make_pool([0.1, 0.9, 0.3, 0.6, 0.49, 0.7])
        selected = net._select_best_candidates(pool, num_candidates=5, strategy="random")
        for r in selected:
            assert abs(r.correlation) >= 0.5, f"random pick {r.correlation} below threshold"


# ---------------------------------------------------------------------------
# mixed strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMixedStrategy:
    def test_mixed_t2_r2_picks_top_two_plus_two_random(self):
        net = _make_network(random_seed=42)
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3, 0.4, 0.2])
        selected = net._select_best_candidates(pool, num_candidates=4, strategy="mixed", top_count=2, random_count=2)
        assert len(selected) == 4
        # First 2 are deterministic (top by correlation): 0.9 and 0.7.
        assert [r.correlation for r in selected[:2]] == [0.9, 0.7]
        # Last 2 are random from {0.5, 0.4, 0.3, 0.2, 0.1} (post-top remainder).
        random_corrs = [r.correlation for r in selected[2:]]
        assert all(c in {0.5, 0.4, 0.3, 0.2, 0.1} for c in random_corrs), random_corrs

    def test_mixed_uses_self_attrs_when_kwargs_omitted(self):
        net = _make_network(candidate_selection="mixed", top_candidates=3, random_candidates=1, random_seed=42)
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3, 0.4])
        selected = net._select_best_candidates(pool, num_candidates=4)
        assert len(selected) == 4
        # First 3 are top by correlation: 0.9, 0.7, 0.5.
        assert [r.correlation for r in selected[:3]] == [0.9, 0.7, 0.5]

    def test_mixed_with_t_zero_is_pure_random(self):
        net = _make_network(random_seed=42)
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3])
        selected = net._select_best_candidates(pool, num_candidates=3, strategy="mixed", top_count=0, random_count=3)
        assert len(selected) == 3

    def test_mixed_with_r_zero_is_pure_top(self):
        net = _make_network(random_seed=42)
        pool = _make_pool([0.1, 0.9, 0.5, 0.7, 0.3])
        selected = net._select_best_candidates(pool, num_candidates=3, strategy="mixed", top_count=3, random_count=0)
        assert [r.correlation for r in selected] == [0.9, 0.7, 0.5]


# ---------------------------------------------------------------------------
# Unknown strategy fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_strategy_falls_back_to_top():
    net = _make_network()
    pool = _make_pool([0.1, 0.9, 0.5])
    selected = net._select_best_candidates(pool, num_candidates=2, strategy="lottery")
    assert [r.correlation for r in selected] == [0.9, 0.5]


# ---------------------------------------------------------------------------
# _effective_candidate_count — wires the multi_candidate + selected_candidates
# PATCH-surface knobs into the grow_network branch selector.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEffectiveCandidateCount:
    def _net(self, **attrs):
        net = CascadeCorrelationNetwork.__new__(CascadeCorrelationNetwork)
        # Defaults match what __init__ would set.
        net.multi_candidate = attrs.get("multi_candidate", False)
        net.selected_candidates = attrs.get("selected_candidates", 1)
        if "candidates_per_layer" in attrs:
            net.candidates_per_layer = attrs["candidates_per_layer"]
        return net

    def test_default_is_one_for_legacy_single_candidate_path(self):
        assert self._net()._effective_candidate_count() == 1

    def test_multi_candidate_off_returns_one_regardless_of_S(self):
        net = self._net(multi_candidate=False, selected_candidates=4)
        assert net._effective_candidate_count() == 1

    def test_multi_candidate_on_returns_selected_candidates(self):
        net = self._net(multi_candidate=True, selected_candidates=4)
        assert net._effective_candidate_count() == 4

    def test_legacy_candidates_per_layer_wins_when_gt_one(self):
        net = self._net(multi_candidate=True, selected_candidates=4, candidates_per_layer=8)
        assert net._effective_candidate_count() == 8

    def test_legacy_candidates_per_layer_one_does_not_force_single(self):
        """Setting candidates_per_layer=1 explicitly shouldn't force single-
        candidate when multi_candidate is True with S>1."""
        net = self._net(multi_candidate=True, selected_candidates=3, candidates_per_layer=1)
        assert net._effective_candidate_count() == 3
