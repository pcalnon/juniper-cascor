#!/usr/bin/env python
"""
Unit tests for candidate-seed derivation (juniper-cascor#532).

Candidate seeds used to be drawn from the process-global ``random`` stream:

    candidate_seeds = [random.randint(0, self.random_max_value) for _ in range(pool_size)]

so a candidate's seed was a function of how many times *anything* in the process had
drawn from that stream before this point -- not of the configured seed. The measured
consequence: on an identical cell with ``network_seed=42`` on both paths, the direct
CLI's round-0 seed list began at the service's FOURTH element and stayed three draws
offset for the whole run, because ``SpiralProblem`` (CLI-only) consumed three values
first. The two entry points trained *different candidates* on identical configuration.

Seeds now come from a network-owned ``random.Random(self.random_seed)``.

Tests focus on:
- round-k seeds are a function of (random_seed, k) alone
- intervening global random draws do not move them (THE regression guard)
- re-seeding the global stream does not move them
- the generator is carried in pickled state, so a resumed run continues the sequence
"""

import os
import pickle
import random
import sys

import pytest
import torch

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

POOL = 4
SEED = 42


def _make_network(random_seed=SEED):
    config = CascadeCorrelationConfig(
        input_size=2,
        output_size=2,
        candidate_pool_size=POOL,
        random_seed=random_seed,
    )
    return CascadeCorrelationNetwork(config=config)


def _release(network):
    """_generate_candidate_tasks allocates a SharedTrainingMemory block per call."""
    for blk in list(getattr(network, "_active_shm_blocks", [])):
        try:
            blk.close_and_unlink()
        except Exception:  # nosec B110 -- test cleanup must not mask the assertion
            pass
    if hasattr(network, "_active_shm_blocks"):
        network._active_shm_blocks.clear()


def _round_seeds(network, rounds=2):
    """Drive _generate_candidate_tasks `rounds` times, returning the seed list per round.

    Tuple layout is (index, input_size, activation_name, random_value_scale, uuid, SEED,
    random_max_value, sequence_max_value); the seed is element 5.
    """
    candidate_input = torch.zeros((6, 2), dtype=torch.float32)
    y = torch.zeros((6, 2), dtype=torch.float32)
    residual_error = torch.zeros((6, 2), dtype=torch.float32)

    out = []
    for _ in range(rounds):
        tasks = network._generate_candidate_tasks(candidate_input, y, residual_error)
        out.append([task[1][5] for task in tasks])
    return out


class TestCandidateSeedDerivation:
    """Candidate seeds must depend on the configured seed, and on nothing else."""

    def test_round_seeds_are_a_function_of_seed_and_round(self):
        """Two networks with the same configured seed produce the same seeds, round for round."""
        net_a = _make_network()
        net_b = _make_network()
        try:
            seeds_a = _round_seeds(net_a, rounds=3)
            seeds_b = _round_seeds(net_b, rounds=3)
        finally:
            _release(net_a)
            _release(net_b)

        assert seeds_a == seeds_b, f"same seed produced different candidate seeds:\n{seeds_a}\n{seeds_b}"
        # And successive rounds must not simply replay round 0, or every round would train
        # the identical pool.
        assert seeds_a[0] != seeds_a[1], f"round 1 replayed round 0's seeds: {seeds_a}"

    def test_seeds_unmoved_by_intervening_global_random_draws(self):
        """THE regression guard.

        Under the old derivation, anything that touched the global ``random`` stream between
        rounds -- a retry, a shuffle, a log line that samples -- silently re-seeded every
        candidate in every later round. Nothing failed; the numbers just changed.
        """
        net_quiet = _make_network()
        net_noisy = _make_network()
        try:
            quiet = _round_seeds(net_quiet, rounds=1)
            noisy_first = _round_seeds(net_noisy, rounds=1)

            # Exactly the perturbation the old code was sensitive to.
            for _ in range(37):
                random.random()  # nosec B311 -- deliberately perturbing the global stream

            noisy_second = _round_seeds(net_noisy, rounds=1)
            quiet_second = _round_seeds(net_quiet, rounds=1)
        finally:
            _release(net_quiet)
            _release(net_noisy)

        assert noisy_first == quiet, "round 0 already diverged"
        assert noisy_second == quiet_second, f"global random draws moved the candidate seeds: {noisy_second} != {quiet_second}"

    def test_seeds_unmoved_by_a_later_global_reseed(self):
        """The measured CLI-vs-service divergence, in miniature.

        Note this is NOT the same as seeding the global stream *before* construction: the
        network itself calls ``random.seed(random_seed)`` on the process-global stream while
        initialising (``_initialize_randomness`` -> ``_seed_random_generator``), which wipes
        any earlier global seeding. A test that seeded first would pass with the bug still
        present and guard nothing.

        What actually moved the seeds was a re-seed or draw AFTER construction, because the
        old derivation read that same global stream. That is exactly what happened on the
        direct CLI: ``SpiralProblem`` calls ``random.seed(random_seed)`` and then draws, so
        the CLI's round-0 seed list began at the service's FOURTH element.
        """
        net_quiet = _make_network()
        net_reseeded = _make_network()
        try:
            quiet = _round_seeds(net_quiet, rounds=1)

            # SpiralProblem's exact shape: re-seed the global stream, then consume from it.
            random.seed(SEED)
            for _ in range(3):
                random.random()  # nosec B311 -- deliberately perturbing the global stream

            reseeded = _round_seeds(net_reseeded, rounds=1)
        finally:
            _release(net_quiet)
            _release(net_reseeded)

        assert reseeded == quiet, f"a global re-seed after construction moved the candidate seeds:\n{reseeded}\n{quiet}"

    def test_different_network_seeds_give_different_candidate_seeds(self):
        """The configured seed is what the derivation actually depends on."""
        net_a = _make_network(random_seed=42)
        net_b = _make_network(random_seed=43)
        try:
            seeds_a = _round_seeds(net_a, rounds=1)
            seeds_b = _round_seeds(net_b, rounds=1)
        finally:
            _release(net_a)
            _release(net_b)

        assert seeds_a != seeds_b, f"different network seeds produced identical candidate seeds: {seeds_a}"

    def test_generator_is_carried_in_pickled_state(self):
        """A resumed network continues the sequence instead of replaying round 0.

        __getstate__ copies __dict__ and pops the non-picklable members; random.Random is
        picklable, so the generator's position survives. If it were dropped, the lazy
        re-init would hand a resumed run round 0's seeds a second time.
        """
        net = _make_network()
        try:
            first = _round_seeds(net, rounds=1)
            state = net.__getstate__()
            assert "_candidate_seed_rng" in state, "seed generator was dropped from pickled state"

            revived = pickle.loads(pickle.dumps(net._candidate_seed_rng))
            expected_next = [revived.randint(0, net.random_max_value) for _ in range(POOL)]

            second = _round_seeds(net, rounds=1)
        finally:
            _release(net)

        assert second[0] != first[0], "resumed sequence replayed round 0"
        assert second[0] == expected_next, f"generator position not preserved: {second[0]} != {expected_next}"
