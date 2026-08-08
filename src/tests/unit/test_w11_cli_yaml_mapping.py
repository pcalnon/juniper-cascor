"""Tests for the W-11 direct-CLI experiment-YAML mapping (CLI experimentation plan SS11 / Wave 3.6).

``main.py``'s thin adapter maps the experiment YAML's ``dataset.params`` /
``training.params`` onto the direct CLI's overridable knobs with the
``cascor_constants`` fallback tier, and reports (never silently drops) keys that
have no direct-CLI counterpart -- the W-1 doctrine applied to the CLI path.
"""

import pytest

pytestmark = pytest.mark.unit


class TestLoadExperimentBlocks:
    def test_unset_env_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_CONFIG_FILE", raising=False)
        from main import _load_experiment_blocks

        assert _load_experiment_blocks() == ({}, {})

    def test_blocks_extracted(self, tmp_path, monkeypatch):
        path = tmp_path / "experiment.yaml"
        path.write_text(
            "schema_version: 1\n" "experiment: {name: t, seed: 7}\n" "dataset:\n  generator: spiral\n  params: {n_spirals: 3, seed: 7}\n" "training:\n  params: {max_hidden_units: 2}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("JUNIPER_CASCOR_CONFIG_FILE", str(path))
        from main import _load_experiment_blocks

        dataset_params, training_params = _load_experiment_blocks()
        assert dataset_params == {"n_spirals": 3, "seed": 7}
        assert training_params == {"max_hidden_units": 2}


class TestResolveCliOverrides:
    def test_dataset_and_training_keys_mapped(self):
        from main import _resolve_cli_overrides

        overrides, unmapped = _resolve_cli_overrides(
            {"n_points_per_spiral": 200, "n_spirals": 2, "noise": 0.05, "seed": 7, "train_ratio": 0.8},
            {"max_hidden_units": 2, "learning_rate": 0.05, "patience": 50},
        )
        assert overrides["n_points"] == 200
        assert overrides["random_seed"] == 7
        assert overrides["train_ratio"] == 0.8
        assert overrides["max_hidden_units"] == 2
        assert overrides["learning_rate"] == 0.05
        assert overrides["patience"] == 50
        assert unmapped == []

    def test_max_epochs_aliases_to_output_epochs(self):
        from main import _resolve_cli_overrides

        overrides, _ = _resolve_cli_overrides({}, {"max_epochs": 50})
        assert overrides["output_epochs"] == 50

    def test_explicit_output_epochs_beats_the_alias(self):
        from main import _resolve_cli_overrides

        overrides, _ = _resolve_cli_overrides({}, {"max_epochs": 50, "output_epochs": 75})
        assert overrides["output_epochs"] == 75

    def test_service_tier_only_keys_reported_not_dropped(self):
        from main import _resolve_cli_overrides

        overrides, unmapped = _resolve_cli_overrides(
            {"algorithm": "modern"},
            {"max_iterations": 12, "candidate_pool_size": 8, "early_stopping": True},
        )
        assert "dataset.params.algorithm" in unmapped
        assert "training.params.max_iterations" in unmapped
        assert "training.params.early_stopping" in unmapped
        assert "max_iterations" not in overrides
        # candidate_pool_size IS mappable since the W-11 pool amendment: SpiralProblem
        # takes _SpiralProblem__candidate_pool_size (spiral_problem.py:129) — P1.2
        # re-run profiling showed the constants pool (156 candidates over 2 rounds)
        # dominating smoke-scale wall time.
        assert overrides["candidate_pool_size"] == 8
        assert "training.params.candidate_pool_size" not in unmapped

    def test_empty_blocks_are_inert(self):
        from main import _resolve_cli_overrides

        assert _resolve_cli_overrides({}, {}) == ({}, [])
