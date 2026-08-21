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
        """Keys with no CLI counterpart anywhere are still reported, never silently dropped.

        L-4 (2026-08-21) narrowed this set considerably. ``max_iterations`` and
        ``early_stopping`` used to be the examples here and are now MAPPED, so the
        remaining genuinely service-tier keys stand in: the multi-candidate machinery
        and the snapshot-lifecycle knobs have no SpiralProblem or
        CascadeCorrelationConfig counterpart to route to.
        """
        from main import _resolve_cli_overrides

        overrides, unmapped = _resolve_cli_overrides(
            {"algorithm": "modern"},
            {"multi_candidate": True, "auto_snap_best": True, "candidate_pool_size": 8},
        )
        assert "dataset.params.algorithm" in unmapped
        assert "training.params.multi_candidate" in unmapped
        assert "training.params.auto_snap_best" in unmapped
        assert "multi_candidate" not in overrides
        # candidate_pool_size IS mappable since the W-11 pool amendment: SpiralProblem
        # takes _SpiralProblem__candidate_pool_size (spiral_problem.py:129) — P1.2
        # re-run profiling showed the constants pool (156 candidates over 2 rounds)
        # dominating smoke-scale wall time.
        assert overrides["candidate_pool_size"] == 8
        assert "training.params.candidate_pool_size" not in unmapped

    def test_epochs_max_stays_unmapped_deliberately(self):
        """``epochs_max`` must NOT be mapped, even though both sides accept it.

        TrainingParams documents it as DEPRECATED (C2b/Q1): "submitted values are
        accepted but reported skipped(not-updatable), never applied". Wiring it on the
        CLI would make the direct CLI honour a knob the SERVICE deliberately ignores --
        manufacturing a divergence rather than closing one, which is the opposite of
        what L-4 is for.
        """
        from main import _W11_TRAINING_KEY_MAP, _resolve_cli_overrides

        assert "epochs_max" not in _W11_TRAINING_KEY_MAP
        _, unmapped = _resolve_cli_overrides({}, {"epochs_max": 999})
        assert "training.params.epochs_max" in unmapped

    def test_the_nine_l4_keys_are_mapped(self):
        """L-4: every key the CLI used to drop with a warning now resolves to a knob.

        Before this, a direct-CLI run was not configured the way its YAML read. The two
        that mattered most were already plumbed end-to-end and missing only a map entry:
        spiral-baseline.yaml asks for candidate_learning_rate 0.05 and
        convergence_threshold 1.0e-5 while the CLI ran the constants 0.1 and 0.001 --
        2x and 100x looser -- silently confounding any CLI-vs-service comparison built
        on that config.
        """
        from main import _resolve_cli_overrides

        overrides, unmapped = _resolve_cli_overrides(
            {},
            {
                "candidate_learning_rate": 0.05,
                "convergence_threshold": 1.0e-5,
                "candidate_convergence_threshold": 2.0e-5,
                "candidate_patience": 100,
                "early_stopping": False,
                "max_iterations": 12,
                "init_output_weights": "zero",
                "optimizer_type": "SGD",
                "activation_function_name": "tanh",
            },
        )
        assert unmapped == [], f"L-4 keys must all map, still unmapped: {unmapped}"
        assert overrides["candidate_learning_rate"] == 0.05
        assert overrides["convergence_threshold"] == 1.0e-5
        assert overrides["candidate_convergence_threshold"] == 2.0e-5
        assert overrides["candidate_patience"] == 100
        assert overrides["early_stopping"] is False
        assert overrides["max_iterations"] == 12
        assert overrides["init_output_weights"] == "zero"
        assert overrides["optimizer_type"] == "SGD"
        # Renamed on the way through: TrainingParams/config say activation_function_name,
        # SpiralProblem says activation_function.
        assert overrides["activation_function"] == "tanh"

    def test_spiral_baseline_resolves_with_nothing_dropped(self):
        """The canonical shipped config must map completely -- it is the regression.

        spiral-baseline.yaml sets 12 training keys; five of them were dropped before
        L-4. If a future key is added to that config without a map entry, this fails.
        """
        from pathlib import Path

        import yaml

        from main import _resolve_cli_overrides

        config_path = Path(__file__).resolve().parents[3] / "conf" / "experiments" / "spiral-baseline.yaml"
        if not config_path.is_file():  # pragma: no cover - only in a partial checkout
            pytest.skip(f"{config_path} not present in this checkout")
        params = yaml.safe_load(config_path.read_text())["training"]["params"]

        overrides, unmapped = _resolve_cli_overrides({}, params)
        assert unmapped == [], f"spiral-baseline keys dropped by the direct CLI: {unmapped}"
        assert len(overrides) == len(params)
        # The two that were silently diverging, now carrying the config's own values.
        assert overrides["candidate_learning_rate"] == params["candidate_learning_rate"]
        assert overrides["convergence_threshold"] == params["convergence_threshold"]

    def test_spiral_problem_accepts_every_mapped_knob(self):
        """A map entry is only real if SpiralProblem actually takes that keyword.

        The map routes YAML keys to ``_SpiralProblem__<knob>`` kwargs. An entry naming a
        knob the constructor does not accept would raise TypeError at run time, hours
        into a campaign, rather than here.
        """
        import inspect

        from main import _W11_DATASET_KEY_MAP, _W11_TRAINING_KEY_MAP
        from spiral_problem.spiral_problem import SpiralProblem

        accepted = {name.replace("_SpiralProblem__", "") for name in inspect.signature(SpiralProblem.__init__).parameters}
        for knob in sorted(set(_W11_TRAINING_KEY_MAP.values()) | set(_W11_DATASET_KEY_MAP.values())):
            assert knob in accepted, f"_W11 map routes to '{knob}', which SpiralProblem does not accept"

    def test_empty_blocks_are_inert(self):
        from main import _resolve_cli_overrides

        assert _resolve_cli_overrides({}, {}) == ({}, [])
