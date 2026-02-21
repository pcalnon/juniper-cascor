#!/usr/bin/env python
"""
Unit tests for CascadeCorrelationConfig pickling to improve code coverage.

Covers:
- __getstate__: removes log_config for serialization (lines 235-238)
- __setstate__: restores state and sets log_config to None (lines 242-244)
"""

import os
import pickle
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

pytestmark = pytest.mark.unit


class TestConfigPickling:
    """Tests for CascadeCorrelationConfig pickling support."""

    def test_config_pickle_roundtrip(self):
        """Config should be picklable and restorable."""
        config = CascadeCorrelationConfig(
            input_size=3,
            output_size=2,
            learning_rate=0.05,
            max_hidden_units=20,
        )

        data = pickle.dumps(config)
        restored = pickle.loads(data)

        assert isinstance(restored, CascadeCorrelationConfig)
        assert restored.input_size == 3
        assert restored.output_size == 2
        assert restored.optimizer_config.learning_rate == 0.05
        assert restored.max_hidden_units == 20

    def test_getstate_removes_log_config(self):
        """__getstate__ should remove log_config from serialized state."""
        config = CascadeCorrelationConfig(input_size=2, output_size=2)
        state = config.__getstate__()

        assert "log_config" not in state
        assert "input_size" in state

    def test_setstate_sets_log_config_to_none(self):
        """__setstate__ should set log_config to None."""
        config = CascadeCorrelationConfig(input_size=2, output_size=2)
        state = config.__getstate__()

        new_config = CascadeCorrelationConfig.__new__(CascadeCorrelationConfig)
        new_config.__setstate__(state)

        assert new_config.log_config is None
        assert new_config.input_size == 2

    def test_pickle_preserves_uuid(self):
        """Pickling should preserve the UUID."""
        config = CascadeCorrelationConfig(input_size=2, output_size=2)
        original_uuid = config.uuid

        data = pickle.dumps(config)
        restored = pickle.loads(data)

        assert restored.uuid == original_uuid

    def test_pickle_preserves_all_config_fields(self):
        """Pickling should preserve all configuration fields."""
        config = CascadeCorrelationConfig(
            input_size=4,
            output_size=3,
            learning_rate=0.02,
            max_hidden_units=50,
            candidate_pool_size=8,
            candidate_epochs=100,
            output_epochs=200,
            patience=5,
        )

        data = pickle.dumps(config)
        restored = pickle.loads(data)

        assert restored.input_size == 4
        assert restored.output_size == 3
        assert restored.max_hidden_units == 50
        assert restored.candidate_pool_size == 8
