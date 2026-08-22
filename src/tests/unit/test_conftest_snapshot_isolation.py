#!/usr/bin/env python
"""The test suite must not write snapshots into the shared archive.

``train_output_layer`` calls ``create_snapshot()`` unconditionally, so any test that
trains an output layer writes a real ``.h5``. With no override that lands in
``<repo>/cascor-snapshots`` -- the project asset store holding ~27.9k research
artifacts. Measured 2026-08-21 before the fix: running ``tests/unit/test_p1_fixes.py``
alone added one file there.

Test exhaust accumulating in the asset store is precisely what makes "which of these
snapshots matter?" hard to answer, which is the question D-C provenance and the §6.2
index exist to fix. Cheaper not to create the problem.

The redirect lives in ``conftest.py`` and must run BEFORE the first cascor import,
because the model tier binds ``JUNIPER_CASCOR_SNAPSHOTS_DIR`` at IMPORT time. That
ordering is the fragile part, so it is pinned here directly.
"""

import os
from pathlib import Path

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_ARCHIVE = REPO_ROOT / "cascor-snapshots"


class TestSnapshotRedirect:
    def test_override_is_set_for_the_session(self):
        value = os.environ.get("JUNIPER_CASCOR_SNAPSHOTS_DIR", "").strip()
        assert value, "conftest must redirect JUNIPER_CASCOR_SNAPSHOTS_DIR for the test session"

    def test_override_is_not_the_shared_archive(self):
        target = Path(os.environ["JUNIPER_CASCOR_SNAPSHOTS_DIR"]).resolve()
        assert target != SHARED_ARCHIVE.resolve(), "the test session must not write into the shared snapshot archive"

    def test_import_time_binding_took_effect(self):
        """The redirect is worthless if it lands after the constant is bound.

        ``constants_hdf5`` resolves the env var at import time, so a redirect set in
        ``pytest_configure`` would leave this pointing at the shared root while the
        env var says otherwise -- passing the two checks above and still leaking.
        """
        from cascor_constants.constants_hdf5.constants_hdf5 import _HDF5_PROJECT_SNAPSHOTS_DIR

        assert Path(_HDF5_PROJECT_SNAPSHOTS_DIR).resolve() != SHARED_ARCHIVE.resolve(), "the model tier still resolves to the shared archive — the redirect ran too late to bind"

    def test_training_writes_no_file_into_the_shared_archive(self):
        """The behavioural arm: actually train, and count.

        The three checks above are all about configuration; this one is about the
        outcome they exist to produce, and is the only one that would catch a future
        code path that snapshots somewhere other than the resolved constant.
        """
        before = len(list(SHARED_ARCHIVE.glob("*.h5"))) if SHARED_ARCHIVE.is_dir() else 0

        network = CascadeCorrelationNetwork(config=CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=11))
        network.train_output_layer(torch.randn(8, 2), torch.randn(8, 2), epochs=2)

        after = len(list(SHARED_ARCHIVE.glob("*.h5"))) if SHARED_ARCHIVE.is_dir() else 0
        assert after == before, f"training added {after - before} snapshot(s) to the shared archive at {SHARED_ARCHIVE}"


class TestConftestOrdering:
    """Anti-resurrection for the ordering constraint the fix depends on."""

    def test_redirect_precedes_the_first_cascor_import(self):
        source = (Path(__file__).resolve().parents[1] / "conftest.py").read_text()
        redirect_at = source.find("JUNIPER_CASCOR_SNAPSHOTS_DIR")
        first_cascor_import = source.find("from cascade_correlation")

        assert redirect_at != -1, "conftest no longer redirects JUNIPER_CASCOR_SNAPSHOTS_DIR"
        assert first_cascor_import != -1, "expected a cascade_correlation import in conftest"
        assert redirect_at < first_cascor_import, "the snapshot redirect must come BEFORE the first cascor import — the model tier binds the env var at import time"
