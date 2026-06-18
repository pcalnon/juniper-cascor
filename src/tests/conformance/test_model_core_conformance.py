"""OUT-13 — cascor ↔ juniper-model-core interface conformance (WS-6 gate, half 2).

Runs the real ``CascadeCorrelationNetwork`` (via the test-only adapter) through the
``juniper_model_core.conformance`` GrowableModel kit. Together with the OUT-12 golden suite
this forms the WS-6 trigger-gate: cascor may refactor onto the shared packages only if both
stay green. Plan:
``juniper-ml/notes/JUNIPER_CASCOR_MODEL_CORE_CONFORMANCE_WIRING_PLAN_2026-06-18.md``.

Run (serial, GIL env):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    CASCOR_NUM_PROCESSES=1 \
    python -m pytest -m conformance --conformance --slow --integration \
        src/tests/conformance
"""

import pytest
from conformance.cascor_model_core_adapter import CascorModelCoreAdapter, two_spiral_classification_dataset
from juniper_model_core.conformance import GrowableModelConformance


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.conformance
class TestCascorConformance(GrowableModelConformance):
    """CascadeCorrelationNetwork satisfies the model-core GrowableModel contract.

    Inherits ~13 contract assertions from ``GrowableModelConformance``; supplies the three
    factory hooks. The base class is not named ``Test*`` so pytest collects only this subclass.
    """

    def make_model(self) -> CascorModelCoreAdapter:
        return CascorModelCoreAdapter()

    def make_dataset(self):
        return two_spiral_classification_dataset()

    def make_serializer(self):
        # D-C4: skip the kit's serialization check — it asserts np.array_equal (bit-exact),
        # but cascor's HDF5 round-trip is allclose(atol=1e-6)-stable, not bit-exact. OUT-12's
        # test_golden_serialization_roundtrip already covers predict-after-load at tolerance.
        return None
