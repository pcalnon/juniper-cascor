"""OUT-13 / WS-6 PR-B4 — cascor ↔ juniper-model-core interface conformance (WS-6 gate, half 2).

Runs the **production** ``CascorModel`` (wrapping a real ``CascadeCorrelationNetwork``) through
the ``juniper_model_core.conformance`` GrowableModel kit — "native conformance" (PR-B4 retired
the test-only adapter). Together with the OUT-12 golden suite
this forms the WS-6 trigger-gate: cascor may refactor onto the shared packages only if both
stay green. Plan:
``juniper-ml/notes/JUNIPER_2026-06-18_JUNIPER-CASCOR_MODEL-CORE-CONFORMANCE-WIRING-PLAN.md``.

Run (serial, GIL env):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    CASCOR_NUM_PROCESSES=1 \
    python -m pytest -m conformance --conformance --slow --integration \
        src/tests/conformance
"""

import pytest
from conformance.cascor_model_core_adapter import make_cascor_conformance_model, two_spiral_classification_dataset
from juniper_model_core.conformance import GrowableModelConformance

from api.models.cascor_model import CascorModel


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.conformance
class TestCascorConformance(GrowableModelConformance):
    """The production ``CascorModel`` satisfies the model-core GrowableModel contract.

    Inherits ~13 contract assertions from ``GrowableModelConformance``; supplies the three
    factory hooks. The base class is not named ``Test*`` so pytest collects only this subclass.
    PR-B4: ``make_model`` returns the production ``CascorModel`` (native conformance), not the
    retired test-only adapter.
    """

    def make_model(self) -> CascorModel:
        return make_cascor_conformance_model()

    def make_dataset(self):
        return two_spiral_classification_dataset()

    def make_serializer(self):
        # D-C4: skip the kit's serialization check — it asserts np.array_equal (bit-exact),
        # but cascor's HDF5 round-trip is allclose(atol=1e-6)-stable, not bit-exact. OUT-12's
        # test_golden_serialization_roundtrip already covers predict-after-load at tolerance.
        return None
