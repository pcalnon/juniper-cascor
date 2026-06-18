"""Golden serialization round-trip regression (OUT-12 / WS-6 pre-refactor baseline).

After a fixed-seed two-spiral train, a trained network's predictions must survive
an HDF5 ``save_network`` -> ``load_network`` round-trip bit-for-bit (within
``atol=1e-6``). This adds the **trained** case to the existing untrained
round-trip coverage; it is the property the WS-6 gate actually targets.

A second assertion pins the trained predictions to a checked-in golden array
(tolerance) so a refactor that silently changes ``predict`` output is caught.

Out of scope (intentionally): **resume-training determinism** — continuing to
train after a save/load is NOT a guaranteed cascor property and remains an
``xfail`` in src/tests/.../test_comprehensive_serialization.py. The golden gate
targets predict-after-load only (build plan §4.2 / §8).

Run / re-capture: see test_golden_trajectory.py header (same flags;
``GOLDEN_CAPTURE=1`` to re-capture).
"""

import golden_support as gs
import pytest
import torch

from snapshots.snapshot_serializer import CascadeHDF5Serializer


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.golden
def test_golden_serialization_roundtrip(tmp_path):
    """Trained predictions survive an HDF5 save/load round-trip and match golden."""
    x, y = gs.load_two_spiral()
    net = gs.build_and_train_golden_network(x, y)

    pred_before = net.predict(x)
    assert isinstance(pred_before, torch.Tensor)
    assert tuple(pred_before.shape) == (x.shape[0], net.output_size)

    # Round-trip through the HDF5 serializer.
    h5_path = tmp_path / "golden_trained_net.h5"
    serializer = CascadeHDF5Serializer()
    saved = serializer.save_network(net, str(h5_path))
    assert saved is True, "save_network returned a falsy status"
    assert h5_path.exists(), "serializer did not write the HDF5 file"

    loaded = serializer.load_network(str(h5_path))
    assert loaded is not None, "load_network returned None"

    pred_after = loaded.predict(x)

    # Core property: predict-after-load is bit-equivalent within a tight atol.
    assert torch.allclose(pred_before, pred_after, atol=gs.PREDICT_ROUNDTRIP_ATOL), "predict drift across save/load: max|Δ| = " f"{(pred_before - pred_after).abs().max().item():.3e} > atol={gs.PREDICT_ROUNDTRIP_ATOL}"

    # Drift guard: the trained predictions themselves match the checked-in
    # golden (tolerance), catching refactors that change predict output.
    gs.assert_or_capture(
        "golden_predict_seed42.json",
        {"predict": pred_before},
        rtol=gs.RTOL,
        atol=gs.ATOL,
    )
