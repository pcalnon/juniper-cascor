"""Regression tests for candidate-core utility helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_save_and_load_dataset_roundtrips_torch_checkpoint(tmp_path):
    from utils.utils import load_dataset, save_dataset

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dataset_path = tmp_path / "candidate-dataset.pt"

    save_dataset(x, y, dataset_path)
    loaded_x, loaded_y = load_dataset(dataset_path)

    assert loaded_x.device.type == "cpu"
    assert torch.equal(loaded_x, x)
    assert torch.equal(loaded_y, y)


def test_get_class_distribution_handles_one_hot_and_index_targets():
    from utils.utils import get_class_distribution

    one_hot = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    indices = torch.tensor([2, 2, 0, 1, 2])

    assert get_class_distribution(one_hot) == {0: 1, 1: 2, 2: 1}
    assert get_class_distribution(indices) == {0: 1, 1: 1, 2: 3}


def test_object_attributes_table_filters_private_attributes_without_columnar(monkeypatch):
    from utils import utils

    monkeypatch.setattr(utils, "HAS_COLUMNAR", False)
    monkeypatch.setattr(utils, "col", None)

    table = utils._object_attributes_to_table(
        obj_dict={"public": 1, "_private": 2},
        keys=["public", "_private"],
        private_attrs=False,
    )

    assert "Attribute: public, Attribute Value: 1" in table
    assert "_private" not in table
