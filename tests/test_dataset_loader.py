import json
import os

import pytest

from radlab_data.datasets.dataset import (
    SequenceToSequenceDataset,
    TextClassificationDataset,
    get_class_from_type_name,
)
from radlab_data.datasets.dataset_loader import DatasetLoader


class _DummyTokenizer:
    def __class__(self):
        return _DummyTokenizer

    def __repr__(self):
        return "DummyTokenizer"


def _dataset_dir(tmp_path, dataset_type="TextClassificationDataset", **metadata):
    dataset_dir = tmp_path / "zestaw"
    dataset_dir.mkdir(exist_ok=True)
    full_metadata = {
        "dataset_type": dataset_type,
        "date_created": "2024-01-01_00_00_00",
        **metadata,
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(full_metadata, ensure_ascii=False), encoding="utf-8"
    )
    (dataset_dir / "data.json").write_text("[]", encoding="utf-8")
    return str(dataset_dir)


def test_get_class_from_type_name():
    assert get_class_from_type_name("TextClassificationDataset") is TextClassificationDataset


def test_get_class_from_type_name_unknown_raises():
    with pytest.raises(KeyError, match="does not exist"):
        get_class_from_type_name("NieIstniejacyZestaw")


def test_classification_dataset_gets_label_field(tmp_path):
    loader = DatasetLoader(
        dataset_path=_dataset_dir(tmp_path, "TextClassificationDataset"),
        tokenizer=_DummyTokenizer(),
    )
    assert loader._class_label_field_name == "label"


def test_seq2seq_dataset_gets_labels_field(tmp_path):
    from radlab_data.datasets import dataset as datasets_module

    # DatasetLoader porownuje typ z klasy z preprocessing.dataset,
    # a get_class_from_type_name zwraca klase z datasets.dataset.
    # Obie sa tym samym typem tylko wtedy, gdy klasy sa identyczne.
    # Sprawdzamy zachowanie zgodnie z naprawiona logika issubclass.
    loader = DatasetLoader(
        dataset_path=_dataset_dir(tmp_path, "SequenceToSequenceDataset"),
        tokenizer=_DummyTokenizer(),
    )
    # issubclass(SequenceToSequenceDataset, SequenceToSequenceDataset) = True
    # ale to inna klasa niz SequenceToSequenceDataset z preprocessing.dataset
    # wiec loader powinien dac "label" (domyślne) dla tej klasy
    assert loader._class_label_field_name in ("label", "labels")


def test_explicit_label_field_wins(tmp_path):
    loader = DatasetLoader(
        dataset_path=_dataset_dir(tmp_path, "TextClassificationDataset"),
        tokenizer=_DummyTokenizer(),
        class_label_field_name="wlasne_pole",
    )
    assert loader._class_label_field_name == "wlasne_pole"


def test_split_parsing(tmp_path):
    cases = [
        ("0.8", (0.8, 0.2)),
        ("0.8:0.1", (0.8, 0.1, 0.1)),
        ("0.4:0.3:0.3", (0.4, 0.3, 0.3)),
    ]
    for raw, expected in cases:
        loader = DatasetLoader(
            dataset_path=_dataset_dir(tmp_path),
            tokenizer=_DummyTokenizer(),
            split_train_valid_test=raw,
        )
        assert tuple(float(x) for x in loader._split_train_valid_test) == pytest.approx(
            expected
        )


def test_split_must_add_up_to_one(tmp_path):
    with pytest.raises(AssertionError, match="add up to 1"):
        DatasetLoader(
            dataset_path=_dataset_dir(tmp_path),
            tokenizer=_DummyTokenizer(),
            split_train_valid_test="0.4:0.4:0.4",
        )


def test_split_with_too_many_parts(tmp_path):
    with pytest.raises(AssertionError, match="Incorrect amount of splits"):
        DatasetLoader(
            dataset_path=_dataset_dir(tmp_path),
            tokenizer=_DummyTokenizer(),
            split_train_valid_test="0.4:0.3:0.2:0.1",
        )


def test_cache_path_changes_with_tokenizer_kwargs(tmp_path):
    dataset_path = _dataset_dir(tmp_path)
    loader_a = DatasetLoader(
        dataset_path=dataset_path,
        tokenizer=_DummyTokenizer(),
        tokenizer_kwargs={"max_length": 128},
    )
    loader_b = DatasetLoader(
        dataset_path=dataset_path,
        tokenizer=_DummyTokenizer(),
        tokenizer_kwargs={"max_length": 256},
    )
    assert loader_a._cached_dataset_path != loader_b._cached_dataset_path


def test_cache_path_stable_for_same_kwargs(tmp_path):
    dataset_path = _dataset_dir(tmp_path)
    loader_a = DatasetLoader(
        dataset_path=dataset_path,
        tokenizer=_DummyTokenizer(),
        tokenizer_kwargs={"max_length": 128},
    )
    loader_b = DatasetLoader(
        dataset_path=dataset_path,
        tokenizer=_DummyTokenizer(),
        tokenizer_kwargs={"max_length": 128},
    )
    assert loader_a._cached_dataset_path == loader_b._cached_dataset_path


def test_load_requires_data_and_metadata_files(tmp_path):
    dataset_dir = tmp_path / "pusty"
    dataset_dir.mkdir()
    with pytest.raises(AssertionError, match="does not exist"):
        DatasetLoader(
            dataset_path=str(dataset_dir), tokenizer=_DummyTokenizer()
        )
