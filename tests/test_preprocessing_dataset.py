import json

import pytest

from radlab_data.preprocessing.dataset import (
    SequenceLabellingDataset,
    TextClassificationDataset,
    json_loader,
    jsonl_loader,
)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


def _dataset_from_records(tmp_path, records, **kwargs):
    return TextClassificationDataset(
        dataset_path=_write_jsonl(tmp_path / "dane.jsonl", records), **kwargs
    )


def test_jsonl_loader_wraps_string_labels(tmp_path):
    path = _write_jsonl(tmp_path / "dane.jsonl", [{"text": "a", "label": "klasa"}])
    assert jsonl_loader(path, "label")[0]["label"] == ["klasa"]


def test_jsonl_loader_without_label_field_keeps_values(tmp_path):
    path = _write_jsonl(tmp_path / "dane.jsonl", [{"text": "a", "label": "klasa"}])
    assert jsonl_loader(path)[0]["label"] == "klasa"


def test_json_loader_reads_whole_file(tmp_path):
    path = tmp_path / "dane.json"
    records = [{"text": "a", "label": ["x"]}]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    assert json_loader(str(path)) == records


def test_directory_loading_reads_only_dataset_extension(tmp_path):
    dataset_dir = tmp_path / "zestaw"
    dataset_dir.mkdir()
    _write_jsonl(dataset_dir / "czesc1.jsonl", [{"text": "a", "label": "x"}])
    _write_jsonl(dataset_dir / "czesc2.jsonl", [{"text": "b", "label": "y"}])
    (dataset_dir / "notatki.txt").write_text("to nie jest dane", encoding="utf-8")

    dataset = TextClassificationDataset(
        dataset_path=str(dataset_dir), dataset_extension=".jsonl"
    )
    assert len(dataset) == 2
    assert [item["text"] for item in dataset.raw_data] == ["a", "b"]


def test_directory_loading_requires_extension(tmp_path):
    dataset_dir = tmp_path / "zestaw"
    dataset_dir.mkdir()
    _write_jsonl(dataset_dir / "czesc1.jsonl", [{"text": "a", "label": "x"}])

    with pytest.raises(AssertionError, match="Specify dataset_extension"):
        TextClassificationDataset(dataset_path=str(dataset_dir))


def test_classification_dataset_maps_labels_to_ids(tmp_path):
    dataset = _dataset_from_records(
        tmp_path,
        [
            {"text": "a", "label": ["x", "y"]},
            {"text": "b", "label": ["x"]},
        ],
    )
    assert dataset.labels == ["x", "y"]
    assert dataset.raw_data[0]["metadata"]["label"] == [
        dataset.label2id["x"],
        dataset.label2id["y"],
    ]


def test_unlabelled_data_becomes_none(tmp_path):
    dataset = _dataset_from_records(tmp_path, [{"text": "a", "label": []}])
    assert dataset.raw_data[0]["metadata"]["label"] is None


def test_label_mapping_file_is_applied(tmp_path):
    mapping = tmp_path / "mapowanie.tsv"
    mapping.write_text("source_label\ttarget_label\nstare\tNowa\n", encoding="utf-8")

    dataset = _dataset_from_records(
        tmp_path,
        [{"text": "a", "label": ["stare"]}],
        class_labels_mapping_file=str(mapping),
    )
    assert dataset.labels == ["Nowa"]
    assert dataset.raw_data[0]["metadata"]["label"] == [dataset.label2id["Nowa"]]


def test_save_writes_data_and_metadata(tmp_path):
    from radlab_data.preprocessing.dataset import SaveConfiguration

    dataset = _dataset_from_records(
        tmp_path, [{"text": "a", "label": ["x"]}, {"text": "b", "label": ["y"]}]
    )
    out_dir = tmp_path / "wynik"
    saved_path = dataset.save(
        SaveConfiguration(path=str(out_dir), add_date_created_to_name=False)
    )

    data = json.loads((out_dir / "data.json").read_text(encoding="utf-8"))
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved_path == str(out_dir)
    assert len(data) == 2
    assert metadata["dataset_type"] == "TextClassificationDataset"
    assert metadata["labels"]["names"] == ["x", "y"]


def test_label_pair_cache_is_not_shared_between_instances():
    def _build(label_names):
        dataset = SequenceLabellingDataset.__new__(SequenceLabellingDataset)
        dataset._class_labels_mapping = {}
        dataset._label_pair_mapping_cache = {}
        dataset.label2id = {
            f"B-{name}": index for index, name in enumerate(label_names)
        }
        dataset.label2id.update(
            {f"I-{name}": index for index, name in enumerate(label_names)}
        )
        return dataset

    first = _build(["KOT", "DOM"])
    second = _build(["KOT", "DOM"])
    assert first._label_pair_mapping_cache is not second._label_pair_mapping_cache

    first._map_label_pair("KOT")
    assert first._label_pair_mapping_cache
    assert second._label_pair_mapping_cache == {}
