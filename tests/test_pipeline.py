import pytest

from radlab_data.preprocessing.dataset import TextClassificationDataset
from radlab_data.preprocessing.pipeline import Pipeline, PipelineElementBase
from radlab_data.preprocessing.pipeline_modules import (
    RemoveDuplicates,
    RemoveLongIOBAnnotation,
    RemoveUnlabelledData,
)


def _dataset(tmp_path, records):
    path = tmp_path / "dane.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return TextClassificationDataset(dataset_path=str(path))


import json


def test_pipeline_runs_elements_in_order(tmp_path):
    calls = []

    class Pierwszy(PipelineElementBase):
        def _process_data(self, dataset):
            calls.append("pierwszy")
            return dataset

    class Drugi(PipelineElementBase):
        def _process_data(self, dataset):
            calls.append("drugi")
            return dataset

    dataset = _dataset(tmp_path, [{"text": "a", "label": ["x"]}])
    result = Pipeline(modules=[Pierwszy(), Drugi()])(dataset)

    assert result is dataset
    assert calls == ["pierwszy", "drugi"]


def test_element_rejects_unsupported_dataset_type():
    class Odmawiajacy(PipelineElementBase):
        ALLOWED_DATASET_TYPES = (TextClassificationDataset,)

        def _process_data(self, dataset):
            return dataset

    class InnyTyp:
        pass

    with pytest.raises(AssertionError, match="does not support"):
        Odmawiajacy()(InnyTyp())


def test_pipeline_element_without_type_restrictions_accepts_everything():
    class Dopuszczajacy(PipelineElementBase):
        def _process_data(self, dataset):
            return dataset

    class InnyTyp:
        pass

    assert Dopuszczajacy()(InnyTyp())


def test_remove_unlabelled_data(tmp_path):
    dataset = _dataset(
        tmp_path,
        [
            {"text": "a", "label": ["x"]},
            {"text": "b", "label": []},
        ],
    )
    result = RemoveUnlabelledData()(dataset)
    assert [item["text"] for item in result.raw_data] == ["a"]


def test_remove_unlabelled_data_rejects_unknown_type():
    class NieznanyTyp:
        pass

    with pytest.raises(TypeError, match="does not support"):
        RemoveUnlabelledData()(NieznanyTyp())


def test_remove_duplicates(tmp_path):
    dataset = _dataset(
        tmp_path,
        [
            {"text": "a", "label": ["x"]},
            {"text": "a", "label": ["y"]},
            {"text": "b", "label": ["x"]},
        ],
    )
    result = RemoveDuplicates()(dataset)
    assert [item["text"] for item in result.raw_data] == ["a", "b"]


def test_remove_long_iob_annotation_drops_annotations_closer_to_full_text():
    class ZbiorIOB:
        """
        Minimalna imitacja SequenceLabellingDataset - RemoveLongIOBAnnotation
        nie wymaga innych metod tego typu.
        """
        class_label_field_name = "label"

        @property
        def raw_data(self):
            return self._raw_data

        @raw_data.setter
        def raw_data(self, value):
            self._raw_data = value

    dataset = ZbiorIOB()
    dataset._raw_data = [
        {
            "text": "aaaa",
            "metadata": {
                "label": [[0, 4, "B-X", "I-X"], [1, 2, "B-Y", "I-Y"]]
            },
        }
    ]

    RemoveLongIOBAnnotation(ans_factor=0.85)(dataset)

    # pierwsza aniacja pokrywa caly tekst i zostaje odrzucona, druga zostaje
    assert dataset._raw_data[0]["metadata"]["label"] == [[1, 2, "B-Y", "I-Y"]]
