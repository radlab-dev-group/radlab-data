import json

import pytest

from radlab_data.datasets.tokenization_handler import TokenizationHandler
from radlab_data.text.utils import TextUtils
from radlab_data.utils.api_handler import APIHandler
from radlab_data.utils.metrics import (
    compute_metrics,
    preds_to_class_label_argmax,
    preds_to_class_label_softmax,
)
from radlab_data.utils.threads import ThreadWorker, WorkerCluster


class _Predictions:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


def test_softmax_and_argmax_are_consistent():
    import torch

    predictions = [[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]]
    soft = preds_to_class_label_softmax(predictions)
    assert soft.shape == (2, 3)
    # softmax w tym module jest liczony po dim=0 (po wszystkich przykladach),
    # wiec suma po wierszu nie wynosi 1.
    assert soft.sum(dim=0).tolist() == pytest.approx([1.0, 1.0, 1.0])
    # funkcja przyjmuje tensory (w compute_metrics dostaje wynik softmax)
    argmax = preds_to_class_label_argmax(
        [torch.Tensor(row) for row in predictions]
    )
    assert [int(x) for x in argmax] == [1, 0]


def test_compute_metrics():
    predictions = _Predictions(
        predictions=[[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]],
        label_ids=[1, 0],
    )
    metrics = compute_metrics(predictions)
    assert set(metrics) == {"f1", "precision", "recall", "accuracy"}
    assert metrics["accuracy"] == 1.0


def test_compute_metrics_with_wrong_labels():
    predictions = _Predictions(
        predictions=[[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]],
        label_ids=[1, 1],
    )
    metrics = compute_metrics(predictions)
    assert metrics["accuracy"] == 0.5


def test_tokenization_handler_tokenizes_dict_strings():
    class _FakeTokenizer:
        def __call__(self, texts):
            return {"input_ids": [[len(text)] for text in texts]}

    handler = TokenizationHandler(_FakeTokenizer())
    data = {"tekst": "abc", "lista": ["x", "yz"], "pozostaw": 5}
    handler.tokenize(data)
    assert data == {"tekst": [3], "lista": [[1], [2]], "pozostaw": 5}


def test_tokenization_handler_respects_blacklist():
    class _FakeTokenizer:
        def __call__(self, texts):
            return {"input_ids": [[len(text)] for text in texts]}

    handler = TokenizationHandler(_FakeTokenizer(), blacklist=["pominij"])
    data = {"pominij": "abc", "inny": "x"}
    handler.tokenize(data)
    assert data == {"pominij": "abc", "inny": [1]}


def test_text_utils_map_sentences_to_text():
    mapping = TextUtils.map_sentences_to_text(
        ["Pierwsze zdanie.", "Drugie zdanie."],
        "Pierwsze zdanie. Drugie zdanie.",
    )
    assert len(mapping) == 2
    assert mapping[0] == [0, 16]
    assert mapping[1][0] == 17
    assert mapping[1][1] == 17 + len("Drugie zdanie.")


def test_thread_worker_runs_task():
    results = []

    class Pracownik(ThreadWorker):
        def _task(self):
            results.append("wykonano")

    worker = Pracownik()
    worker.start()
    worker.join()
    assert results == ["wykonano"]


def test_worker_cluster_start_and_join():
    results = []

    class Pracownik(ThreadWorker):
        def _task(self):
            results.append("ok")

    cluster = WorkerCluster("test", [Pracownik(), Pracownik()])
    cluster.start()
    cluster.join()
    assert sorted(results) == ["ok", "ok"]


def test_api_handler_propagates_unexpected_errors(monkeypatch):
    def fake_get(url, **kwargs):
        raise Exception("połaczenie nieudane")

    monkeypatch.setattr("radlab_data.utils.api_handler.requests.get", fake_get)
    handler = APIHandler("http://127.0.0.1:1")
    with pytest.raises(Exception, match="połaczenie nieudane"):
        handler.call(endpoint="/ping")


def test_api_handler_returns_none_on_request_exception(monkeypatch):
    import requests

    class FakeResponse:
        def raise_for_status(self):
            raise requests.ConnectionError("brak polaczenia")

    def fake_get(url, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("radlab_data.utils.api_handler.requests.get", fake_get)
    handler = APIHandler("http://127.0.0.1:1")
    assert handler.call(endpoint="/ping") is None


def test_api_handler_can_raise_on_failure(monkeypatch):
    import requests

    class FakeResponse:
        def raise_for_status(self):
            raise requests.ConnectionError("brak polaczenia")

    def fake_get(url, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("radlab_data.utils.api_handler.requests.get", fake_get)
    handler = APIHandler("http://127.0.0.1:1", raise_errors=True)
    with pytest.raises(Exception, match="brak polaczenia"):
        handler.call(endpoint="/ping")


def test_hf_training_arguments_use_eval_strategy(tmp_path):
    from radlab_data.utils.hf_training_arguments import HFTrainingArgumentHandler

    handler = HFTrainingArgumentHandler(out_dir=str(tmp_path))
    arguments = handler.training_arguments.to_dict()
    assert arguments["eval_strategy"] == "steps"
    assert "evaluation_strategy" not in arguments


def test_hf_training_arguments_save_and_load_roundtrip(tmp_path):
    from radlab_data.utils.hf_training_arguments import HFTrainingArgumentHandler

    out_file = tmp_path / "arguments.json"
    source = HFTrainingArgumentHandler(out_dir=str(tmp_path))
    source.save(str(out_file))

    loaded = HFTrainingArgumentHandler.from_config(str(out_file))
    assert loaded.training_arguments.to_dict() == source.training_arguments.to_dict()
