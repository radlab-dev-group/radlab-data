import logging
from pathlib import Path

import docx
import pytest

from radlab_data.text.pair_reader import (
    AVAILABLE_POSTFIXES,
    DirectoryPairsReader,
)


def _make_docx(path: Path, text: str):
    document = docx.Document()
    document.add_paragraph(text)
    document.save(str(path))


def test_numbered_configuration_uses_numbered_class():
    assert AVAILABLE_POSTFIXES["1_2"].__name__ == "FirstSecondNumberPairElems"
    assert AVAILABLE_POSTFIXES["1_2"].ELEMS_TO_FILE_POSTFIX == {
        "1": "_1.docx",
        "2": "_2.docx",
    }


def test_first_second_configuration_uses_first_second_class():
    assert AVAILABLE_POSTFIXES["first_second"].__name__ == "FirstSecondPairElems"


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="Invalid postfix configuration"):
        DirectoryPairsReader("/katalog", "nie_ma_takiej_konfiguracji")


def test_pairs_are_matched_and_unmatched_files_skipped(tmp_path):
    _make_docx(tmp_path / "dokument_1.docx", "jedynka")
    _make_docx(tmp_path / "dokument_2.docx", "dwójka")
    _make_docx(tmp_path / "samotny.docx", "samotny")

    reader = DirectoryPairsReader(str(tmp_path), "1_2")
    pairs = reader.prepare_pairs()

    assert list(pairs.keys()) == ["dokument"]
    pair = pairs["dokument"]
    assert pair["1"].endswith("dokument_1.docx")
    assert pair["2"].endswith("dokument_2.docx")


def test_invalid_filenames_are_not_added_to_pairs(tmp_path, caplog):
    (tmp_path / "przypadkowy_plik.txt").write_text("nie docx", encoding="utf-8")
    (tmp_path / "samotny.docx").write_text("samotny", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        reader = DirectoryPairsReader(str(tmp_path), "1_2")
        pairs = reader.prepare_pairs()

    assert pairs == {}


def test_load_documents_builds_sentence_pairs(tmp_path):
    _make_docx(tmp_path / "parz_1.docx", "Pierwsze zdanie. Drugie zdanie.")
    _make_docx(tmp_path / "parz_2.docx", "Odpowiedz na pierwsze. Odpowiedz na drugie.")

    reader = DirectoryPairsReader(str(tmp_path), "1_2")
    reader.prepare_pairs()
    reader.load_documents()

    sentences = reader.document_pairs_sentences
    assert "parz" in sentences
    first, second = sentences["parz"]["1"], sentences["parz"]["2"]
    assert first == ["Pierwsze zdanie.", "Drugie zdanie."]
    assert second == ["Odpowiedz na pierwsze.", "Odpowiedz na drugie."]


def test_unique_rows_in_cell_deduplicates_table_rows(tmp_path):
    import docx

    path = tmp_path / "tabela.docx"
    document = docx.Document()
    document.add_paragraph("Naglowek akapitu.")
    table = document.add_table(rows=1, cols=3)
    table.cell(0, 0).text = "wartosc"
    table.cell(0, 1).text = "wartosc"
    table.cell(0, 2).text = "inna"
    document.save(str(path))

    from radlab_data.text.pair_reader import DirectoryPairsReader

    reader = DirectoryPairsReader(str(tmp_path))
    paragraphs = reader._load_docx_file_content(
        file_path=str(path), unique_rows_in_cell=True
    )
    assert paragraphs == ["Naglowek akapitu.", "wartosc", "inna"]

    reader2 = DirectoryPairsReader(str(tmp_path))
    without_dedup = reader2._load_docx_file_content(
        file_path=str(path), unique_rows_in_cell=False
    )
    assert without_dedup == [
        "Naglowek akapitu.",
        "wartosc",
        "wartosc",
        "inna",
    ]
