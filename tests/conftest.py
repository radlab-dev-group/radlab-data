import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minipdf import build_pdf  # noqa: E402

from radlab_data.text.loaders import loader_i

TEXT_CONTENT = (
    "Pierwsza linia tekstu.\nDruga linia z  wieloma   spacjami.\n\n"
    "Trzeci akapit ze skrótem itp. i numerem 12,5.\n\nOstatnia linia.\n"
)

MARKDOWN_CONTENT = """---
title: Metadane dokumentu
---

# Nagłówek pierwszy

Tekst **pogrubiony**, *kursywa* i [link](https://example.org) oraz `kod inline`.

## Lista

- pierwszy punkt
- drugi *punkt*
- [ ] zadanie do zrobienia

> Cytat z bloku.

```python
print('*w kodzie*')
```

| A | B |
|---|---|
| 1 | 2 |
"""

HTML_CONTENT = """<html>
<head><title>Tytuł strony</title><style>.a{color:red}</style></head>
<body>
<header>Menu nawigacji</header>
<h1>Nagłówek</h1>
<p>Pierwszy akapit treści.</p>
<p>Drugi akapit treści.</p>
<script>var nieIstotne = 1;</script>
<footer>Stopka strony</footer>
</body>
</html>
"""


@pytest.fixture(scope="session")
def word_level_tokenizer():
    """
    Miniaturowy tokenizer słowny, dzięki czemu testy nie pobierają żadnego
    modelu z Hugging Face.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tokenizer.train_from_iterator(
        [TEXT_CONTENT, MARKDOWN_CONTENT, HTML_CONTENT, "ala ma kot kota"], trainer
    )
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


@pytest.fixture(autouse=True)
def patched_tokenizer(word_level_tokenizer, monkeypatch):
    """
    Podmiana ładowania tokenera w loaderach dokumentów. Bez niej każdy
    loader pobierałby model z sieci.
    """
    monkeypatch.setattr(loader_i, "get_tokenizer", lambda tokenizer_path=None: word_level_tokenizer)
    return word_level_tokenizer


@pytest.fixture
def txt_file(tmp_path):
    path = tmp_path / "dokument.txt"
    path.write_text(TEXT_CONTENT, encoding="utf-8")
    return str(path)


@pytest.fixture
def latin2_txt_file(tmp_path):
    path = tmp_path / "dokument_zaorany.txt"
    path.write_bytes("Zażółć gęślą jaźń — test kodowania.\n".encode("cp1250"))
    return str(path)


@pytest.fixture
def markdown_file(tmp_path):
    path = tmp_path / "dokument.md"
    path.write_text(MARKDOWN_CONTENT, encoding="utf-8")
    return str(path)


@pytest.fixture
def html_file(tmp_path):
    path = tmp_path / "strona.html"
    path.write_text(HTML_CONTENT, encoding="utf-8")
    return str(path)


@pytest.fixture
def docx_file(tmp_path):
    import docx

    path = tmp_path / "dokument.docx"
    document = docx.Document()
    document.add_paragraph("Akapit pierwszy dokumentu docx.")
    document.add_paragraph("Akapit drugi dokumentu docx.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "wiersz1 kolumna1"
    table.cell(0, 1).text = "wiersz1 kolumna2"
    table.cell(1, 0).text = "wiersz2 kolumna1"
    table.cell(1, 1).text = "wiersz2 kolumna2"
    document.save(str(path))
    return str(path)


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "dokument.pdf"
    path.write_bytes(
        build_pdf(["Pierwsza strona pdf", "Druga strona pdf"], title="Tytul pdf")
    )
    return str(path)


@pytest.fixture
def sample_documents_dir(tmp_path):
    """Katalog z plikami o wszystkich obsługiwanych rozszerzeniach."""
    documents_dir = tmp_path / "katalog_dokumentow"
    documents_dir.mkdir()

    (documents_dir / "dokument.txt").write_text(TEXT_CONTENT, encoding="utf-8")
    (documents_dir / "dokument.md").write_text(MARKDOWN_CONTENT, encoding="utf-8")
    (documents_dir / "strona.html").write_text(HTML_CONTENT, encoding="utf-8")
    (documents_dir / "dokument.pdf").write_bytes(build_pdf(["Strona pdf"]))
    (documents_dir / "plik_do_pominiecia.csv").write_text("a;b\n", encoding="utf-8")

    import docx

    document = docx.Document()
    document.add_paragraph("Akapit z dokumentu docx.")
    document.save(str(documents_dir / "dokument.docx"))

    return str(documents_dir)
