import pytest

from radlab_data.text.document import Document
from radlab_data.text.loaders.base import DocumentPage, read_text_file
from radlab_data.text.loaders.docx import DOCXLoader
from radlab_data.text.loaders.html import HTMLLoader
from radlab_data.text.loaders.input_txt import InputTextLoader
from radlab_data.text.loaders.loader_i import get_tokenizer
from radlab_data.text.loaders.markdown import MarkdownLoader
from radlab_data.text.loaders.pdf import PDFLoader
from radlab_data.text.loaders.txt import TXTLoader


def test_document_page_to_dict():
    page = DocumentPage(page_content="treść", metadata={"page": 1})
    assert page.to_dict() == {"page_content": "treść", "metadata": {"page": 1}}


def test_document_page_equality_and_independent_metadata():
    first = DocumentPage(page_content="a")
    second = DocumentPage(page_content="a")
    third = DocumentPage(page_content="a")
    assert first == second
    first.metadata["page"] = 1
    assert third.metadata == {}


def test_read_text_file_utf8(txt_file):
    assert read_text_file(txt_file).startswith("Pierwsza linia tekstu.")


def test_read_text_file_falls_back_to_cp1250(latin2_txt_file):
    assert read_text_file(latin2_txt_file).startswith("Zażółć gęślą jaźń")


def test_read_text_file_without_autodetect_raises(latin2_txt_file):
    with pytest.raises(RuntimeError):
        read_text_file(latin2_txt_file, autodetect_encoding=False)


def test_read_text_file_raises_on_binary(tmp_path):
    binary = tmp_path / "smieci.bin"
    binary.write_bytes(b"\xff\xfe\x00\x01\x81\x8d\x8f")
    with pytest.raises(RuntimeError):
        read_text_file(str(binary))


from types import SimpleNamespace


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(path):
        _FakeAutoTokenizer.calls.append(path)
        return SimpleNamespace(model_max_length=1)


def test_get_tokenizer_is_cached(monkeypatch):
    _FakeAutoTokenizer.calls = []
    monkeypatch.setattr("radlab_data.text.loaders.loader_i.AutoTokenizer", _FakeAutoTokenizer)
    get_tokenizer.cache_clear()
    first = get_tokenizer("sczytnik/testowy")
    second = get_tokenizer("sczytnik/testowy")
    assert first is second
    assert _FakeAutoTokenizer.calls == ["sczytnik/testowy"]
    get_tokenizer.cache_clear()


def test_txt_loader(txt_file):
    pages = TXTLoader(txt_file).load()
    assert len(pages) == 1
    assert pages[0].metadata["source"] == txt_file
    assert "Trzeci akapit" in pages[0].page_content


def test_markdown_loader_keeps_markup_by_default(markdown_file):
    page = MarkdownLoader(markdown_file).load()[0]
    assert "# Nagłówek pierwszy" in page.page_content
    assert page.metadata["document_format"] == "markdown"


def test_markdown_loader_strips_markup(markdown_file):
    loader = MarkdownLoader(markdown_file, options={"strip_markdown": True})
    content = loader.load()[0].page_content
    assert "Nagłówek pierwszy" in content
    for markup in ["#", "**", "](", "```", "|---", "---\ntitle"]:
        assert markup not in content


def test_markdown_loader_keeps_code_content(markdown_file):
    loader = MarkdownLoader(markdown_file, options={"strip_markdown": True})
    content = loader.load()[0].page_content
    assert "print('*w kodzie*')" in content


def test_docx_loader_paragraphs_and_table(docx_file):
    pages = DOCXLoader(docx_file).load()
    contents = [page.page_content for page in pages]
    assert "Akapit pierwszy dokumentu docx." in contents
    assert "wiersz1 kolumna1" in contents

    table_pages = [page for page in pages if "table" in page.metadata]
    assert table_pages[0].metadata["source"] == docx_file


def test_html_loader_removes_markup(html_file):
    content = HTMLLoader(html_file).load()[0].page_content
    assert "Pierwszy akapit treści." in content
    for fragment in ["Menu nawigacji", "Stopka strony", "nieIstotne", "color:red"]:
        assert fragment not in content


def test_html_loader_split_by_paragraph(html_file):
    loader = HTMLLoader(html_file, options={"split_by_paragraph": True})
    pages = loader.load()
    assert len(pages) > 1
    assert all(page.metadata["source"] == html_file for page in pages)
    assert [page.metadata["paragraph"] for page in pages] == list(range(1, len(pages) + 1))


def test_pdf_loader_pages_and_metadata(pdf_file):
    pages = PDFLoader(pdf_file).load()
    assert len(pages) == 2
    assert [page.page_content for page in pages] == [
        "Pierwsza strona pdf",
        "Druga strona pdf",
    ]
    first = pages[0].metadata
    assert first["source"] == pdf_file
    assert first["page"] == 0
    assert first["total_pages"] == 2
    assert first["title"] == "Tytul pdf"


def test_input_text_loader():
    loader = InputTextLoader(
        "tekst.input_text", options={"document_content": "Tekst z argumentu."}
    )
    assert loader.load()[0].page_content == "Tekst z argumentu."


def test_input_text_loader_requires_content():
    with pytest.raises(Exception, match="document_content"):
        InputTextLoader("tekst.input_text", options={})


@pytest.mark.parametrize(
    "loader_class", [TXTLoader, MarkdownLoader, DOCXLoader, HTMLLoader, PDFLoader]
)
def test_loaders_accept_options_none_and_are_pickleable(
    loader_class, txt_file, markdown_file, docx_file, html_file, pdf_file
):
    import copy

    path = {
        TXTLoader: txt_file,
        MarkdownLoader: markdown_file,
        DOCXLoader: docx_file,
        HTMLLoader: html_file,
        PDFLoader: pdf_file,
    }[loader_class]
    loader = loader_class(path)
    assert loader.options == {}
    loader.load()
    assert copy.deepcopy(loader.doc_pages) == loader.doc_pages


def test_document_resolves_extension_and_builds_dict(txt_file):
    document = Document(file_path=txt_file)
    document.load()
    as_dict = document.as_dict
    assert as_dict["filepath"] == txt_file
    assert as_dict["pages"][0]["page_content"].startswith("Pierwsza linia")


def test_document_rejects_unknown_extension(tmp_path):
    arkusz = tmp_path / "dane.xlsx"
    arkusz.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        Document(file_path=str(arkusz)).load()


def test_document_supports_every_registered_extension(
    txt_file, markdown_file, docx_file, html_file, pdf_file
):
    for path in [txt_file, markdown_file, docx_file, html_file, pdf_file]:
        document = Document(file_path=path)
        document.load()
        assert document.as_dict["pages"]


def test_as_dict_does_not_mutate_options(txt_file):
    document = Document(
        file_path=txt_file,
        relative_file_path="podkatalog/dokument.txt",
        document_category="kategoria",
    )
    document.load()
    first = document.as_dict
    assert first["relative_filepath"] == "podkatalog/dokument.txt"
    assert first["category"] == "kategoria"
    assert "relative_file_path" in document.global_reader_options
    assert "document_category" in document.global_reader_options


def test_document_can_be_loaded_twice(txt_file):
    document = Document(
        file_path=txt_file,
        relative_file_path="podkatalog/dokument.txt",
        document_category="kategoria",
    )
    document.load()
    document.load()
    assert document.as_dict["relative_filepath"] == "podkatalog/dokument.txt"
    assert document.as_dict["category"] == "kategoria"


def test_document_merges_pages(txt_file):
    document = Document(file_path=txt_file, merge_document_pages=True)
    document.load()
    pages = document.as_dict["pages"]
    assert len(pages) == 1
    assert "Pierwsza linia" in pages[0]["page_content"]
    assert "Ostatnia linia" in pages[0]["page_content"]


def test_document_splits_into_chunks(txt_file):
    document = Document(file_path=txt_file, max_tokens_in_chunk=4)
    document.load()
    pages = document.as_dict["pages"]
    assert len(pages) > 1
    chunk_numbers = [page["metadata"]["chunk"] for page in pages]
    assert chunk_numbers == list(range(1, len(pages) + 1))


def test_document_clears_texts(txt_file):
    document = Document(file_path=txt_file, clear_texts=True)
    document.load()
    assert document.as_dict["pages"][0]["metadata"]["text_chunk_type"] == "clear_texts"
