import logging
import subprocess
import sys

import pytest

from radlab_data.text.loaders.config import AVAILABLE_FILE_EXTENSIONS
from radlab_data.text.reader import DirectoryFileReader


def test_lists_files_and_skips_unsupported(sample_documents_dir):
    reader = DirectoryFileReader(main_dir_path=sample_documents_dir)
    reader.load()

    extensions = {path.split(".")[-1] for path in reader._files_in_directory}
    assert extensions == {"txt", "md", "html", "pdf", "docx"}
    assert len(reader.documents) == 5


def test_default_accept_extensions_is_a_copy():
    before = list(AVAILABLE_FILE_EXTENSIONS)
    DirectoryFileReader(main_dir_path="/nieistniejacy-katalog")
    assert AVAILABLE_FILE_EXTENSIONS == before


def test_custom_accept_extensions(sample_documents_dir):
    reader = DirectoryFileReader(
        main_dir_path=sample_documents_dir, accept_extensions=["md"]
    )
    assert reader._files_in_directory


def test_relative_file_path(sample_documents_dir):
    reader = DirectoryFileReader(main_dir_path=sample_documents_dir)
    reader.load()

    relative_paths = {
        document.as_dict["relative_filepath"]
        for document in reader.documents
    }
    assert all(
        not relative_path.startswith("/") for relative_path in relative_paths
    )


def test_relative_file_path_with_repeated_directory_name(tmp_path):
    """
    Poprawka: sciezka wzgledna nie moze tracic fragmentu, gdy nazwa katalogu
    glownego powtarza sie w głąbszych poziomach.
    """
    main_dir = tmp_path / "x"
    (main_dir / "x").mkdir(parents=True)
    plik = main_dir / "x" / "plik.txt"
    plik.write_text("treść", encoding="utf-8")

    reader = DirectoryFileReader(main_dir_path=str(main_dir))
    reader.load()

    assert reader.documents[0].as_dict["relative_filepath"] == "x/plik.txt"


def test_failed_file_is_skipped_and_logged(tmp_path, caplog):
    main_dir = tmp_path / "dokumenty"
    main_dir.mkdir()
    (main_dir / "ok.txt").write_text("poprawny plik", encoding="utf-8")
    (main_dir / "zlamany.txt").write_bytes(b"\xff\xfe\x00\x81\x8d")

    with caplog.at_level(logging.WARNING):
        reader = DirectoryFileReader(main_dir_path=str(main_dir))
        reader.load()

    assert len(reader.documents) == 1
    assert reader.documents[0].file_path.endswith("ok.txt")
    assert any("zlamany.txt" in record.message for record in caplog.records)


def test_multiprocessing_loading(sample_documents_dir):
    reader = DirectoryFileReader(main_dir_path=sample_documents_dir, processes_count=2)
    reader.load()

    assert len(reader.documents) == 5
    assert {document.file_type for document in reader.documents} == {
        "txt",
        "md",
        "html",
        "pdf",
        "docx",
    }


def test_core_modules_import_without_the_model_stack():
    """
    The document-loading core must be importable on a host without
    ``torch``/``transformers``/``spacy`` (the query-only hosts of the
    semantic search engine).  A subprocess is used, because a module which is
    already imported cannot be un-imported in the test process.
    """
    import subprocess

    probe = (
        "import builtins, sys\n"
        "blocked = {'torch', 'transformers', 'spacy', 'sentence_transformers'}\n"
        "real = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in blocked:\n"
        "        raise ImportError(name)\n"
        "    return real(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "import radlab_data.text.document\n"
        "import radlab_data.text.reader\n"
        "import radlab_data.text.utils\n"
        "import radlab_data.text.loaders.loader_i\n"
        "import radlab_data.utils.threads\n"
        "import radlab_data.utils.api_handler\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
