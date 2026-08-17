import os.path

from radlab_data.text.loaders.pdf import PDFLoader
from radlab_data.text.loaders.txt import TXTLoader
from radlab_data.text.loaders.docx import DOCXLoader
# from radlab_data.text.loaders.html import HTMLLoader
from radlab_data.text.loaders.input_txt import InputTextLoader

from radlab_data.text.loaders.config import (
    AVAILABLE_FILE_EXTENSIONS,
    AVAILABLE_FILE_EXTENSIONS_STR,
)


class Document:
    """
    Definition of any loaded document (pdf, docx, txt). It is general definition
    which may be used to read content from any file with supported extensions.

    The full list of supported extension to read files, are defined into
    radlab_data.text.loaders.config.AVAILABLE_FILE_EXTENSIONS
    """

    EXTENSION_TO_LOADER = {
        "txt": TXTLoader,
        "pdf": PDFLoader,
        "docx": DOCXLoader,
        # "html": HTMLLoader,
        "input_text": InputTextLoader,
    }

    def __init__(
        self,
        file_path: str,
        relative_file_path: str = None,
        file_name: str = None,
        prepare_proper_pages: bool = False,
        merge_document_pages: bool = False,
        clear_texts: bool = False,
        use_text_denoiser: bool = False,
        max_tokens_in_chunk: int = None,
        number_of_overlap_tokens: int = None,
        check_text_language: bool = False,
        document_category: str = None,
        document_content: str = None,
    ) -> None:
        """
        :param file_path: Path to file to read
        :param relative_file_path: Relative path to file
        :param file_name: Name of read file
        :param prepare_proper_pages: IF set to True, then all texts/sentences
        from the same page will be merged to single page.
        :param merge_document_pages: If set to True then all pages from document
        will be merged to single page, then document will contain single page.
        :param clear_texts: If set to True, then each text will be
        :param max_tokens_in_chunk: If given then each text will be split
        to chunks with given max number of tokens in single chunk
        :param number_of_overlap_tokens: If given then each  generated chunk
        will contain number_of_overlap_tokens from the previous chunk.
        :param check_text_language: If options is set to True, then language
        of each text will be checked and stored to metadata
        :param document_category: Category of document
        :param document_content: Content of document
        """
        self.file_path = file_path
        self.relative_file_path = relative_file_path
        self.file_name = file_name
        self.file_type = None

        # Loaded document
        self._document = None
        self._document_as_dict = None

        # Some global options passed to each document loader
        self.global_reader_options = {
            "merge_to_proper_pages": prepare_proper_pages,
            "merge_document_pages": merge_document_pages,
            "clear_texts": clear_texts,
            "use_text_denoiser": use_text_denoiser,
            "check_text_language": check_text_language,
            "max_tokens_in_chunk": max_tokens_in_chunk,
            "number_of_overlap_tokens": number_of_overlap_tokens,
            "relative_file_path": relative_file_path,
            "document_category": document_category,
            "document_content": document_content,
        }

    def __iter__(self):
        """
        Added iterator over document pages
        :return: Iterator over document pages
        """
        for p in self._document:
            yield p

    @property
    def as_dict(self):
        return self._document_as_dict

    def load(self):
        """
        Load the document. Prepare document loader and load the document.
        :return: loaded document
        """
        assert self.file_path is not None
        if self.file_name is None:
            self.file_name = os.path.basename(self.file_path)
        assert self.file_name is not None and len(self.file_name) > 0

        self.file_type = self._resolve_file_type()
        document_loader = self.EXTENSION_TO_LOADER[self.file_type](
            self.file_path, options=self.global_reader_options
        )
        self._document = document_loader.load()

        # Preprocess document if is needed
        document_loader.merge_all_pages()
        document_loader.clear_doc_pages()
        document_loader.split_to_chunks()
        document_loader.check_language()

        # convert preprocessed document as dict
        self._document_as_dict = document_loader.as_dict()

        return self._document

    def _resolve_file_type(self):
        file_extension = self.file_path.split(".")[-1].lower().strip()
        if file_extension not in AVAILABLE_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension {file_extension}. "
                f"Available extensions are: {AVAILABLE_FILE_EXTENSIONS_STR}"
            )
        return file_extension
