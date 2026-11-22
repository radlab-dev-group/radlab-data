import logging
from datetime import datetime

import pypdf

from radlab_data.text.loaders.base import DocumentPage
from radlab_data.text.loaders.loader_i import LoaderI


class PDFLoader(LoaderI):
    """
    Loader for PDF files based on the `pypdf` library. Each PDF page is
    loaded as a single document page.
    """

    STANDARD_METADATA_KEYS = {
        "source",
        "total_pages",
        "creationdate",
        "creator",
        "producer",
    }

    def __init__(
        self,
        filepath: str,
        options: dict = None,
        password: str = None,
        extraction_mode: str = "plain",
        extraction_kwargs: dict = None,
    ) -> None:
        """
        :param filepath: Path to pdf file
        :param options: Options shared with other loaders (see `LoaderI`)
        :param password: Password for an encrypted pdf file
        :param extraction_mode: `plain` or `layout` text extraction mode
        :param extraction_kwargs: Additional arguments passed to
        `pypdf.PageObject.extract_text`
        """
        LoaderI.__init__(self, filepath=filepath, options=options)
        self.password = password
        self.extraction_mode = extraction_mode
        self.extraction_kwargs = {} if extraction_kwargs is None else extraction_kwargs

    def load(self) -> list:
        """
        Load all pages of the pdf file.
        :return: List of loaded document pages
        """
        self.doc_pages = []
        with open(self.filepath, "rb") as pdf_file:
            pdf_reader = pypdf.PdfReader(pdf_file, password=self.password)
            document_metadata = self._document_metadata(pdf_reader)
            page_labels = self._page_labels(pdf_reader)

            for page_number, pdf_page in enumerate(pdf_reader.pages):
                metadata = dict(document_metadata)
                metadata["page"] = page_number
                if page_labels is not None:
                    metadata["page_label"] = page_labels[page_number]
                self.doc_pages.append(
                    DocumentPage(
                        page_content=self._extract_text(pdf_page).strip(),
                        metadata=self._validate_metadata(metadata),
                    )
                )
        return self.doc_pages

    def _extract_text(self, pdf_page) -> str:
        """
        Extract text from the single pdf page
        :param pdf_page: `pypdf` page object to extract text from
        :return: Extracted page text
        """
        if self.extraction_mode == "plain" and not self.extraction_kwargs:
            text = pdf_page.extract_text()
        else:
            text = pdf_page.extract_text(
                extraction_mode=self.extraction_mode, **self.extraction_kwargs
            )
        return text if text is not None else ""

    def _document_metadata(self, pdf_reader) -> dict:
        """
        Prepare metadata shared by all pages of the loaded document
        :param pdf_reader: `pypdf` reader of the loaded document
        :return: Document metadata
        """
        metadata = {"producer": "PyPDF", "creator": "PyPDF", "creationdate": ""}
        metadata.update(dict(pdf_reader.metadata or {}))
        metadata["source"] = str(self.filepath)
        metadata["total_pages"] = len(pdf_reader.pages)
        return self._purge_metadata(metadata)

    @staticmethod
    def _page_labels(pdf_reader):
        """
        Read labels of all pages. Malformed documents may not provide them.
        :param pdf_reader: `pypdf` reader of the loaded document
        :return: List of page labels or None when labels are not available
        """
        try:
            return list(pdf_reader.page_labels)
        except Exception as error:
            logging.debug(f"Page labels are not available: {error}")
            return None

    @staticmethod
    def _purge_metadata(metadata: dict) -> dict:
        """
        Clean metadata keys and values of the loaded document. Keys are
        lowercased and stripped with the leading `/` (as used by pdf files),
        dates are converted to the ISO format and only string and integer
        values are kept.
        :param metadata: Metadata read from the pdf document
        :return: Cleaned metadata
        """
        normalized_keys = {"page_count": "total_pages", "file_path": "source"}
        purged_metadata = {}
        for key, value in metadata.items():
            if not isinstance(value, (str, int)):
                value = str(value)
            if key.startswith("/"):
                key = key[1:]
            key = key.lower()

            if key in ("creationdate", "moddate"):
                try:
                    purged_metadata[key] = datetime.strptime(
                        value.replace("'", ""), "D:%Y%m%d%H%M%S%z"
                    ).isoformat("T")
                except ValueError:
                    purged_metadata[key] = value
            elif key in normalized_keys:
                purged_metadata[normalized_keys[key]] = value
                purged_metadata[key] = value
            elif isinstance(value, str):
                purged_metadata[key] = value.strip()
            elif isinstance(value, int):
                purged_metadata[key] = value
        return purged_metadata

    def _validate_metadata(self, metadata: dict) -> dict:
        """
        Check if metadata of the page contains all standard keys and
        the page number is an integer
        :param metadata: Metadata of the loaded page
        :return: Validated metadata
        """
        if not self.STANDARD_METADATA_KEYS.issubset(metadata.keys()):
            raise ValueError(
                "Metadata of the loaded pdf document have to contain "
                f"the following keys: {sorted(self.STANDARD_METADATA_KEYS)}"
            )
        if not isinstance(metadata.get("page", 0), int):
            raise ValueError("The PDF metadata page must be an integer.")
        return metadata
