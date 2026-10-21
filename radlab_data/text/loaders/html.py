import re

from bs4 import BeautifulSoup

from radlab_data.text.loaders.base import DocumentPage, read_text_file
from radlab_data.text.loaders.loader_i import LoaderI


class HTMLLoader(LoaderI):
    """
    Loader for HTML files. All markup which does not contribute to the
    readable content (scripts, styles, metadata, ...) is removed and the
    cleaned text is exposed as one or many document pages.

    Recognised options:
      * `split_by_paragraph`: when set to True, each paragraph (a fragment
        separated by a blank line) becomes a separate document page. Without
        this option the whole file is loaded as a single page.
    """

    TAGS_TO_REMOVE = [
        "script",
        "style",
        "noscript",
        "svg",
        "meta",
        "link",
        "head",
        "footer",
        "header",
    ]

    MULTIPLE_EMPTY_LINES_RE = re.compile(r"\n\s*\n+")
    MULTIPLE_SPACES_RE = re.compile(r"[ \t]+")

    def __init__(
        self,
        filepath: str,
        options: dict = None,
        encoding: str = "utf-8",
        autodetect_encoding: bool = True,
    ) -> None:
        """
        :param filepath: Path to html file
        :param options: Options shared with other loaders (see `LoaderI`),
        see the class docstring for the loader specific options
        :param encoding: Encoding used to read the file
        :param autodetect_encoding: If the file cannot be decoded with
        `encoding`, encodings from `FALLBACK_ENCODINGS` are tried
        """
        LoaderI.__init__(self, filepath=filepath, options=options)
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> list:
        """
        Load the html file and remove all its markup.
        :return: List of loaded document pages
        """
        if self.doc_pages:
            return self.doc_pages

        clean_text = self._clean_html(
            read_text_file(
                self.filepath,
                encoding=self.encoding,
                autodetect_encoding=self.autodetect_encoding,
            )
        )

        if not self.options.get("split_by_paragraph", False):
            self.doc_pages = [
                DocumentPage(
                    page_content=clean_text,
                    metadata={"source": str(self.filepath)},
                )
            ]
            return self.doc_pages

        paragraph_number = 0
        for paragraph in clean_text.split("\n\n"):
            if not paragraph.strip():
                continue
            paragraph_number += 1
            self.doc_pages.append(
                DocumentPage(
                    page_content=paragraph.strip(),
                    metadata={
                        "source": str(self.filepath),
                        "paragraph": paragraph_number,
                    },
                )
            )
        return self.doc_pages

    def _clean_html(self, raw_html: str) -> str:
        """
        Remove tags which are not a content of the document and normalize
        whitespace. Paragraph breaks are preserved, multiple spaces and
        tabs are reduced to a single space.
        :param raw_html: Raw html content
        :return: Cleaned, readable text
        """
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(self.TAGS_TO_REMOVE):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = self.MULTIPLE_EMPTY_LINES_RE.sub("\n\n", text)
        text = self.MULTIPLE_SPACES_RE.sub(" ", text)
        return text.strip()
