import re
import langchain_core.documents.base

from bs4 import BeautifulSoup

from langchain_community import document_loaders
from radlab_data.text.loaders.loader_i import LoaderI


class HTMLLoader(document_loaders.HTMLLoader, LoaderI):
    """
    Load an HTML file, strip away all markup (including scripts, styles, and other
    noisy elements), and expose the cleaned text as one or many
    :class:`langchain_core.documents.base.Document` objects.

    The loader mirrors the behaviour of the existing ``PDFLoader`` and ``DOCXLoader``:
    * it caches the result in ``self.doc_pages`` so repeated calls are cheap,
    * it accepts an optional ``options`` dictionary for fine‑grained control.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the HTML file to be processed.
    options : dict, optional
        Additional configuration values. Recognised keys:

        * ``split_by_paragraph`` (bool): when ``True`` each paragraph (separated by a
          blank line) becomes a separate ``Document``; otherwise the whole file is
          returned as a single document.

    Attributes
    ----------
    doc_pages : list
        Cached list of :class:`langchain_core.documents.base.Document` objects
        created after the first call to :meth:`load`.
    """

    def __init__(self, filepath: str, options: dict = None) -> None:
        """
        Initialise the loader.

        This constructor forwards ``filepath`` to the upstream
        :class:`langchain_community.document_loaders.HTMLLoader` (which only stores the
        path) and also initialises our own :class:`LoaderI` base class, which prepares
        ``self.doc_pages`` and stores the ``options`` dictionary.

        Args
        ----
        filepath : str
            Path to the HTML file.
        options : dict, optional
            See class docstring for supported keys. If omitted an empty dictionary
            is created internally.
        """
        if options is None:
            options = {}

        # Initialise the LangChain HTML loader (it only stores the path)
        document_loaders.HTMLLoader.__init__(self, file_path=filepath)
        # Initialise our own LoaderI base class
        LoaderI.__init__(self, filepath=filepath, options=options)

    def load(self) -> list:
        """
        Parse the HTML, clean it, and build ``Document`` objects.

        The method reads the file, removes unwanted tags via :meth:`_clean_html`,
        then either splits the resulting plain‑text into separate documents per
        paragraph (if ``split_by_paragraph`` is ``True``) or returns a single
        document containing the whole text.

        Returns
        -------
        list[langchain_core.documents.base.Document]
            A list of documents, each with ``page_content`` holding the cleaned text
            and ``metadata`` containing at least the source file path (and the
            paragraph number when splitting).
        """
        if self.doc_pages:  # cache – already loaded
            return self.doc_pages

        # Read the raw HTML
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw_html = f.read()

        clean_text = self._clean_html(raw_html)

        # Decide whether to split into separate documents
        split_by_paragraph = self.options.get("split_by_paragraph", False)

        if split_by_paragraph:
            # Each non‑empty paragraph becomes its own Document
            for idx, para in enumerate(
                filter(None, clean_text.split("\n\n")), start=1
            ):
                doc = langchain_core.documents.base.Document(
                    page_content=para.strip(),
                    metadata={"source": self.filepath, "paragraph": idx},
                )
                self.doc_pages.append(doc)
        else:
            # One big Document for the whole file
            doc = langchain_core.documents.base.Document(
                page_content=clean_text,
                metadata={"source": self.filepath},
            )
            self.doc_pages.append(doc)

        return self.doc_pages

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        """
        Strip unwanted tags and normalise whitespace.

        The function uses **BeautifulSoup** with the ``lxml`` parser to remove
        tags that do not contribute to the readable content (scripts, styles,
        meta data, etc.). After tag removal it extracts the raw text, collapses
        multiple line‑breaks to a single blank line (preserving paragraph
        separation) and reduces consecutive spaces/tabs to a single space.

        Parameters
        ----------
        raw_html : str
            The raw HTML source read from the file.

        Returns
        -------
        str
            Clean, human‑readable text ready for downstream processing.
        """
        soup = BeautifulSoup(raw_html, "lxml")

        # Tags we never want to keep
        for tag in soup(
            [
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
        ):
            tag.decompose()

        # Get the text, then tidy up whitespace
        text = soup.get_text(separator="\n")
        # Collapse multiple new‑lines / spaces
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # keep paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
