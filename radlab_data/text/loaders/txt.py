from radlab_data.text.loaders.base import DocumentPage, read_text_file
from radlab_data.text.loaders.loader_i import LoaderI


class TXTLoader(LoaderI):
    """
    Loader for plain text files. The whole file is loaded as a single page.
    """

    def __init__(
        self,
        filepath: str,
        options: dict = None,
        encoding: str = "utf-8",
        autodetect_encoding: bool = True,
    ) -> None:
        """
        :param filepath: Path to text file
        :param options: Options shared with other loaders (see `LoaderI`)
        :param encoding: Encoding used to read the file
        :param autodetect_encoding: If the file cannot be decoded with
        `encoding`, encodings from `FALLBACK_ENCODINGS` are tried
        """
        LoaderI.__init__(self, filepath=filepath, options=options)
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> list:
        """
        Load the text file.
        :return: List with the single document page holding the file content
        """
        text = read_text_file(
            self.filepath,
            encoding=self.encoding,
            autodetect_encoding=self.autodetect_encoding,
        )
        self.doc_pages = [
            DocumentPage(
                page_content=text,
                metadata={"source": str(self.filepath)},
            )
        ]
        return self.doc_pages
