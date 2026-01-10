from langchain_community import document_loaders
from radlab_data.text.loaders.loader_i import LoaderI


class PDFLoader(document_loaders.PyPDFLoader, LoaderI):
    """
    Wrapper for document_loaders.PyPDFLoader
    """

    def __init__(self, filepath: str, options: dict = None) -> None:
        """
        :param filepath: Path to pdf file
        """
        document_loaders.PyPDFLoader.__init__(self, filepath)
        LoaderI.__init__(self, filepath=filepath, options=options)

    def load(self):
        self.doc_pages = document_loaders.PyPDFLoader.load(self)
        return self.doc_pages
