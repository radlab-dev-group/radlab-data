from langchain_community import document_loaders

from radlab_data.text.loaders.loader_i import LoaderI


class TXTLoader(document_loaders.TextLoader, LoaderI):
    """
    Wrapper for document_loaders.PyPDFLoader
    """

    def __init__(self, filepath: str, options: dict = None) -> None:
        """
        :param filepath: Path to pdf file
        """
        document_loaders.TextLoader.__init__(self, filepath)
        LoaderI.__init__(self, filepath=filepath, options=options)

    def load(self):
        self.doc_pages = document_loaders.TextLoader.load(self)
        return self.doc_pages
