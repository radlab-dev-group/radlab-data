from langchain_core.documents import Document

from radlab_data.text.loaders.loader_i import LoaderI


class InputTextLoader(LoaderI):
    """
    Loads given text string as Document
    """

    DOCUMENT_CONTENT_VARIABLE = "document_content"

    def __init__(self, filepath: str, options: dict = None) -> None:
        """
        :param filepath: Filename extension have to be: `input_text`
        """
        LoaderI.__init__(self, filepath=filepath, options=options)

        if self.DOCUMENT_CONTENT_VARIABLE not in options:
            raise Exception(
                f"No {self.DOCUMENT_CONTENT_VARIABLE} in "
                f"options while loading input_text!"
            )
        self._content = options[self.DOCUMENT_CONTENT_VARIABLE]

    def load(self):
        self.doc_pages = [Document(page_content=self._content, metadata={"page": 1})]
        return self.doc_pages
