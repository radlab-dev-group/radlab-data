import abc


class TextProcessor(abc.ABC):
    """
    Base class to all text processors
    """

    def __init__(self):
        pass

    # def __call__(self, *args, **kwargs):
    #   self.apply()

    @abc.abstractmethod
    def apply(self, text: str, args: dict = None) -> str:
        """
        Abstract method to process any text and return processed text.
        :param text: Raw text
        :param args: Optionally arguments for text processor
        :return: Processed text as raw text
        """
        pass
