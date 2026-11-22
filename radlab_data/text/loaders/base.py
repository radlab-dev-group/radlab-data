from dataclasses import dataclass, field


@dataclass
class DocumentPage:
    """
    Single page (fragment) of a loaded document.

    Objects of this class are created by document loaders and consumed by
    `LoaderI` (merging, clearing, chunking, converting to dictionary).
    Both fields are mutable in place, pages are compared by their content
    and metadata, and instances can be pickled and deep copied, so they can
    be safely used with multiprocessing.
    """

    page_content: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Return page content and metadata as a plain dictionary
        :return: Dictionary with `page_content` and `metadata` keys
        """
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }


# Encodings used to read a text file when it cannot be decoded with the
# requested one. Single byte encodings which decode every byte sequence
# (like `latin-1`) are intentionally not listed here, because they would
# silently accept garbage instead of reporting a problem.
FALLBACK_ENCODINGS = ["utf-8-sig", "cp1250", "cp1252"]


def read_text_file(
    filepath: str, encoding: str = "utf-8", autodetect_encoding: bool = True
) -> str:
    """
    Read the whole text file into a single string.

    At first the file is read with the given `encoding`. When it fails and
    `autodetect_encoding` is set to True, encodings from `FALLBACK_ENCODINGS`
    are tried one after another.

    :param filepath: Path to the text file to read
    :param encoding: Encoding used to read the file
    :param autodetect_encoding: If True, fallback encodings will be tried
    when the file cannot be decoded with the given `encoding`
    :return: Content of the file
    """
    candidate_encodings = [encoding]
    if autodetect_encoding:
        candidate_encodings += [enc for enc in FALLBACK_ENCODINGS if enc != encoding]

    decoding_error = None
    for candidate_encoding in candidate_encodings:
        try:
            with open(filepath, "r", encoding=candidate_encoding) as text_file:
                return text_file.read()
        except UnicodeDecodeError as error:
            decoding_error = error

    raise RuntimeError(f"Error loading {filepath}") from decoding_error
