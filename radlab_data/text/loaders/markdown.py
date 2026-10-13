import re

from radlab_data.text.loaders.txt import TXTLoader


class MarkdownLoader(TXTLoader):
    """
    Loader for markdown (`md`) files.

    Markdown file is a plain text file, so reading is fully delegated to
    the TXTLoader. Additionally, each page is marked with the
    `document_format` metadata and, if the `strip_markdown` option is set
    to True, markdown markup (headings, emphasis, links, code fences,
    lists, tables, ...) is removed from the loaded text.
    """

    STRIP_MARKDOWN_OPTION = "strip_markdown"
    DOCUMENT_FORMAT = "markdown"

    _CODE_PLACEHOLDER = "\x00markdown_code_{index}\x00"

    _FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*\n", re.DOTALL)
    _FENCED_CODE_RE = re.compile(r"(?:```|~~~)[^\n]*\n(.*?)\n(?:```|~~~)", re.DOTALL)
    _INLINE_CODE_RE = re.compile(r"`+([^`\n]+?)`+")
    _ESCAPED_CHAR_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>~|])")
    _IMAGE_RE = re.compile(r"!\[([^\]]*)\](?:\([^)]*\)|\[[^\]]*\])")
    _LINK_RE = re.compile(r"\[([^\]]*)\](?:\([^)]*\)|\[[^\]]*\])")
    _AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^<>\s]+)>")
    _HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
    _HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
    _BLOCKQUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
    _TASK_LIST_MARKER_RE = re.compile(
        r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+\[[ xX]\][ \t]+", re.MULTILINE
    )
    _LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+", re.MULTILINE)
    _EMPTY_LINE_MARKER_RE = re.compile(
        r"^[ \t]{0,3}(?:=+|-{2,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE
    )
    _TABLE_DIVIDER_RE = re.compile(r"^[ \t]*\|[ \t:|-]*\|[ \t]*$", re.MULTILINE)
    _TABLE_PIPE_RE = re.compile(r"[ \t]*\|[ \t]*")
    _STRIKETHROUGH_RE = re.compile(r"~~(?P<text>[^\n]+?)~~")
    _ASTERISK_EMPHASIS_RE = re.compile(
        r"(?P<marker>\*{1,3})(?P<text>[^\n]*?\S)(?P=marker)"
    )
    _UNDERSCORE_EMPHASIS_RE = re.compile(
        r"(?<![\w~_])(?P<marker>_{1,3})(?=\S)(?P<text>[^\n]*?\S)(?P=marker)(?![\w~_])"
    )
    _LEADING_SPACES_RE = re.compile(r"^[ \t]+", re.MULTILINE)
    _TRAILING_SPACES_RE = re.compile(r"[ \t]+\n")
    _EMPTY_LINES_RE = re.compile(r"\n{3,}")

    def __init__(self, filepath: str, options: dict = None) -> None:
        """
        :param filepath: Path to markdown file
        :param options: Reader options. If `strip_markdown` is set to True,
        then markdown markup is removed from the loaded text.
        """
        TXTLoader.__init__(self, filepath=filepath, options=options)
        self._strip_markdown = bool(self.options.get(self.STRIP_MARKDOWN_OPTION, False))

    def load(self) -> list:
        """
        Load markdown file. Content is read the same way as in TXTLoader,
        then (optionally) markdown markup is removed and every page is
        marked with the markdown document format.
        :return: List of loaded document pages
        """
        self.doc_pages = TXTLoader.load(self)

        for doc_page in self.doc_pages:
            if self._strip_markdown:
                doc_page.page_content = self.strip_markdown(doc_page.page_content)
            doc_page.metadata["document_format"] = self.DOCUMENT_FORMAT
        return self.doc_pages

    def strip_markdown(self, text: str) -> str:
        """
        Remove markdown markup from the given text. Text of headings, links,
        list items and code is kept, only the markup itself is removed.
        :param text: Text with markdown markup
        :return: Text without markdown markup
        """
        if not text:
            return text

        text = self._FRONT_MATTER_RE.sub("", text)

        placeholders = []

        def _keep_code(match: re.Match) -> str:
            placeholders.append(match.group(1))
            return self._CODE_PLACEHOLDER.format(index=len(placeholders) - 1)

        def _keep_escaped_char(match: re.Match) -> str:
            placeholders.append(match.group(1))
            return self._CODE_PLACEHOLDER.format(index=len(placeholders) - 1)

        text = self._FENCED_CODE_RE.sub(_keep_code, text)
        text = self._ESCAPED_CHAR_RE.sub(_keep_escaped_char, text)
        text = self._INLINE_CODE_RE.sub(_keep_code, text)

        text = self._IMAGE_RE.sub(lambda match: match.group(1), text)
        text = self._LINK_RE.sub(lambda match: match.group(1), text)
        text = self._AUTOLINK_RE.sub(lambda match: match.group(1), text)
        text = self._HTML_TAG_RE.sub("", text)

        text = self._HEADING_RE.sub("", text)
        text = self._BLOCKQUOTE_RE.sub("", text)
        text = self._TASK_LIST_MARKER_RE.sub("", text)
        text = self._LIST_MARKER_RE.sub("", text)
        text = self._EMPTY_LINE_MARKER_RE.sub("", text)

        text = self._STRIKETHROUGH_RE.sub(lambda match: match.group("text"), text)
        text = self._ASTERISK_EMPHASIS_RE.sub(lambda match: match.group("text"), text)
        text = self._UNDERSCORE_EMPHASIS_RE.sub(lambda match: match.group("text"), text)

        text = self._TABLE_DIVIDER_RE.sub("", text)
        text = self._TABLE_PIPE_RE.sub(" ", text)

        text = self._LEADING_SPACES_RE.sub("", text)
        text = self._TRAILING_SPACES_RE.sub("\n", text)
        text = self._EMPTY_LINES_RE.sub("\n\n", text)
        text = text.strip()

        for index, placeholder_value in reversed(list(enumerate(placeholders))):
            text = text.replace(
                self._CODE_PLACEHOLDER.format(index=index), placeholder_value
            )
        return text
