import re
import abc

from radlab_data.text.processors.base import TextProcessor
from radlab_data.text.processors.regex_defs import (
    T_REPL,
    REPL_W,
    GENERAL_CLEANERS_REGEX,
    ABBREV_REPLACE_MAP,
    ADD_NEW_LINE_IN_CASE,
    SENTENCE_END_CHARS,
    ENUMERATION_REGEX,
)


class TextCleaner(TextProcessor, abc.ABC):
    """
    Base class for all text cleaners
    """

    def __init__(self):
        super().__init__()


class Strip(TextProcessor):
    """
    Simple text processor to strip given text
    """

    def __init__(self):
        super().__init__()

    def apply(self, text: str, args: dict = None) -> str:
        """
        Strip text and return new text after strip()
        :param text: Text to strip
        :param args: Additional arguments (not used)
        :return: Text after strip
        """
        return text.strip()


class RemovePhrase(TextCleaner):
    """
    Main class to remove exact match of phrase into given text.
    """

    def __init__(self, phrases):
        super().__init__()
        self.phrases_to_remove = phrases if phrases is not None else []

    def apply(self, text: str, args: dict = None) -> str:
        """
        Remove all `self.phrases_to_remove` from given `text`
        :param text: Text to remove phrases
        :param args: Additional arguments (not used)
        :return: Text without phrases
        """
        text_clear = text
        for ph in self.phrases_to_remove:
            text_clear = text_clear.replace(ph, "")
        return text_clear


class NormalizeAbbreviations(TextCleaner):
    """
    Replace abbreviations with it normalized form, something line replace
    phrase to another phrase.
    """

    def __init__(self, action=None):
        super().__init__()

        self.args = None
        if action is not None:
            self.args = {"action": action}

    def apply(self, text: str, args: dict = None) -> str:
        """
        Abbreviations normalization
        :param text: Text to normalize
        :param args: Additional arguments
        :return:
        """
        if args is None:
            args = self.args

        normalize = True
        if args is not None and "action" in args:
            normalize = "norm" in args["action"].lower()
        clear_text = text
        for ab, abn in ABBREV_REPLACE_MAP:
            if normalize:
                clear_text = clear_text.replace(ab, abn)
            else:
                clear_text = clear_text.replace(abn, ab)
        return clear_text


class GeneralLookImprover(TextCleaner):
    """
    Improve general text look
    """

    def __init__(self):
        super().__init__()

    def apply(self, text: str, args: dict = None) -> str:
        """
        Improve general look of the given `text`
        :param text:
        :param args:
        :return:
        """
        for regex in GENERAL_CLEANERS_REGEX.values():
            text = re.sub(regex.get(T_REPL), regex.get(REPL_W), text)
        return text


class RemoveEnums(TextCleaner):
    """
    Remove enums from text
    """

    def __init__(self):
        super().__init__()

    def apply(self, text: str, args: dict = None) -> str:
        """
        Remove enumerations from the text
        :param text:
        :param args:
        :return:
        """
        for regex in ENUMERATION_REGEX.values():
            text = re.sub(regex.get(T_REPL), regex.get(REPL_W), text)
        return text


class RemoveQuestionMarks(TextCleaner):
    """
    Remove qmarks from text
    """

    def __init__(self):
        super().__init__()

    def apply(self, text: str, args: dict = None) -> str:
        """
        Remove question marks from the text
        :param text:
        :param args:
        :return:
        """
        normalized_text = ""
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 5:
                # n_txt += "\"
                continue
            line_s = line[0:-1].replace("?", "")
            normalized_text += line_s + line[-1] + "\n\n"
        return normalized_text


class ProperLinesConstructor(TextCleaner):
    """
    Prepare proper lines to split text to sentences
    """

    def __init__(self, force_add_dot: bool = False):
        super().__init__()
        self.force_add_dot = force_add_dot

    def apply(self, text: str, args: dict = None) -> str:
        """
        Prepare text with proper lines.
        :param text:
        :param args:
        :return:
        """
        prev_line = ""
        normalized_text = ""
        first_line = True
        for line in text.split("\n"):
            line = line.strip()
            if len(line):
                if first_line:
                    first_line = False
                    normalized_text = self._sentence_upper_start(line)
                else:
                    new_sentence = not len(prev_line)
                    if new_sentence:
                        normalized_text = self._add_dot_at_the_end(
                            normalized_text, self.force_add_dot
                        )
                        normalized_text += "\n\n"
                    normalized_text += " " + self._sentence_upper_start(line)
            prev_line = line
        if self.force_add_dot:
            return self._norm_text(normalized_text)
        return normalized_text

    def _norm_text(self, norm_text: str):
        """
        Little text normalisation
        :param norm_text:
        :return:
        """
        norm_text = norm_text.replace(":-", ": ")
        norm_text = norm_text.replace(": -", ": ")
        norm_text = norm_text.replace(",,", ",")
        norm_text = norm_text.replace(" .", ".")
        norm_text = norm_text.replace("  ", " ")
        norm_text = norm_text.replace("  ", " ")
        return norm_text

    def _sentence_upper_start(self, sentence):
        sentence = sentence.strip()
        n_sent = sentence[0].upper()
        if len(sentence) > 1:
            n_sent += sentence[1:]
        return n_sent

    def _add_dot_at_the_end(self, norm_text_clear, force_add_dot=False):
        """
        :param norm_text_clear:
        :param force_add_dot:
        :return:
        """
        norm_text_clear = norm_text_clear.strip()
        last_char = norm_text_clear[-1]
        if last_char not in SENTENCE_END_CHARS:
            norm_text_clear += "."
        elif force_add_dot:
            if last_char in ":,-":
                norm_text_clear = norm_text_clear[0:-1] + "."
            elif last_char in ')]"':
                norm_text_clear += "."
        return norm_text_clear


class SimpleTextPreprocessor(TextCleaner):
    """
    Prepare text to next steps
    """

    def __init__(self, force_add_dot: bool = None):
        super().__init__()
        self._text = None
        self.force_add_dot = force_add_dot

        self._plc = ProperLinesConstructor(force_add_dot=False)

    def apply(self, text: str, args: dict = None) -> str:
        """
        Preprocess given text
        :param text: Text to prepare
        :param args: Additional arguments
        :return: Prepared text
        """
        self._text = text
        # self._add_new_line_at_start()
        self._text = self._plc.apply(self._text)
        self._ws_clear()
        return self._text

    def _add_new_line_at_start(self) -> str:
        for nline_in_case in ADD_NEW_LINE_IN_CASE.values():
            self._text = re.sub(
                nline_in_case.get(T_REPL), nline_in_case.get(REPL_W), self._text
            )
        self._text = self._text.replace(": -", ":\n-")
        self._text = self._text.replace(":-", ":\n-")
        return self._text

    def _ws_clear(self):
        self._text = self._text.replace("?  ?", "?")
        self._text = self._text.replace("\t", " ")
        self._text = self._text.replace("  ", " ")
        self._text = self._text.replace("  ", " ")
        self._text = self._text.replace("  ", " ")
        self._text = self._text.replace("  ", " ")
        self._text = self._text.replace(": ,", ":")
        self._text = self._text.replace(" ,", ",")
        return self._text


class FullChainProcessor:
    """
    Chain of Text Processors
    """

    def __init__(self, phrases_to_remove):
        self.chain = [
            RemovePhrase(phrases=phrases_to_remove),
            GeneralLookImprover(),
            SimpleTextPreprocessor(),
            NormalizeAbbreviations(action="norma"),
            RemoveEnums(),
            RemoveQuestionMarks(),
            NormalizeAbbreviations(action="restore"),
            ProperLinesConstructor(force_add_dot=False),
        ]

    def apply(self, text: str) -> str:
        """
        Apply whole processors chain
        :return: Processed text
        """
        for processor in self.chain:
            text = processor.apply(text)
        return text


class CleanerChainProcessor:
    """
    Chain of Text Processors
    """

    def __init__(self, phrases_to_remove):
        self.chain = [
            RemovePhrase(phrases=phrases_to_remove),
            GeneralLookImprover(),
            NormalizeAbbreviations(action="norma"),
            RemoveEnums(),
            RemoveQuestionMarks(),
            NormalizeAbbreviations(action="restore"),
        ]

    def apply(self, text: str) -> str:
        """
        Apply whole processors chain
        :return: Processed text
        """
        for processor in self.chain:
            text = processor.apply(text)
        return text
