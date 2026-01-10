import abc


class BaseMapper(abc.ABC):
    def __init__(self):
        self._obj1 = None
        self._obj2 = None
        self._mapping = None

    @abc.abstractmethod
    def apply(self, obj1, obj2) -> list:
        """
        Abstract method to apply mapping obj1 on obj2
        :param obj1: Object to map
        :param obj2: On this object obj1 will be mapped
        :return: List (self._mapping) of mapping
        """
        pass


class RawSentencesToTextMapper(BaseMapper):
    ESTIMATED_POSITION_LEN_CHARS_BIAS = 4

    def __init__(
        self,
        simple: bool = True,
        lower: bool = True,
        remove_dot_when_mapping: bool = False,
        exact_sentence_match: bool = True,
    ):
        """
        Actually only simple mapping is available. Simple means, that only first
        and the last sentence words are used to determine sentence position into
        the whole text, text and the sentences are also lowered.
        Mapping objects:
          * self._obj1 have to be set as the list of raw sentences
          * self._obj2 have to be set as the whole raw text
        :param simple - if set to True, then simple mapping will be used
        :param lower - if set to True, then compared texts will be converted to lower
        :param remove_dot_when_mapping - In case when mapped sentence is not found
        then mapping without dot at sentence end will be done
        :param exact_sentence_match - If set to True, then exact sentence
        will be match instead of heuristic
        :exception - Raise Exception when not simple method will be used.
        """
        if not simple:
            raise Exception("Only simple mapping is supporting now!")
        super().__init__()
        self._lower = lower
        self._simple_map = simple
        self._exact_sentence_match = exact_sentence_match
        self._remove_dot_when_mapping = remove_dot_when_mapping

    def apply(self, sentences: list, text: str):
        self._obj1 = sentences
        self._obj2 = text
        self._mapping = []

        end_position = 0
        for s in sentences:
            begin_position, end_position = self.map_sentence_to_text(
                s, text, end_position
            )
            self._mapping.append([begin_position, end_position])
        return self._mapping

    def map_sentence_to_text(
        self, sentence: str, text: str, begin_position: int
    ) -> (int, int):
        """
        Map the single sentence on the given text, mapping procedure starts
        from given begin_position. If begin_position is smaller than 0,
        then mapping will not be done (assert).

        :param sentence:
        :param text:
        :param begin_position:
        :return:
        """
        assert begin_position >= 0, "Begin position cannot be smaller than 0!"
        if self._exact_sentence_match:
            return self.__exact_map_sentence_to_text(
                sentence=sentence, text=text, begin_position=begin_position
            )

        _epos = (
            begin_position + len(sentence) + self.ESTIMATED_POSITION_LEN_CHARS_BIAS
        )
        if _epos > len(text):
            _epos = len(text)

        begin_position -= 1
        if begin_position < 0:
            begin_position = 0

        context = text[begin_position:_epos]
        if self._lower:
            context = context.lower()
            sentence = sentence.lower()

        spl_sent = sentence.split()
        w_first, w_last = spl_sent[0], spl_sent[-1]
        b_pos = self.__find_word_position(w_first, context, 0, None)
        e_pos = self.__find_word_position(
            word=w_last,
            context=context,
            start_position=b_pos,
            estimated_len=len(sentence),
        )

        if e_pos == -1 and self._remove_dot_when_mapping:
            if w_last[-1] == ".":
                w_last = w_last[:-1]
                e_pos = self.__find_word_position(
                    word=w_last,
                    context=context,
                    start_position=b_pos,
                    estimated_len=len(sentence),
                )
        return begin_position + b_pos, e_pos + begin_position

    def __exact_map_sentence_to_text(
        self, sentence: str, text: str, begin_position: int
    ) -> (int, int):
        """
        Map the single sentence on the given text, mapping procedure starts
        from given begin_position. If begin_position is smaller than 0,
        then mapping will not be done (assert).

        :param sentence:
        :param text:
        :param begin_position:
        :return:
        """
        b_position = begin_position - 5
        if b_position < 0:
            b_position = 0
        find_sentence = sentence
        context = text[b_position:]
        if self._lower:
            context = context.lower()
            find_sentence = find_sentence.lower()
        find_position = context.lower().find(find_sentence.lower())
        if find_position < 0:
            return 0, len(sentence)
        start_position = b_position + find_position
        end_position = start_position + len(sentence)
        return start_position, end_position

    def __find_word_position(
        self,
        word: str,
        context: str,
        start_position: int,
        estimated_len: int = None,
    ):
        if estimated_len is None:
            return context[start_position:].find(word)

        w_pos = context.rfind(word)
        w_sent_len = (w_pos - start_position) + len(word)
        if w_sent_len < 0:
            return -1

        if abs(estimated_len - w_sent_len) <= self.ESTIMATED_POSITION_LEN_CHARS_BIAS:
            return w_pos + len(word)
        return -1
