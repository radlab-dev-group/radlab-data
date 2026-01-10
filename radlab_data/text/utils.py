from radlab_data.text.processors.mappers import RawSentencesToTextMapper


class TextUtils:
    """
    TextUtils like sentences to text position mapping.
    """

    def __int__(self):
        pass

    @staticmethod
    def map_sentences_to_text(
        sentences,
        text,
        remove_dot_when_mapping: bool = False,
        exact_sentence_match: bool = True,
    ):
        """
        Map the sentences to the text positions. Each sentence is described
        as begin and end position. As the sentence mapper is used
        `RawSentencesToTextMapper` from `radlab_data.text.processors.mappers`
        :param sentences: List of the raw sentences to map
        :param text: Raw text on which the sentences will be mapped
        :param remove_dot_when_mapping: When sentence mapping is not found then
        additionally will be done mapping without dot at the sentence end
        :param exact_sentence_match: Exact sentence on text mapping
        :return:
        """
        return RawSentencesToTextMapper(
            remove_dot_when_mapping=remove_dot_when_mapping,
            exact_sentence_match=exact_sentence_match,
        ).apply(sentences, text)

    @staticmethod
    def text_language(text_str: str, probs: bool = False):
        from ftlangdetect import detect

        text_str = text_str.replace("\n", " ")
        result = detect(text=text_str, low_memory=False)
        return result if probs else result["lang"]

    @staticmethod
    def translate_text_deepl(text_str: str, target_lang: str, auth_key: str) -> str:
        """
        Translates text into the target language.
        Target must be an ISO 639-1 language code.
        See https://g.co/cloud/translate/v2/translate-reference#supported_languages
        """
        import deepl

        translator = deepl.Translator(auth_key)
        result = translator.translate_text(text_str, target_lang=target_lang.upper())
        return result.text
