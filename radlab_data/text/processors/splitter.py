import spacy

nlp_spacy = spacy.load("pl_core_news_lg")


class SentenceSplitter:
    min_sentence_len = 1

    def split(
        self,
        text: str,
        return_stats_dict: bool = False,
        split_words_with_ws: bool = True,
    ):
        """
        Split text to sentences and optionally return statistics

        :param text: Text given as string to split to sentences
        :param return_stats_dict: If set, then dictionary with stats will be returned
        :param split_words_with_ws: If set, then number of words will be calculated
         using standard str.split() method
        :return: List of sentences (+ optionally statistics dict)
        """
        doc = nlp_spacy(text)

        sentences = []
        text_stats = {
            "number_of_sentences": 0,
            "number_of_words": 0,
            "number_of_characters": len(text),
        }

        for s in doc.sents:
            if return_stats_dict:
                text_stats["number_of_sentences"] += 1
                if split_words_with_ws:
                    text_stats["number_of_words"] += len(s.text.strip().split())
                else:
                    text_stats["number_of_words"] += len(s)

            sentence = s.text.strip()
            if len(sentence.split()) >= self.min_sentence_len:
                sentences.append(sentence)
        return (sentences, text_stats) if return_stats_dict else sentences
