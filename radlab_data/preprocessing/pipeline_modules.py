import copy
import warnings

import spacy
from radlab_data.preprocessing.dataset import (
    ClassificationDataset,
    Dataset,
    SequenceLabellingDataset,
    SequenceToSequenceDataset,
    TextClassificationDataset,
)
from radlab_data.preprocessing.pipeline import PipelineElementBase


class SplitLabels(PipelineElementBase):
    def __init__(self) -> None:
        super().__init__()

    def _process_data(
        self, dataset: SequenceToSequenceDataset
    ) -> SequenceToSequenceDataset:
        new_data = []
        for elem in dataset._raw_data:
            new_elems = []
            labels = elem[dataset.class_label_field_name]
            if len(labels) > 1:
                for label in labels:
                    new_elem = copy.deepcopy(elem)
                    new_elem[dataset.class_label_field_name] = [label]
                    new_elems.append(new_elem)
            else:
                new_elems.append(elem)
            new_data.extend(new_elems)
        dataset._raw_data = new_data
        return dataset


class RemoveUnlabelledData(PipelineElementBase):
    def __init__(self) -> None:
        super().__init__()

    def _process_data(self, dataset: Dataset) -> Dataset:
        if isinstance(dataset, ClassificationDataset):
            dataset._raw_data = list(
                elem
                for elem in dataset.raw_data
                if elem["metadata"][dataset.class_label_field_name]
            )
            return dataset
        elif isinstance(dataset, SequenceToSequenceDataset):
            dataset._raw_data = list(
                elem
                for elem in dataset.raw_data
                if elem[dataset.class_label_field_name]
            )
            return dataset


class RemoveDuplicates(PipelineElementBase):
    def __init__(self) -> None:
        super().__init__()

    def _process_data(self, dataset: ClassificationDataset) -> ClassificationDataset:
        exists = set()

        def generator():
            removed = 0
            for elem in dataset.raw_data:
                representation_hash = hash(elem["text"])
                if representation_hash not in exists:
                    exists.add(representation_hash)
                    yield elem
                else:
                    removed += 1
            print(f"Removed {removed} duplicates.")

        dataset._raw_data = list(generator())
        return dataset


class SplitSentences(PipelineElementBase):
    ALLOWED_DATASET_TYPES: tuple = (
        SequenceLabellingDataset,
        TextClassificationDataset,
    )

    def __init__(self) -> None:
        self.nlp = spacy.load("pl_core_news_lg")
        self.ACTION_MAPPING: dict = {
            SequenceLabellingDataset: self._action_sequence_labelling_dataset,
            TextClassificationDataset: self._action_text_classification_dataset,
        }
        super().__init__()

    def _process_data(self, dataset: ClassificationDataset) -> ClassificationDataset:
        action = self.ACTION_MAPPING[type(dataset)]
        new_data = []
        for elem in dataset.raw_data:
            output = action(dataset, elem)
            new_data.extend(output)
        dataset.raw_data = new_data
        return dataset

    def _action_sequence_labelling_dataset(self, dataset, element) -> list:
        text = element["text"]
        labels = element["metadata"][dataset.class_label_field_name]
        sentences = self.nlp(text).sents
        output = []

        for sentence in sentences:
            sentence_start_char = sentence.start_char
            sentence_last_char = sentence.end_char - 1
            sentence_labels = []
            sentence_text = sentence.text

            # text stripping
            original_text_len = len(sentence_text)
            sentence_text = sentence_text.lstrip()
            if not sentence_text:
                continue
            chars_lost = original_text_len - len(sentence_text)
            sentence_start_char += chars_lost

            original_text_len = len(sentence_text)
            sentence_text = sentence_text.rstrip()
            if not sentence_text:
                continue
            chars_lost = original_text_len - len(sentence_text)
            sentence_last_char -= chars_lost

            for label in labels:
                label_start_char, label_end_char, *data = label
                label_last_char = label_end_char - 1

                if (
                    label_start_char > sentence_last_char
                    or label_last_char < sentence_start_char
                ):
                    continue
                intersection_start = max(sentence_start_char, label_start_char)
                intersection_end = min(sentence_last_char, label_last_char)

                # shift to local coordinates
                intersection_start -= sentence_start_char
                intersection_end -= sentence_start_char

                new_label_data = [intersection_start, intersection_end + 1, *data]
                sentence_labels.append(new_label_data)

            new_elem = copy.deepcopy(element)
            new_elem["text"] = sentence_text
            new_elem["metadata"][dataset.class_label_field_name] = sentence_labels
            output.append(new_elem)
        return output

    def _action_text_classification_dataset(self, dataset, element) -> list:
        text = element["text"]
        sentences = [x.text.strip() for x in self.nlp(text).sents]
        output = []
        for sent in sentences:
            if not sent:
                # remove empty sentences
                continue
            new_elem = copy.deepcopy(element)
            new_elem["text"] = sent
            output.append(new_elem)
        return output


class RemoveLongIOBAnnotation(PipelineElementBase):
    def __init__(self, ans_factor: float = 0.85) -> None:
        """
        :param ans_factor: Annotation to sentence factor
        """
        super().__init__()
        self.anno_sent_perc_len = ans_factor

    def _process_data(self, dataset: ClassificationDataset) -> ClassificationDataset:
        new_dataset = []
        for elem in dataset.raw_data:
            if not len(elem["metadata"][dataset.class_label_field_name]):
                continue
            new_annos = []
            text_str_len = len(elem["text"])
            for annotation in elem["metadata"][dataset.class_label_field_name]:
                anno_len = annotation[1] - annotation[0]
                if anno_len / text_str_len >= self.anno_sent_perc_len:
                    continue
                new_annos.append(annotation)
            elem["metadata"][dataset.class_label_field_name] = new_annos
            new_dataset.append(elem)
        dataset.raw_data = new_dataset
        return dataset


class AlignAnnotationToWordBoundaries(PipelineElementBase):
    def __init__(self, align_begin: bool = True, align_end: bool = True) -> None:
        """
        :param ans_factor: Annotation to sentence factor
        """
        super().__init__()
        self._align_begin = align_begin
        self._align_end = align_end

    def _process_data(self, dataset: ClassificationDataset) -> ClassificationDataset:
        # No alignment is needed!
        if not self._align_begin and not self._align_end:
            return dataset
        new_dataset = []
        for elem in dataset.raw_data:
            if not len(elem["metadata"][dataset.class_label_field_name]):
                continue
            new_annotations = []
            text_str = elem["text"]
            for annotation in elem["metadata"][dataset.class_label_field_name]:
                # TODO: Tutaj jest sytuacja, że po "odsiewaniu" labelek
                # TODO: zrobione jest obejście na odisanie przykładów, bez labelki
                # TODO: czyli bez kompletu beg_pos, end_pos, atag_b, atag_e
                if len(annotation) < 4:
                    continue
                a_beg_pos, a_end_pos, *data = annotation
                a_last_pos = a_end_pos - 1
                text_len = len(text_str)

                if not (
                    (0 <= a_beg_pos < text_len) and (0 <= a_last_pos < text_len)
                ):
                    warnings.warn(
                        f"Annotation boundaries incorrect ({annotation = }, {text_len = }, {text_str[:64] = }). Skipping annotation."
                    )
                    continue

                if self._align_begin:
                    a_beg_pos = self._align_begin_annotation_position(
                        a_beg_pos, text_str, strategy="most_word"
                    )

                if self._align_end:
                    a_last_pos = self._align_end_annotation_position(
                        a_last_pos, text_str, strategy="most_word"
                    )

                a_end_pos = a_last_pos + 1
                new_annotations.append([a_beg_pos, a_end_pos, *data])
            elem["metadata"][dataset.class_label_field_name] = new_annotations
            new_dataset.append(elem)
        dataset.raw_data = new_dataset
        return dataset

    def _align_begin_annotation_position(
        self, position: int, text_str: str, strategy: str = None
    ):
        if strategy == "most_word":
            return self._align_to_most_word(position, text_str, direction="right")

        for corr_pos in range(position, -1, -1):
            if text_str[corr_pos].isspace():
                return corr_pos
        return 0

    def _align_end_annotation_position(
        self, position: int, text_str: str, strategy: str = None
    ):
        """
        Depends on given strategy annotation will be aligned to mostly annotated
        word on begin/end position of annotation

        :param position:
        :param text_str:
        :param strategy:
        :return:
        """
        if strategy == "most_word":
            return self._align_to_most_word(position, text_str, direction="left")

        for corr_pos in range(position, len(text_str)):
            if text_str[corr_pos].isspace():
                return corr_pos
        return len(text_str) - 1

    def _align_to_most_word(
        self, position: int, text_str: str, direction: str
    ) -> int:
        """
        Align annotation to most annotated part of word.
        Depends on direction (left, right) begin position will be moved to
        begin/end position of word in case when most word will (or will not)
        be annotated.

        :param position: Starting position
        :param text_str: Text to move annotation
        :param direction: l/r if dir
        :return: Position of annotation point
        """

        direction_mapping = {"left": 0, "l": 0, "right": 1, "r": 1}
        if direction.lower() not in direction_mapping.keys():
            raise Exception("Direction have to be one of: {left, right, l, r}")

        assert 0 <= position < len(text_str)
        if text_str[position].isspace() or position == 0:
            return position

        binary_direction = direction_mapping[direction.lower()]
        first_space_left = position
        first_space_right = position

        while not text_str[first_space_left].isspace():
            first_space_left -= 1
            if first_space_left < 0:
                first_space_left = 0
                break
        while not text_str[first_space_right].isspace():
            first_space_right += 1
            if first_space_right >= len(text_str):
                first_space_right = len(text_str) - 1
                break

        chars_to_left = position - first_space_left
        chars_to_right = first_space_right - position

        return_position = 0
        if binary_direction == 0:  # left
            if chars_to_left >= chars_to_right:
                return_position = (
                    first_space_right - 1
                )  # -1/+1 because we want the word before/after space, not exactly the space
            else:
                return_position = first_space_left + 1
        else:  # right
            if chars_to_right >= chars_to_left:
                return_position = first_space_left + 1
            else:
                return_position = first_space_right - 1

        if return_position < 0:
            return 0
        elif return_position >= len(text_str):
            return len(text_str) - 1
        else:
            return return_position
