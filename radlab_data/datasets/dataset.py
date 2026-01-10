import inspect
import os
import sys
from abc import ABC, abstractmethod

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


class DatasetBaseClass(ABC, Dataset):
    BATCHED_TOKENIZATION = True
    MAPPING_ENABLED = True

    def __init__(
        self,
        data,
        metadata,
        cached_dataset_path,
        tokenizer,
        tokenizer_kwargs=None,
        class_label_field="label",
        split_train_valid_test=None,
        pre_shuffle=False,
    ) -> None:
        self._random_seed = 1337
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        self.tokenizer_kwargs = tokenizer_kwargs

        self._raw_data = data
        self._metadata = metadata
        self.class_label_field = class_label_field
        self._label_data = self._metadata.get("labels", None)
        self._cached_dataset_path = cached_dataset_path
        self._tokenizer: PreTrainedTokenizer = tokenizer
        if (
            "special_token_info" in self._metadata
            and self._metadata["special_token_info"] is not None
        ):
            self._tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": self._metadata[
                        "special_token_info"
                    ]["special_tokens"]
                }
            )

        if self._label_data is not None:
            self.labels = self._label_data["names"]
            self.id2label = {
                int(k): v for k, v in self._label_data["id2label"].items()
            }
            self.label2id = self._label_data["label2id"]
            self.num_labels = self._label_data["num_labels"]

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None
        self._split_train_valid_test = split_train_valid_test
        self._pre_shuffle = pre_shuffle

        self._load()

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def _load(self):
        if os.path.exists(self._cached_dataset_path):
            hf_dataset = Dataset.from_file(self._cached_dataset_path)
        else:
            for idx, elem in tqdm(
                enumerate(self._raw_data), desc="Preprocessing raw data"
            ):
                self._preprocess_raw_data(idx, elem)

            self._raw_data = self._apply_custom_preprocess_function(self._raw_data)

            hf_dataset = self._transform_to_hf_dataset(self._raw_data)

            if self.MAPPING_ENABLED:
                hf_dataset = hf_dataset.map(
                    self._mapping_function,
                    batched=self.BATCHED_TOKENIZATION,
                    cache_file_name=self._cached_dataset_path,
                    remove_columns=hf_dataset.column_names,
                )

        if self._pre_shuffle:
            hf_dataset = hf_dataset.shuffle(seed=self._random_seed)

        self._train_dataset = hf_dataset
        self._valid_dataset = None
        self._test_dataset = None

        if self._split_train_valid_test is not None:
            train_split = self._split_train_valid_test[0]
            valid_split = self._split_train_valid_test[1]
            train_split_idx = int(len(hf_dataset) * train_split)
            valid_split_idx = int(len(hf_dataset) * (train_split + valid_split))
            self._train_dataset = Dataset.from_dict(hf_dataset[:train_split_idx])
            self._valid_dataset = Dataset.from_dict(
                hf_dataset[train_split_idx:valid_split_idx]
            )
            if len(self._split_train_valid_test) == 3:
                self._test_dataset = Dataset.from_dict(hf_dataset[valid_split_idx:])

    def _transform_to_hf_dataset(self, dataset) -> Dataset:
        texts = []
        labels = []
        for item in dataset:
            texts.append(item["text"])
            cl_label = item["label"]
            labels.append(cl_label)
        return Dataset.from_dict({"text": texts, "label": labels})

    @abstractmethod
    def _mapping_function(self, batch):
        pass

    @abstractmethod
    def _preprocess_raw_data(self, idx, elem):
        """
        Raw data preprocessing, element by element.
        Prepares data for further usage and conversion to datasets.Dataset object.
        """
        pass

    def _apply_custom_preprocess_function(self, raw_data):
        """
        It is highly recommended to use _preprocess_raw_data method instead (if possible)!

        This method allows to apply a custom function to a dataset after element-wise preprocessing,
        but prior to transforming the dataset into a huggingface dataset.
        """
        return raw_data

    NOT_IMPLEMENTED_MESSAGE = "Use dataset.train_dataset or dataset.test_dataset."

    def __len__(self):
        raise NotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)

    def __getitem__(self, index):
        raise NotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)


class SequenceLabellingDataset(DatasetBaseClass):
    def _preprocess_raw_data(self, idx, elem) -> None:
        self._raw_data[idx] = {
            "text": elem["text"],
            self.class_label_field: elem["metadata"][self.class_label_field],
        }

    def _mapping_function(self, batch):
        tokenized_texts = self._tokenizer(batch["text"], **self.tokenizer_kwargs)
        labels = batch[self.class_label_field]

        # generate ner_tags lists
        all_ner_tags = []
        outside_label = self.label2id["O"]
        for text_idx, (t_text, label_list) in enumerate(
            zip(tokenized_texts["input_ids"], labels)
        ):
            ner_tags = np.full(len(t_text), outside_label)
            #     # print(100 * "-")
            #     # print("text_id", text_idx)
            #     # print("label_list", label_list)
            #     # print("t_text", t_text)
            #     # print(batch["text"][text_idx])
            #     for begin, end, b_label, i_label in label_list:
            #         # subtract 1 because end is the index of a character succeeding the labelled sequence
            #         end -= 1
            #
            #         begin_token_idx = tokenized_texts.char_to_token(text_idx, begin)
            #         end_token_idx = tokenized_texts.char_to_token(text_idx, end)
            #         # print("begin=", begin, "end=", end, "b_label=", b_label, "i_label=", i_label)
            #         # print("begin_token_idx=", begin_token_idx, "end_token_idx=", end_token_idx)
            #         # in case the tokenizer truncates text
            #         # if begin_token_idx is None or end_token_idx is None:
            #         # 	continue
            #         if begin_token_idx is None:
            #             continue
            #         if end_token_idx is None:
            #             end_token_idx = len(batch["text"][text_idx])
            #         if begin_token_idx < end_token_idx:
            #             ner_tags[begin_token_idx + 1 : end_token_idx + 1] = i_label
            #
            #         # print("begin=", begin, "end=", end, "b_label=", b_label, "i_label=", i_label)
            #         # print(
            #         # 	"begin_token_idx=", begin_token_idx, type(begin_token_idx),
            #         # 	"end_token_idx=", end_token_idx
            #         # )
            #         ner_tags[begin_token_idx] = b_label
            #     all_ner_tags.append(list(ner_tags))
            #     # print(ner_tags)
            # tokenized_texts["labels"] = all_ner_tags
            # # print("all_ner_tags", all_ner_tags)
            # # print(100 * "#")
            for begin, end, b_label, i_label in label_list:
                end -= 1
                begin_token_idx = tokenized_texts.char_to_token(text_idx, begin)
                end_token_idx = tokenized_texts.char_to_token(text_idx, end)

                # in case a whitespace character is selected at begin or end idx
                while begin_token_idx is None and begin <= end:
                    begin += 1
                    begin_token_idx = tokenized_texts.char_to_token(text_idx, begin)

                while end_token_idx is None and begin <= end:
                    end -= 1
                    end_token_idx = tokenized_texts.char_to_token(text_idx, end)

                if begin_token_idx is None or end_token_idx is None or begin > end:
                    continue

                if begin_token_idx < end_token_idx:
                    ner_tags[begin_token_idx + 1 : end_token_idx + 1] = i_label
                ner_tags[begin_token_idx] = b_label

            all_ner_tags.append(list(ner_tags))

        tokenized_texts["labels"] = all_ner_tags
        return tokenized_texts


class TextClassificationDataset(DatasetBaseClass):
    def _preprocess_raw_data(self, idx, elem):
        self._raw_data[idx] = {
            "text": elem["text"],
            self.class_label_field: elem["metadata"][self.class_label_field][0],
        }

    def _mapping_function(self, batch):
        tokenized_batch = self._tokenizer(batch["text"], **self.tokenizer_kwargs)
        tokenized_batch["labels"] = batch[self.class_label_field]
        return tokenized_batch


class SequenceToSequenceDataset(DatasetBaseClass):
    def _preprocess_raw_data(self, idx, elem):
        left_ctx = None
        right_ctx = None
        if "left_ctx" in elem["metadata"]:
            left_ctx = elem["metadata"]["left_ctx"]
        if "right_ctx" in elem["metadata"]:
            right_ctx = elem["metadata"]["right_ctx"]

        text = f"{left_ctx + ' ' if left_ctx is not None else ''}{elem['text']}{' ' + right_ctx if right_ctx is not None else ''}"
        self._raw_data[idx] = {
            "text": text,
            self.class_label_field: elem[self.class_label_field][0],
        }

    def _mapping_function(self, batch):
        tokenized_batch = self._tokenizer(
            text=batch["text"],
            text_target=batch[self.class_label_field],
            is_split_into_words=False,
            **self.tokenizer_kwargs,
        )
        return tokenized_batch


class SemanticSimilarityDataset(DatasetBaseClass):
    # MAPPING_ENABLED = False

    def _preprocess_raw_data(self, idx, elem):
        self._raw_data[idx] = {
            "text": elem["text"],
            "label": elem["labels"],
            "score": elem["metadata"]["score"],
            "relation": elem["metadata"]["relation"],
        }

    def _apply_custom_preprocess_function(self, raw_data):
        positive_raw_data = [x for x in raw_data if x["relation"] == "positive"]
        negative_unlabelled_raw_data = [
            x for x in raw_data if x["relation"] == "negative" and not x["label"]
        ]
        # TODO move these variables outside
        negative_to_positive_proportion = 2
        first_method_weight = 0.8
        second_method_weight = 1 - first_method_weight
        score_noise_std = 0.001

        # adjust if there's not enough data in the unlabelled negative subset
        positive_raw_data_len = len(positive_raw_data)
        max_possible_second_method_weight = min(
            (len(negative_unlabelled_raw_data) / positive_raw_data_len), 1
        )
        second_method_weight = min(
            max_possible_second_method_weight, second_method_weight
        )
        first_method_weight = 1 - second_method_weight

        # 1. Generation of negative examples by mixing positives.
        negative_count = int(
            positive_raw_data_len
            * negative_to_positive_proportion
            * first_method_weight
        )
        pair_indexes = np.random.choice(
            negative_count, negative_count, replace=False
        )  # we're doing it this way to maximize randomness, overflowing indices will sadly be repeats, but in case of no overflow, there will be no repetitions
        pair_indexes = pair_indexes % len(
            positive_raw_data
        )  # ensure indices are in range of len(positive_raw_data)

        for i in tqdm(
            range(negative_count),
            desc="Generating negative examples by mixing positives.",
        ):
            idx = i % positive_raw_data_len
            pair_idx = pair_indexes[idx]
            elem = positive_raw_data[idx]
            pair_elem = positive_raw_data[pair_idx]
            new_elem = {
                "text": elem["text"],
                "label": pair_elem["label"],
                "score": 0,
                "relation": "negative",
            }
            raw_data.append(new_elem)

        # 2. Generating negative examples by attaching positive labels to unlabelled negative texts.
        negative_count = int(
            negative_count / first_method_weight * second_method_weight
        )  # inversion of the count
        pair_indexes = np.random.choice(
            len(positive_raw_data), negative_count, replace=True
        )
        negative_indexes = np.random.choice(
            len(negative_unlabelled_raw_data),
            negative_count,
            replace=True,  # TODO make sure True is correct
        )  # subset of negatives
        for i in tqdm(
            range(negative_count),
            desc="Generating negative examples by attaching positive labels to unlabelled negative texts.",
        ):
            negative_elem = negative_unlabelled_raw_data[negative_indexes[i]]
            negative_elem["label"] = positive_raw_data[pair_indexes[i]]["label"]
            # no need to append to raw_data, because we are overwriting 'label' of negative_elem by reference

        # remove unlabelled data
        raw_data = [x for x in raw_data if x["label"]]

        # expand along the label dimension
        tmp_data = []
        for elem in raw_data:
            for label in elem["label"]:
                tmp_elem = elem.copy()
                tmp_elem["label"] = label
                tmp_data.append(tmp_elem)
        raw_data = tmp_data

        # add noise to score to mimic a real dataset
        for elem in raw_data:
            if elem["relation"] == "positive":
                elem["score"] -= abs(np.random.normal(0, score_noise_std))
            else:
                elem["score"] += abs(np.random.normal(0, score_noise_std))

        return raw_data

    def _mapping_function(self, batch):
        # tokenized_text = self._tokenizer(batch["text"], **self.tokenizer_kwargs)[
        #     "input_ids"
        # ]
        # tokenized_label = self._tokenizer(batch["label"], **self.tokenizer_kwargs)[
        #     "input_ids"
        # ]
        # return {
        #     "text": tokenized_text,
        #     "label": tokenized_label,
        #     "score": batch["score"],
        #     "relation": batch["relation"],
        # }
        return {
            "sentence1": batch["text"],
            "sentence2": batch["label"],
            "score": batch["score"],
        }

    def _transform_to_hf_dataset(self, dataset) -> Dataset:
        texts = []
        labels = []
        scores = []
        relations = []
        for item in dataset:
            texts.append(item["text"])
            cl_label = item["label"]
            labels.append(cl_label)
            scores.append(item["score"])
            relations.append(item["relation"])

        return Dataset.from_dict(
            {"text": texts, "label": labels, "score": scores, "relation": relations}
        )


class DPODataset(DatasetBaseClass):
    MAPPING_ENABLED = False  # this type of dataset is not preprocessed (does not utilize preprocessing.dataset.Dataset), not tokenized therefore does not require caching

    def _preprocess_raw_data(self, idx, elem):
        data_dict = {
            "prompt": elem["prompt"],
            "chosen": elem["chosen"],
            "rejected": elem["rejected"],
        }
        self._raw_data[idx] = data_dict

    def _mapping_function(self, element):
        return element

    def _transform_to_hf_dataset(self, dataset) -> Dataset:
        prompt = []
        chosen = []
        rejected = []
        for item in dataset:
            prompt.append(item["prompt"])
            chosen.append(item["chosen"])
            rejected.append(item["rejected"])
        return Dataset.from_dict(
            {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        )


class DialogueAlternativesDataset(DatasetBaseClass):
    BATCHED_TOKENIZATION = False

    def _preprocess_raw_data(self, idx, elem):
        data_dict = {
            "shared_part": elem["dialogue"],
            "endings": elem["endings"],
            "scores": elem["metadata"]["ending_scores"],
        }
        self._raw_data[idx] = data_dict

    def _mapping_function(self, element):
        tokenized_endings = {"endings": []}
        shared_part = element["shared_part"]
        for ending in element["endings"]:
            # TODO optimize (batched processing) like here https://github.com/imoneoi/openchat/blob/master/ochat/data/generate_dataset.py
            # TODO invent a way to tokenize the context only once and share it via reference (which is the dialogue before an ending)
            dialogue = shared_part + [ending]
            templated_dialogue: str = self._tokenizer.apply_chat_template(
                dialogue, tokenize=False
            )
            tokenized_dialogue = self._tokenizer(
                templated_dialogue, **self.tokenizer_kwargs
            )
            del tokenized_dialogue["attention_mask"]
            tokenized_endings["endings"].append(tokenized_dialogue)

        shared_part_end_idx = 0
        dialogue_input_ids = [
            dialogue["input_ids"] for dialogue in tokenized_endings["endings"]
        ]
        for matching_tokens in zip(*dialogue_input_ids):
            if len(set(matching_tokens)) == 1:
                shared_part_end_idx += 1
            else:
                break

        shared_part = tokenized_endings["endings"][0]["input_ids"][
            :shared_part_end_idx
        ]
        for dialogue in tokenized_endings["endings"]:
            dialogue["input_ids"] = dialogue["input_ids"][shared_part_end_idx:]
        tokenized_endings["shared_part"] = shared_part
        tokenized_endings["scores"] = element["scores"]
        return tokenized_endings

    def _transform_to_hf_dataset(self, dataset) -> Dataset:
        shared_part = []
        scores = []
        endings = []
        for item in dataset:
            shared_part.append(item["shared_part"])
            scores.append(item["scores"])
            endings.append(item["endings"])
        return Dataset.from_dict(
            {"shared_part": shared_part, "scores": scores, "endings": endings}
        )


cached_classes_and_names = None


def get_class_from_type_name(type_name: str) -> DatasetBaseClass:
    global cached_classes_and_names
    if cached_classes_and_names is None:
        cached_classes_and_names = {
            k: v
            for k, v in inspect.getmembers(sys.modules[__name__])
            if inspect.isclass(v) and issubclass(v, DatasetBaseClass)
        }
    if type_name in cached_classes_and_names:
        return cached_classes_and_names[type_name]
    else:
        raise KeyError(f"Dataset class of type: {type_name} does not exist.")
