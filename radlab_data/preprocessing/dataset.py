import itertools
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

import pandas as pd
import sklearn.model_selection
from radlab_data.utils.special_tokens_rules import (
    RuleApplicator,
    SpecialTokensRulesParser,
)

T_dtype = TypeVar("T_dtype")


def json_loader(dataset_path, class_label_field_name=None):
    """
    Json loader
    :param dataset_path:
    :param class_label_field_name:
    :return:
    """
    # TODO: proper class_label_field_name handling (see _jsonl_loader)
    with open(dataset_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def jsonl_loader(dataset_path, class_label_field_name=None):
    """
    Load jsonl file
    :param dataset_path: Path to jsonl file
    :param class_label_field_name: If given then additionally type of class
           label will be checked
    :return:
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        json_list = f.readlines()

    if class_label_field_name is None:
        return [json.loads(json_str) for json_str in json_list]

    all_dataset_elems = []
    for json_str in json_list:
        jdata = json.loads(json_str)
        cl_data = jdata[class_label_field_name]
        if type(cl_data) in [str]:
            cl_label = cl_data
            jdata[class_label_field_name] = [cl_label]
        all_dataset_elems.append(jdata)
    return all_dataset_elems


@dataclass
class SaveConfiguration:
    path: str
    extra_metadata: dict = None
    add_date_created_to_name: bool = True
    save_iob_file: bool = False
    save_text_label_xlsx: bool = False
    iob_window_o_class: int = 3
    show_class_labels_histogram: bool = False


class Dataset(ABC):
    ALLOWED_EXTENSIONS = {".json": json_loader, ".jsonl": jsonl_loader}
    DEFAULT_CLASS_LABEL_FIELD = "label"

    def _load_raw_data(
        self, dataset_path, class_label_field_name, dataset_extension=None
    ):
        is_dir = os.path.isdir(dataset_path)
        if is_dir:
            # if loading dataset from a folder (usually that means multiple
            # files inside, that require merging) the dataset_type must be defined
            assert (
                dataset_extension is not None
            ), "Specify dataset_extension when loading a directory."
        if dataset_extension is None:
            # extension will define the dataset_type
            _, dataset_extension = os.path.splitext(dataset_path)
        assert dataset_extension in self.ALLOWED_EXTENSIONS, (
            f"{dataset_extension =} is not supported. Allowed extensions: "
            f"{list(self.ALLOWED_EXTENSIONS.keys())}"
        )

        loader_class_label = None
        if class_label_field_name != self.DEFAULT_CLASS_LABEL_FIELD:
            loader_class_label = class_label_field_name

        self.class_label_field_name = class_label_field_name

        data_loader_function = self.ALLOWED_EXTENSIONS[dataset_extension]
        if is_dir:
            files = os.listdir(dataset_path)
            all_data = []
            for filename in files:
                file_path = os.path.join(dataset_path, filename)
                data = data_loader_function(file_path, loader_class_label)
                if data:
                    all_data.append(data)
            self._raw_data = list(itertools.chain.from_iterable(all_data))
        else:
            self._raw_data = data_loader_function(dataset_path, loader_class_label)

    def _generate_date_created(self):
        return str(datetime.now()).replace(" ", "_")


class ClassificationDataset(Dataset, ABC):
    """
    Constructor parameters:
        dataset_path: str - path to the dataset, can be file or folder
        dataset_extension: str = None - type of the dataset, see self.ALLOWED_TYPES
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_extension: str = None,
        class_label_field_name: str = Dataset.DEFAULT_CLASS_LABEL_FIELD,
        class_labels_mapping_file: str = None,
        ignore_labels_not_mapped: bool = True,
    ) -> None:
        """

        :param dataset_path:
        :param dataset_extension:
        :param class_label_field_name:
        :param class_labels_mapping_file: If given, then
        class labels mapping will be done
        :param ignore_labels_not_mapped: If mapping is enabled,
        ignore these class labels, which are not mapped
        """
        self.date_created = self._generate_date_created()
        self._load_raw_data(dataset_path, class_label_field_name, dataset_extension)

        self._class_labels_mapping = {}
        self._ignore_labels_not_mapped = ignore_labels_not_mapped
        if class_labels_mapping_file is not None:
            self._prepare_class_labels_mapping(class_labels_mapping_file)

        self._random_state = None
        self._is_processed = False
        self._process_raw_data()
        self._label_histogram = self._get_label_histogram(self._raw_data)
        self._map_labels()

    def _get_mapped_label(self, label_str: str) -> str:
        main_label = label_str
        main_label = self._prepare_single_label(label_str=main_label)
        main_label = self._class_labels_mapping.get(main_label, None)
        if main_label is None and not self._ignore_labels_not_mapped:
            main_label = label_str
        return main_label

    @property
    def is_processed(self):
        return self._is_processed

    def __len__(self):
        return len(self._raw_data)

    def __getitem__(self, index):
        return self._raw_data[index]

    def _process_raw_data(self):
        assert not self._is_processed
        self._raw_data = [
            {
                "text": x["text"],
                "metadata": {k: v for k, v in x.items() if k != "text"},
            }
            for x in self._raw_data
        ]
        self.labels = list(sorted(self._get_labels(self._raw_data)))
        self.id2label = dict(enumerate(self.labels))
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.labels)
        self._is_processed = True

    @staticmethod
    def _prepare_single_label(label_str):
        return (
            label_str.replace('\\"', "")
            .replace('"', "")
            .replace("'", "")
            .replace(" ", "_")
            .replace("\t", "_")
        )

    def _prepare_class_labels_mapping(self, csv_mapping_filepath):
        df = pd.read_csv(csv_mapping_filepath, sep="\t")
        for idx, row in df.iterrows():
            self._class_labels_mapping[
                self._prepare_single_label(row["source_label"])
            ] = self._prepare_single_label(row["target_label"])

        for k, v in self._class_labels_mapping.items():
            print(k, "=====>", v)

    def get_save_dir(self, path, add_date_created_to_name: bool = True) -> str:
        if add_date_created_to_name:
            path = path + "_" + self.date_created

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def save(
        self,
        save_config: SaveConfiguration,
    ) -> str:
        path = self.get_save_dir(
            save_config.path, save_config.add_date_created_to_name
        )

        metadata_to_save = {
            "dataset_type": self.__class__.__name__,
            "date_created": self.date_created,
            "labels": {
                "names": self.labels,
                "id2label": self.id2label,
                "label2id": self.label2id,
                "num_labels": self.num_labels,
            },
        }

        if save_config.extra_metadata is not None:
            metadata_to_save.update(save_config.extra_metadata)

        if self._class_labels_mapping:
            metadata_to_save["labels"]["labels_mapping"] = self._class_labels_mapping

        data_to_save = self._raw_data
        data_file_path = os.path.join(path, "data.json")
        with open(data_file_path, mode="w", encoding="utf-8") as f:
            json.dump(obj=data_to_save, fp=f, indent=4, ensure_ascii=False)

        metadata_file_path = os.path.join(path, "metadata.json")
        with open(metadata_file_path, mode="w", encoding="utf-8") as f:
            json.dump(obj=metadata_to_save, fp=f, indent=4, ensure_ascii=False)
        return path

    @abstractmethod
    def _get_label_distribution(self, data, dtype: T_dtype) -> T_dtype:
        """
        This method returns labels distribution of dtype (can be list or set for example).
        """
        pass

    @abstractmethod
    def _get_labels(self, data) -> set:
        """
        This method should utilize the _get_label_distribution.
        """
        pass

    @abstractmethod
    def _map_labels(self):
        pass

    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        self._raw_data = value

    def _get_label_histogram(self, data):
        all_labels = self._get_label_distribution(data, dtype=list)
        if not isinstance(all_labels[0], str):
            raise TypeError(
                "Histogram should be generated before mapping labels to types other than str."
            )
        hist = defaultdict(int)
        for label in all_labels:
            hist[label] += 1
        return hist


class SequenceToSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        specials_tokens_file_path: str = None,
        dataset_extension: str = None,
        class_label_field_name: str = Dataset.DEFAULT_CLASS_LABEL_FIELD,
    ) -> None:
        self.date_created = self._generate_date_created()
        self._load_raw_data(dataset_path, class_label_field_name, dataset_extension)

        self.special_token_info = None
        self.special_tokens = None
        self.special_tokens_rules = None
        if specials_tokens_file_path is not None:
            self.special_token_info = self._load_special_token_info(
                specials_tokens_file_path
            )
            self.special_tokens = self.special_token_info["special_tokens"]
            self.special_tokens_rules = self.special_token_info[
                "special_token_rules"
            ]

        self._random_state = None
        self._is_processed = False

        if specials_tokens_file_path is not None:
            self._special_tokens_rules_applicator = self._parse_special_token_rules()
            self._add_special_tokens_to_data()

    @property
    def is_processed(self):
        return self._is_processed

    def __len__(self):
        return len(self._raw_data)

    def __getitem__(self, index):
        return self._raw_data[index]

    def _load_special_token_info(self, specials_tokens_file_path):
        return json_loader(specials_tokens_file_path)

    def _parse_special_token_rules(self) -> RuleApplicator:
        parser = SpecialTokensRulesParser(self.special_tokens)
        applicator_trees = parser.parse(self.special_tokens_rules)
        applicator = RuleApplicator(
            rule_trees=applicator_trees,
            special_tokens=self.special_tokens,
        )
        return applicator

    def _add_special_tokens_to_data(self):
        self._special_tokens_rules_applicator(self.raw_data)

    def get_save_dir(self, path, add_date_created_to_name: bool = True) -> str:
        if add_date_created_to_name:
            path = path + "_" + self.date_created

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def save(
        self,
        save_config: SaveConfiguration,
    ) -> str:
        path = self.get_save_dir(
            save_config.path, save_config.add_date_created_to_name
        )

        metadata_to_save = {
            "dataset_type": self.__class__.__name__,
            "date_created": self.date_created,
            "special_token_info": self.special_token_info,
        }

        if save_config.extra_metadata is not None:
            metadata_to_save.update(save_config.extra_metadata)

        data_to_save = self._raw_data
        data_file_path = os.path.join(path, "data.json")
        with open(data_file_path, mode="w", encoding="utf-8") as f:
            json.dump(obj=data_to_save, fp=f, indent=4, ensure_ascii=False)

        metadata_file_path = os.path.join(path, "metadata.json")
        with open(metadata_file_path, mode="w", encoding="utf-8") as f:
            json.dump(obj=metadata_to_save, fp=f, indent=4, ensure_ascii=False)
        return path

    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        self._raw_data = value


class TextClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_extension: str = None,
        class_label_field_name: str = "label",
        class_labels_mapping_file: str = None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            dataset_extension=dataset_extension,
            class_label_field_name=class_label_field_name,
            class_labels_mapping_file=class_labels_mapping_file,
        )

    def _get_label_distribution(self, data, dtype):
        labels = (
            l for elem in data for l in elem["metadata"][self.class_label_field_name]
        )
        if len(self._class_labels_mapping):
            labels = (
                self._get_mapped_label(label_str=l)
                for l in labels
                if l is not None
                if self._get_mapped_label(label_str=l) is not None
            )
        return dtype(labels)

    def _get_labels(self, data):
        return self._get_label_distribution(data, dtype=set)

    def _map_labels(self):
        for elem in self._raw_data:
            mapped_labels = []
            for label in elem["metadata"][self.class_label_field_name]:
                if len(self._class_labels_mapping):
                    label = self._get_mapped_label(label_str=label)
                if label is None:
                    continue
                mapped_label = self.label2id[label]
                mapped_labels.append(mapped_label)
            if not mapped_labels:  # empty list converts to None == unlabelled data
                mapped_labels = None
            elem["metadata"][self.class_label_field_name] = mapped_labels


class SequenceLabellingDataset(ClassificationDataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_extension: str = None,
        class_label_field_name: str = "label",
        class_labels_mapping_file: str = None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            dataset_extension=dataset_extension,
            class_label_field_name=class_label_field_name,
            class_labels_mapping_file=class_labels_mapping_file,
        )
        self._show_histogram = False
        self._assert_message = True

    def save(self, save_config: SaveConfiguration) -> str:
        self._show_histogram = save_config.show_class_labels_histogram
        extra_metadata = {}
        path = self.get_save_dir(
            save_config.path,
            add_date_created_to_name=save_config.add_date_created_to_name,
        )

        if save_config.save_iob_file or save_config.save_text_label_xlsx:
            out_dataset = self._get_iob_data_format(
                self._raw_data, True, save_config.iob_window_o_class
            )

            if save_config.save_text_label_xlsx:
                text_label_xlsx_path = os.path.join(
                    path,
                    "annotations__text_label_pairs.xlsx",
                )
                text_label_pairs = self._get_text_label_xlsx_format(out_dataset)
                text_label_pairs_flat = []
                # concat all splits
                for split in text_label_pairs.values():
                    text_label_pairs_flat.extend(split)
                df = pd.DataFrame(
                    text_label_pairs_flat, columns=["tagged_text", "label"]
                )
                df.to_excel(text_label_xlsx_path, index=False)

            if save_config.save_iob_file:
                iob_file_path = os.path.join(
                    path,
                    "data_iob.json",
                )
                with open(iob_file_path, "wt") as fout:
                    json.dump(out_dataset, fout, indent=2)

            cl_hist = self._get_advanced_iob_label_histogram(out_dataset)
            extra_metadata["advanced_iob_class_labels_hist"] = cl_hist

            if self._show_histogram:
                print(json.dumps(cl_hist, indent=2))

        extra_metadata["class_labels_hist"] = self._label_histogram
        save_config.extra_metadata = extra_metadata
        return super().save(save_config)

    def _get_text_label_xlsx_format(self, iob_format_dataset):
        pairs = {}
        for split, split_values in iob_format_dataset.items():
            if split not in pairs:
                pairs[split] = []
            for text_pack in split_values:
                for text in text_pack:
                    labels = set(
                        [x[2:] for x in text["iob_tags"].split() if x != "O"]
                    )
                    if len(labels) == 1:
                        label = list(labels)[0]
                    else:
                        raise RuntimeError()
                    elem = {"tagged_text": text["tagged_text"], "label": label}
                    pairs[split].append(elem)
        return pairs

    def _get_iob_data_format(
        self,
        dataset: list,
        merge_example_labels: bool = True,
        iob_window_o_class: int = 3,
    ):
        all_examples = self._convert_to_iob_examples(
            dataset, merge_example_labels, iob_window_o_class
        )
        train_data, test_data = sklearn.model_selection.train_test_split(
            all_examples,
            test_size=0.05,
            shuffle=True,
            random_state=self._random_state,
        )
        out_dataset = {"train": train_data, "test": test_data}
        return out_dataset

    @staticmethod
    def _get_advanced_iob_label_histogram(dataset_dict: dict):
        """
        Generates a histogram of class labels distribution
        on the IOB level after tokenization and label
        assignment to each token.
        """
        hist = {}
        for split, split_dataset in dataset_dict.items():
            hist[split] = {}
            for example in split_dataset:
                examples = [example]
                if type(example) in [list]:
                    examples = example
                for e in examples:
                    for label in e["iob_tags"].split():
                        if label not in hist[split]:
                            hist[split][label] = 0
                        hist[split][label] += 1
        return hist

    def _convert_to_iob_examples(
        self,
        data_to_save: list,
        merge_example_labels: bool,
        iob_window_o_class: int = 3,
    ):
        all_examples = []
        d_idx = 0
        for d_elem in data_to_save:
            iob_anno, d_idx = self._iob_for_text_labels(
                dataset_elem=d_elem,
                dataset_idx=d_idx,
                merge_example_labels=merge_example_labels,
                iob_window_o_class=iob_window_o_class,
            )
            # all_examples.extend(iob_anno)
            all_examples.append(iob_anno)
        return all_examples

    def _iob_for_text_labels(
        self,
        dataset_elem: dict,
        dataset_idx: int,
        merge_example_labels: bool = True,
        iob_window_o_class: int = 3,
    ):
        out_iob_anno = []
        text_str = dataset_elem["text"]
        text_labels = dataset_elem["metadata"][self.class_label_field_name]
        for labels in text_labels:
            b_label_pos, e_label_pos, _, iob_ann_i_tag = labels
            begin_text = text_str[:b_label_pos]
            anno_text = text_str[b_label_pos:e_label_pos]
            end_text = text_str[e_label_pos:]
            label_str = self._prepare_single_label(
                label_str=self.id2label.get(iob_ann_i_tag)
            )

            if not len(anno_text.strip()):
                print(
                    "Empty text:",
                    b_label_pos,
                    e_label_pos,
                    "\n",
                    len(text_str),
                    text_str,
                )
                continue

            iob_anno = [f"{label_str}"] * len(anno_text.split())
            iob_anno[0] = "B" + iob_anno[0][1:]
            o_begin = ["O"] * len(begin_text.split())
            o_end = ["O"] * len(end_text.split())

            m_text_str = text_str
            if iob_window_o_class > 0:
                o_begin = o_begin[-iob_window_o_class:]
                o_end = o_end[:iob_window_o_class]
                bt_spl = begin_text.split()[-iob_window_o_class:]
                et_spl = end_text.split()[:iob_window_o_class]
                at_spl = anno_text.split()
                m_text_str = " ".join(bt_spl + at_spl + et_spl)

                end_text = " ".join(et_spl)
                begin_text = " ".join(bt_spl)

            all_str_labels = o_begin + iob_anno + o_end
            labels_str = " ".join(all_str_labels).strip()
            tagged_text = f"{begin_text}[[{anno_text}]]{end_text}".rstrip()

            if len(m_text_str.strip().split()) != len(labels_str.split()):
                if self._assert_message:
                    assert len(m_text_str.strip().split()) == len(
                        labels_str.split()
                    ), (
                        f"\n\nDifferent number of words and tags after "
                        f"converting text {dataset_idx}!\n\n"
                        f"{len(labels_str.split())}: {labels_str}\n\n"
                        f"{len(m_text_str.split())}: {m_text_str}\n\n"
                        f"{len(tagged_text.split())}: {tagged_text}\n\n"
                    )
                else:
                    print("Different length of annotation and text, skipping...")
            else:
                out_iob_anno.append(
                    {
                        "id": dataset_idx,
                        "raw_text": m_text_str,
                        "iob_tags": labels_str,
                        "tagged_text": tagged_text,
                    }
                )
            dataset_idx += 1
        if merge_example_labels:
            out_iob_anno = self._merge_text_annotations(
                text_annotations=out_iob_anno
            )

        return out_iob_anno, dataset_idx

    @staticmethod
    def _merge_text_annotations(text_annotations: list):
        # annotations = [
        #     t.split() for t in (a['iob_tags'] for a in text_annotations)
        # ]
        #
        # print(annotations)
        #
        # new_annotations = []
        # for i in range(len(annotations[0])):
        #     new_label_str = 'O'
        #     for a in annotations:
        #         label = a[i]
        #         if label != 'O':
        #             new_label_str = label
        #             break
        #     new_annotations.append(new_label_str)
        #
        # new_anno_str = " ".join(new_annotations)
        # new_annotation = text_annotations[0]
        # new_annotation["iob_tags"] = new_anno_str
        #
        # return [new_annotation]
        return text_annotations

    def _get_label_distribution(self, data, dtype=set):
        original_labels = dtype(
            l[2]
            for elem in data
            for l in elem["metadata"][self.class_label_field_name]
        )
        # TODO: why len instead of "if self._class_labels_maping:"
        if len(self._class_labels_mapping):
            original_labels = dtype(
                self._get_mapped_label(label_str=l)
                for l in original_labels
                if l is not None
                if self._get_mapped_label(label_str=l) is not None
            )

        labels_I = dtype(f"I-{elem}" for elem in original_labels)
        labels_B = dtype(f"B-{elem}" for elem in original_labels)
        outside_labels = dtype("O")

        if dtype == set:
            return labels_I | labels_B | outside_labels
        elif dtype == list:
            return labels_I + labels_B + outside_labels
        else:
            return labels_I, labels_B, outside_labels

    def _get_labels(self, data):
        return self._get_label_distribution(data, dtype=set)

    LABEL_PAIR_MAPPING_CACHE = {}

    def _map_label_pair(self, label_str):
        label_pair = self.LABEL_PAIR_MAPPING_CACHE.get(label_str, None)
        if label_pair is None:
            label_pair = (
                self.label2id[f"B-{label_str}"],
                self.label2id[f"I-{label_str}"],
            )
            self.LABEL_PAIR_MAPPING_CACHE[label_str] = label_pair
        return label_pair

    def _map_labels(self):
        for elem in self._raw_data:
            mapped_labels = []
            for idx, label in enumerate(
                elem["metadata"][self.class_label_field_name]
            ):
                label_str = label[2]
                if len(self._class_labels_mapping):
                    label_str = self._get_mapped_label(label_str)

                if label_str is None:
                    continue
                label_b_and_i = self._map_label_pair(label_str)
                label[2:4] = label_b_and_i
                mapped_labels.append(label)

            if not mapped_labels:  # empty list converts to None == unlabelled data
                mapped_labels = None
            elem["metadata"][self.class_label_field_name] = mapped_labels
