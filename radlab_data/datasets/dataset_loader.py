import os

from radlab_data.datasets.dataset import DatasetBaseClass, get_class_from_type_name
from radlab_data.preprocessing.dataset import SequenceToSequenceDataset, json_loader


class DatasetLoader:
    """
    An interface for dataset loading and tokenization.

    Usage:
        1. Initialize the class.
        2. Use the load method.
    """

    def __init__(
        self,
        dataset_path,
        tokenizer,
        tokenizer_kwargs=None,
        cache_path="./cache/",
        class_label_field_name=None,
        split_train_valid_test: str = None,
        pre_shuffle=False,
    ) -> None:
        """
        :param dataset_path: Path to dataset
        :param tokenizer: Tokenizer
        :param cache_path: Path to cached data
        :param class_label_field_name: Class label field name
        :param split_train_test: Example format (train:valid:test): "0.4" (train 0.4, valid 0.6) or "0.4:0.6" (train 0.4, valid 0.6) or "0.4:0.1" (train 0.4, valid 0.1, test 0.5) or "0.2:0.3:0.5" (train 0.2, valid 0.3, test 0.5) or None. Create train and test datasets by a given ratio,
            create only train dataset if this variable is None
        :param pre_shuffle: Shuffle the dataset at initialization.
        """
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self._dataset_path = dataset_path
        self._metadata = self._read_json("metadata.json")
        self._dataset_type = get_class_from_type_name(self._metadata["dataset_type"])

        if class_label_field_name is None:
            if type(self._dataset_type) == type(SequenceToSequenceDataset):
                self._class_label_field_name = "labels"
            else:
                self._class_label_field_name = "label"
        else:
            self._class_label_field_name = class_label_field_name

        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._cache_path = cache_path
        self._cached_dataset_path = self._get_cached_dataset_path()

        if split_train_valid_test is not None:
            split_train_valid_test = split_train_valid_test.split(":")
            assert len(split_train_valid_test) in [
                1,
                2,
                3,
            ], "Incorrect amount of splits."
            split_train_valid_test = tuple(float(x) for x in split_train_valid_test)
            if len(split_train_valid_test) == 1:
                split_train_valid_test = (
                    split_train_valid_test[0],
                    1 - split_train_valid_test[0],
                )
            elif (
                len(split_train_valid_test) == 2 and sum(split_train_valid_test) != 1
            ):
                test_num = 1 - sum(split_train_valid_test)
                split_train_valid_test = (*split_train_valid_test, test_num)

            for split in split_train_valid_test:
                assert (
                    0 < split < 1
                ), "Split values should be greater than 0 and less than 1"
            assert (
                sum(split_train_valid_test) == 1
            ), "Train, Valid and Test rations must add up to 1."

        self._split_train_valid_test = split_train_valid_test
        self._pre_shuffle = pre_shuffle

    def load(self) -> DatasetBaseClass:
        """
        Loads the dataset.
        """
        raw_data = self._read_json("data.json")

        dataset = self._dataset_type(
            raw_data,
            self._metadata,
            self._cached_dataset_path,
            self._tokenizer,
            self._tokenizer_kwargs,
            self._class_label_field_name,
            self._split_train_valid_test,
            self._pre_shuffle,
        )

        return dataset

    def _get_cached_dataset_path(self) -> str:
        date_created = self._metadata["date_created"]
        tokenizer_class_name = self._tokenizer.__class__.__name__
        dataset_name = os.path.basename(self._dataset_path)
        dataset_class_name = self._dataset_type.__name__
        key_string = (
            dataset_name
            + "__"
            + dataset_class_name
            + "__"
            + tokenizer_class_name
            + "__"
            + date_created
        )
        cache_name = key_string + ".cache"
        cached_dataset_path = os.path.join(self._cache_path, cache_name)
        return cached_dataset_path

    def _read_json(self, file_name) -> dict:
        """
        Loads JSON formatted file and tokenizes allowed fields.
        """
        path_to_load = os.path.join(self._dataset_path, file_name)
        assert os.path.exists(path_to_load), f"{path_to_load} does not exist."
        assert os.path.isfile(path_to_load), f"{path_to_load} is not a file."
        _, extension = os.path.splitext(path_to_load)
        assert (
            extension == ".json"
        ), "Only JSON formatted dataset files are supported."
        return json_loader(path_to_load)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("radlab/polish-fast-tokenizer")
    dl = DatasetLoader(
        "/home/kschnetzer/projects/radlab-data/temp/dialoguetestdataset",
        tokenizer,
        split_train_valid_test="0.1:0.1",
    )
    dataset = dl.load()
    train = dataset.train_dataset
    valid = dataset.valid_dataset
    test = dataset.test_dataset
    print(train[0])
    if train:
        print(len(train))
    if valid:
        print(len(valid))
    if test:
        print(len(test))
