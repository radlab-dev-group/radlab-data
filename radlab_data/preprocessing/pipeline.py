from abc import ABC, abstractmethod
from typing import List

from radlab_data.preprocessing.dataset import ClassificationDataset


class PipelineElementBase(ABC):
    ALLOWED_DATASET_TYPES: tuple = ()
    ACTION_MAPPING: dict = {}

    @abstractmethod
    def _process_data(self, dataset: ClassificationDataset) -> ClassificationDataset:
        pass

    def __call__(self, dataset: ClassificationDataset) -> ClassificationDataset:
        assert self._dataset_type_check(
            dataset
        ), f"{self.__class__.__name__} pipeline element does not support dataset of type {dataset.__class__.__name__}."
        return self._process_data(dataset)

    def _dataset_type_check(self, dataset: ClassificationDataset) -> bool:
        if self.ALLOWED_DATASET_TYPES:
            return isinstance(dataset, self.ALLOWED_DATASET_TYPES)
        else:
            # ALLOWED_DATASET_TYPES empty => any type allowed
            return True


class Pipeline:
    def __init__(self, modules: List[PipelineElementBase]) -> None:
        self.modules = modules

    def __call__(self, dataset: ClassificationDataset) -> ClassificationDataset:
        module_count = len(self.modules)
        for idx, module in enumerate(self.modules, 1):
            print(f"[{idx}/{module_count}] Pipeline: {module.__class__.__name__}")
            dataset = module(dataset)
        return dataset
