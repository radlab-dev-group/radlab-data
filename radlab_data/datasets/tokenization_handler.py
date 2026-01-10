from typing import List


class TokenizationHandler:
    """
    Handles all the tokenization (and paralellization) logic.

    Algorithm details:
    This class is used for fast dictionary tokenization.
        Acyclic Graph Tokenizer by Kacper Schnetzer
        Created: 2023

    @ :)
    """

    def __init__(
        self,
        tokenizer,
        blacklist: List[str] = None,
        tokenization_batch_size: int = 256,
    ) -> None:
        """
        Parameters:
        tokenizer - a HuggingFace tokenizer class used for tokenization
        blacklist: List[str] - if tokenizing a dict: list of keys in the dict
                   that the tokenizer pass through
        """
        self._tokenizer = tokenizer
        self._blacklist = blacklist
        self._tokenization_batch_size = tokenization_batch_size

    def _get_gatherer(self, obj, parent=None, key=None):
        if isinstance(obj, str):
            yield obj, parent, key
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in self._blacklist:
                    continue
                yield from self._get_gatherer(v, parent=obj, key=k)
        elif isinstance(obj, list):
            for k, v in enumerate(obj):
                yield from self._get_gatherer(v, parent=obj, key=k)

    def _gather(self, gatherer, element_count):
        out_arr = []
        for _ in range(element_count):
            elem = next(gatherer, None)
            if elem is None:
                return out_arr, True
            else:
                out_arr.append(elem)
        return out_arr, False

    def _update_branch(self, parent, value, key):
        parent[key] = value

    def _update_tree(self, branches):
        for branch in branches:
            self._update_branch(parent=branch[1], value=branch[0], key=branch[2])

    def tokenize(self, dataset):
        element_gatherer = self._get_gatherer(dataset)
        while True:
            branches, finished = self._gather(
                element_gatherer, element_count=self._tokenization_batch_size
            )
            if not branches:
                break
            leaves = [branch[0] for branch in branches]
            tokenized_leaves = self._tokenizer(leaves)["input_ids"]
            branches = [
                (tokenized_leaf, branch[1], branch[2])
                for tokenized_leaf, branch in zip(tokenized_leaves, branches)
            ]
            self._update_tree(branches=branches)
            if finished:
                break
        return dataset
