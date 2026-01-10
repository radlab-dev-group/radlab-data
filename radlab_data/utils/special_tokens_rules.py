from collections import defaultdict
from typing import List

import tqdm

OPERATORS = {
    ">": {"func": lambda x, y: x > y, "priority": 0},
    "<": {"func": lambda x, y: x < y, "priority": 0},
    ">=": {"func": lambda x, y: x >= y, "priority": 0},
    "<=": {"func": lambda x, y: x <= y, "priority": 0},
    "==": {"func": lambda x, y: x == y, "priority": 0},
    "and": {"func": lambda x, y: x and y, "priority": 1},
    "or": {"func": lambda x, y: x or y, "priority": 2},
}


class ApplicatorNode:
    NODE_TYPES = ["operator", "accessor", "value"]

    def __init__(self, type) -> None:
        assert type in self.NODE_TYPES
        self.type = type

        self.value = None
        self.child_a: ApplicatorNode = None
        self.child_b: ApplicatorNode = None
        self.operator_method = None
        self.access_method = None
        self.str_representation = None

    def _evaluate(self, data):
        if self.type == "accessor":
            return self._access(data)
        elif self.type == "operator":
            return self._operation(data)
        elif self.type == "value":
            return self._get_value()

    def _access(self, data):
        return self.access_method(data)

    def _operation(self, data):
        return self.operator_method(
            self.child_a._evaluate(data), self.child_b._evaluate(data)
        )

    def _get_value(self):
        return self.value

    def __str__(self) -> str:
        return f"{self.type} [{self.str_representation}]"


class PlacementAccessor:
    WHERE_KEYS = ["beginning", "end"]

    def __init__(self, object: dict, class_label_field_name: str = "labels") -> None:
        self.class_label_field_name = class_label_field_name
        self.object = object

    def set_value(self, placement_str: str, value):
        access_keys = placement_str.split("::")
        where = access_keys[-1]
        assert len(access_keys) > 1
        assert where in self.WHERE_KEYS
        first_key = access_keys[0]
        object_ptr = self.object
        accessed_data = []
        if first_key == "text":
            assert access_keys[1] == where
            accessed_data.append((object_ptr, "text"))
        elif first_key == self.class_label_field_name:
            assert access_keys[1] == where
            object_ptr = object_ptr[self.class_label_field_name]
            for idx in range(len(object_ptr)):
                accessed_data.append((object_ptr, idx))
        else:
            object_ptr = object_ptr["metadata"]
            for key in access_keys[:-2]:
                object_ptr = object_ptr[key]
            last_ptr = object_ptr[access_keys[-2]]
            if isinstance(last_ptr, list):
                object_ptr = last_ptr
                for idx in range(len(object_ptr)):
                    accessed_data.append((object_ptr, idx))
            else:
                accessed_data.append((object_ptr, access_keys[-2]))

        for ptr, key in accessed_data:
            ptr[key] = self._modify_string(ptr[key], value, where)

    @staticmethod
    def _modify_string(string: str, value: str, where: str):
        if where == "beginning":
            return value + string
        elif where == "end":
            return string + value


class ApplicatorTree:
    def __init__(
        self,
        root: ApplicatorNode,
        rule_str: str,
        placement: str,
        special_token: str,
    ) -> None:
        self.root: ApplicatorNode = root
        if self.root.type != "operator":
            raise ValueError(f"The following rule contains no operators: {rule_str}")
        self.placement: str = placement
        self.special_token: str = special_token

    def _evaluate(self, data):
        return self.root._evaluate(data)

    def print_tree(self):
        print(f"{self.__class__.__name__}")
        print(str(self))

    def _get_str_representation(self):
        return self._get_str_representation_recursive(self.root, 0)

    def _get_str_representation_recursive(self, node, depth):
        if node is not None:
            string = "\t" * depth + str(node)
            if node.child_a is not None:
                string += "\n" + self._get_str_representation_recursive(
                    node.child_a, depth + 1
                )
            if node.child_b is not None:
                string += "\n" + self._get_str_representation_recursive(
                    node.child_b, depth + 1
                )
            return string
        return ""

    def __str__(self) -> str:
        return self._get_str_representation()

    def __call__(self, *args, **kwargs):
        return self._evaluate(*args, **kwargs)


class RuleApplicator:
    def __init__(
        self, rule_trees: List[ApplicatorTree], special_tokens: List[str]
    ) -> None:
        self.rule_trees: List[ApplicatorTree] = rule_trees
        self.special_tokens: List[str] = special_tokens

    def _process(self, data):
        for elem in tqdm.tqdm(data):
            true_trees = [tree for tree in self.rule_trees if tree(elem)]
            self._apply_trees(true_trees, elem)

    @staticmethod
    def _apply_trees(trees: List[ApplicatorTree], elem: dict):
        accessor = PlacementAccessor(elem)
        placement_buckets = defaultdict(list)
        all_placements = [tree.placement for tree in trees]
        all_special_tokens = [tree.special_token for tree in trees]
        for placement, special_token in zip(all_placements, all_special_tokens):
            placement_buckets[placement].append(special_token)
        for placement, bucket_values in placement_buckets.items():
            tokens_fused = "".join(bucket_values)
            accessor.set_value(placement, tokens_fused)

    def __call__(self, data):
        return self._process(data)


class SpecialTokensRulesParser:
    def __init__(self, special_tokens) -> None:
        self.special_tokens = special_tokens

    def _merge_rules(self, special_token_rules):
        groups = defaultdict(list)
        for rule in special_token_rules:
            group_unique_id = rule["token"] + " " + rule["placement"]
            groups[group_unique_id].append(rule)
        new_rules = []
        for group in groups.values():
            if len(group) == 1:
                rule = group[0]
            else:
                when = " or ".join(f"({s['when']})" for s in group)
                rule = {
                    "token": group[0]["token"],
                    "placement": group[0]["placement"],
                    "when": when,
                }
            new_rules.append(rule)
        return new_rules

    def _sort_rules(self, special_tokens_rules):
        buckets = {k: [] for k in self.special_tokens}  # ordered dict (Python 3.7)
        for rule in special_tokens_rules:
            token = rule["token"]
            buckets[token].append(rule)
        new_rules = []
        for bucket in buckets.values():
            new_rules.extend(bucket)
        return new_rules

    def parse(self, special_tokens_rules) -> List[ApplicatorTree]:
        special_tokens_rules = self._merge_rules(special_tokens_rules)
        special_tokens_rules = self._sort_rules(special_tokens_rules)

        rule_trees: List[ApplicatorTree] = []
        for rule in special_tokens_rules:
            token = rule["token"]
            when = rule["when"]
            placement = rule["placement"]

            when_words = list(
                enumerate(when.replace("(", " ( ").replace(")", " ) ").split())
            )  # whitespace split

            root_node = self.parse_when_words(when_words, when)
            when_tree = ApplicatorTree(
                root=root_node,
                rule_str=when,
                placement=placement,
                special_token=token,
            )
            rule_trees.append(when_tree)
        return rule_trees

    def parse_when_words(self, when_words, sentence):
        try:
            return self._parse_when_words_recursive(when_words)
        except:
            raise Exception(
                f"Error has occured while parsing the following rule: {sentence}"
            )

    def _are_outer_brackets_present(self, expression):
        if (
            len(expression) >= 3
            and expression[0][1] == "("
            and expression[-1][1] == ")"
        ):
            stack = 0
            for _, word in expression[1:-1]:
                if word == "(":
                    stack += 1
                elif word == ")":
                    if stack == 0:
                        return False
                    stack -= 1
            return stack == 0
        return False

    def _parse_when_words_recursive(
        self, when_words: list, used_operators: list = None
    ) -> ApplicatorNode:
        if self._are_outer_brackets_present(when_words):
            when_words = when_words[1:-1]

        used_operators = []
        bracket_count = 0
        for idx, word in when_words:
            if word == "(":
                bracket_count += 1
            elif word == ")":
                bracket_count -= 1
            elif word in OPERATORS and bracket_count == 0:
                operator = OPERATORS[word]
                used_operators.append(
                    {"word_idx": idx, "operator": operator, "word": word}
                )

        if len(used_operators) == 0:
            assert len(when_words) == 1
            word = when_words[0][
                1
            ]  # 0 because first word in the list, and 1 because index 0 is the index of a word and index 1 is the string

            is_value = word.replace(".", "").isnumeric()
            if is_value:
                node = ApplicatorNode(type="value")
                node.value = float(word)
                node.str_representation = word
                return node

            access_keys = word.split("::")

            def access_method(x):
                first_key = access_keys[0]
                if first_key not in ["text", "labels"]:
                    x = x["metadata"]
                for access_key in access_keys:
                    x = x[access_key]
                return x

            node = ApplicatorNode(type="accessor")
            node.access_method = access_method
            node.str_representation = word
            return node
        else:
            active_operator = max(
                used_operators,
                key=lambda x: (x["operator"]["priority"], x["word_idx"]),
            )
            split_point = active_operator["word_idx"]
            words_left = [w for w in when_words if w[0] < split_point]
            words_right = [w for w in when_words if w[0] > split_point]
            assert len(words_left) > 0
            assert len(words_right) > 0
            operators_left = [
                op for op in used_operators if op["word_idx"] < split_point
            ]
            operators_right = [
                op for op in used_operators if op["word_idx"] > split_point
            ]
            node = ApplicatorNode(type="operator")
            node.child_a = self._parse_when_words_recursive(
                words_left, operators_left
            )
            node.child_b = self._parse_when_words_recursive(
                words_right, operators_right
            )
            node.operator_method = active_operator["operator"]["func"]
            node.str_representation = f"{active_operator['word']}"
            return node

    def get_rule_applicator(self) -> RuleApplicator:
        raise NotImplementedError()


if __name__ == "__main__":
    import copy

    data = [
        {
            "text": "tekst input przyklad",
            "labels": ["przykladowy tekst label", "przykladowy tekst label 2"],
            "metadata": {
                "left_ctx": "kontekst tekstu z lewej strony",
                "right_ctx": "kontekst tekstu z prawej strony",
                "tov_classification_results": {
                    "prosty_glos_ing": 0.5,
                    "jezyk_skomplikowany": 0.6,
                },
            },
        }
    ]
    max_data_len = 1000
    for idx in range(1, max_data_len):
        new_elem = copy.deepcopy(data[0])
        tov = new_elem["metadata"]["tov_classification_results"]
        tov["prosty_glos_ing"] = (max_data_len % idx) / max_data_len
        tov["jezyk_skomplikowany"] = ((max_data_len * 3) % idx) / max_data_len
        data.append(new_elem)

    special_token_data = {
        "special_tokens": ["<test1>", "<test2>"],
        "special_token_rules": [
            {
                "token": "<test2>",
                "when": "(tov_classification_results::prosty_glos_ing > 0.2) or (tov_classification_results::prosty_glos_ing > 0.6 or tov_classification_results::prosty_glos_ing == 0.9)",
                "placement": "text::beginning",
            },
            {
                "token": "<test1>",
                "when": "(tov_classification_results::prosty_glos_ing > 0.2) and (tov_classification_results::prosty_glos_ing > 0.6 or tov_classification_results::prosty_glos_ing == 0.9)",
                "placement": "text::end",
            },
            {
                "token": "<test2>",
                "when": "tov_classification_results::jezyk_skomplikowany > 0.5",
                "placement": "text::end",
            },
            {
                "token": "<test1>",
                "when": "tov_classification_results::jezyk_skomplikowany < 0.9",
                "placement": "text::end",
            },
        ],
    }
    parser = SpecialTokensRulesParser(special_token_data["special_tokens"])
    applicator_trees = parser.parse(special_token_data["special_token_rules"])
    applicator = RuleApplicator(
        rule_trees=applicator_trees,
        special_tokens=special_token_data["special_tokens"],
    )
    for tree in applicator_trees:
        print(tree.special_token)
        print(tree)
        print()
        print("-" * 100)
        print()

    import json

    # with open("temp/test_dataset/data.jsonl", mode="w") as fp:
    #     for entry in data:
    #         json.dump(entry, fp)
    #         fp.write("\n")

    applicator._process(data)
    print(data)
