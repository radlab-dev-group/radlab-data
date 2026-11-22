import pytest

from radlab_data.utils.special_tokens_rules import (
    RuleApplicator,
    SpecialTokensRulesParser,
)

SPECIAL_TOKENS = ["<test1>", "<test2>"]


def _parse(rules):
    parser = SpecialTokensRulesParser(SPECIAL_TOKENS)
    return parser.parse(rules)


def test_parse_simple_rule():
    trees = _parse(
        [
            {
                "token": "<test1>",
                "when": "(metadata::score > 0.5)",
                "placement": "text::beginning",
            }
        ]
    )
    assert len(trees) == 1
    assert trees[0].special_token == "<test1>"
    assert trees[0].placement == "text::beginning"


def test_rule_evaluation_on_data():
    trees = _parse(
        [
            {
                "token": "<test1>",
                "when": "(score > 0.5)",
                "placement": "text::beginning",
            },
            {
                "token": "<test2>",
                "when": "(score < 0.5)",
                "placement": "text::end",
            },
        ]
    )
    applicator = RuleApplicator(rule_trees=trees, special_tokens=SPECIAL_TOKENS)
    data = [
        {
            "text": "treść",
            "labels": ["etykieta"],
            "metadata": {"score": 0.9},
        }
    ]
    applicator(data)
    assert data[0]["text"] == "<test1>treść"


def test_rule_with_or_combines_conditions():
    trees = _parse(
        [
            {
                "token": "<test1>",
                "when": "(score > 0.9) or (score < 0.1)",
                "placement": "text::beginning",
            }
        ]
    )
    applicator = RuleApplicator(rule_trees=trees, special_tokens=SPECIAL_TOKENS)
    data = [{"text": "a", "labels": [], "metadata": {"score": 0.05}}]
    applicator(data)
    assert data[0]["text"] == "<test1>a"


def test_unknown_special_token_reports_the_token():
    with pytest.raises(KeyError, match="Special token <brak>"):
        _parse(
            [
                {
                    "token": "<brak>",
                    "when": "(metadata::score > 0.5)",
                    "placement": "text::beginning",
                }
            ]
        )


def test_invalid_rule_keeps_cause():
    try:
        _parse(
            [
                {
                    "token": "<test1>",
                    "when": "(metadata::score >",
                    "placement": "text::beginning",
                }
            ]
        )
        pytest.fail("parse powinno zglosic blad")
    except Exception as error:
        assert "Error has occured while parsing" in str(error)
        assert error.__cause__ is not None
