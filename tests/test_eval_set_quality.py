import json
from collections import Counter
from pathlib import Path

from rag.evaluate_retrieval import validate_eval_set


EVAL_SET_PATH = Path("data/rag_eval_set.example.json")



def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding = "utf-8"))

def test_eval_set_has_minimum_size_and_required_fields():
    eval_set = load_eval_set()

    validate_eval_set(eval_set)

    # 断言测试
    assert len(eval_set) >= 25
    # 不能为空
    for item in eval_set:
        assert item["query"].strip()
        assert item["expected_sources"]
        assert item["category"].strip()
        assert item["reason"].strip()


def test_eval_set_covers_core_categories_and_mixed_intent():
    eval_set = load_eval_set()
    counts = Counter(item["category"] for item in eval_set)

    for category in ["维护保养", "故障排除", "选购指南", "扫拖一体"]:
        assert counts[category] >= 4

    assert counts["混合意图"] >= 4



