from rag import evaluate_retrieval


def test_evaluate_query_calculates_metrics_with_category(monkeypatch):
    # 临时替换某个对象的属性
    monkeypatch.setattr(
        evaluate_retrieval,
        "get_top_k_sources",
        lambda query,k: ["维护保养.txt", "故障排除.txt", "选购指南.txt"],
    )

    metric = evaluate_retrieval.evaluate_query(
        {
            "query": "吸力变弱怎么处理",
            "expected_sources": ["维护保养.txt", "故障排除.txt"],
            "category": "混合意图",
            "reason": "同时涉及故障排查和维护清理",
        },
        k=3,
    )

    assert metric.category == "混合意图"
    assert metric.recall_at_k == 1.0
    assert metric.precision_at_k == 2 / 3
    assert metric.reciprocal_rank == 1.0


def test_validate_eval_set_rejects_missing_reason():
    invalid_eval_set = [
        {
            "query": "扫地机器人怎么维护",
            "expected_sources": ["维护保养.txt"],
            "category": "维护保养",
        }
    ]

    try:
        evaluate_retrieval.validate_eval_set(invalid_eval_set)
    except ValueError as exc:
        assert "reason" in str(exc)
    else:
        raise AssertionError("validate_eval_set 函数应当拒绝那些没有提供 reason 字段的样本。")

