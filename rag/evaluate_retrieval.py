import argparse # 处理命令行参数
from collections import defaultdict
import json 
from dataclasses import dataclass # 自动生成 __init__、__repr__、__eq__等
from pathlib import Path
import sys
from typing import Any

from utils.path_tool import get_abs_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 定义数据类
@dataclass
class QueryMetrics:
    query: str
    rewritten_query: str
    expected_sources: list[str]
    category: str
    reason: str
    matched_sources: list[str]
    top_k_sources: list[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


@dataclass
class CompareMetrics:
    baseline: QueryMetrics
    enhanced: QueryMetrics

# 加载评估数据集
def load_eval_set(eval_path: Path) -> list[dict[str, Any]]:
    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("评测文件必须是 JSON 数组")
    
    validate_eval_set(data)
    return data
        

# 评测集的结构校验函数
def validate_eval_set(data: list[dict[str, Any]]) -> None:
    required_keys = ("query", "expected_sources", "category", "reason", "evidence")
    errors: list[str] = []

    for idx, item in enumerate(data, start = 1):
        if not isinstance(item, dict):
            errors.append(f"第 {idx} 条样本必须是 JSON 对象")
            continue
        
        for field in required_keys:
            if field not in item:
                errors.append(f"第 {idx} 条样本缺少字段 {field}")
        
        # 读取字段并作基础清洗
        query = str(item.get("query","")).strip()
        category = str(item.get("category", "")).strip()
        reason = str(item.get("reason", "")).strip()
        expected_sources = item.get("expected_sources")
        evidence = item.get("evidence")

        if not query:
            errors.append(f"第 {idx} 条样本的 query 字段不能为空")
        if not category:
            errors.append(f"第 {idx} 条样本的 category 字段不能为空")
        if not reason:
            errors.append(f"第 {idx} 条样本的 reason 字段不能为空")
        if not isinstance(expected_sources, list) or not expected_sources:
            errors.append(f"第 {idx} 条样本的 expected_sources 字段必须是非空数组")
            continue
        
        if not isinstance(evidence, dict) or not evidence:
            errors.append(f"第 {idx} 条样本的 evidence 字段必须是非空对象")
            continue

        expected_sources_name = {normalize_source(str(source)) for source in expected_sources}
        evidence_source_names = {normalize_source(str(source)) for source in evidence}
        missing_evidence = sorted(expected_sources_name - evidence_source_names)
        if missing_evidence:
            errors.append(f"第 {idx} 条样本的 evidence 缺少来源说明：{missing_evidence}")
        
        for source, evidence_text in evidence.items():
            if not str(evidence_text).strip():
                errors.append(f"第 {idx} 条样本的 evidence [{source}] 的内容不能为空")
    
    if errors:
        raise ValueError("评测集格式不符合要求:\n" + "\n".join(errors))
    

# 来源路径规范化成只有文件名的形式
def normalize_source(source: str) -> str:
    return Path(source).name.replace("\\", "/")


# 加载默认的 k 值
def load_default_k() -> int:
    chroma_config_path = Path(get_abs_path("config/chroma.yml"))
    try:
        for line in chroma_config_path.read_text(encoding="utf-8").splitlines():
            key, sep,value = line.partition(":")
            if  sep and key.strip() == "k":
                return int(value.strip())
    except Exception:
        pass
    return 4


def get_top_k_sources(query: str, k: int, mode: str = "baseline") -> list[str]:
    from rag.vector_store import VectorStoreService # 延迟导入
     
    # 创建向量库服务对象 
    vector_store = VectorStoreService()
    # 返回page_content：文档片段正文 metadata：元数据，比如来源文件路径
    if mode == "enhanced":
        docs = vector_store.similarity_search_enhanced(query, k = k) 
    else:
        docs = vector_store.similarity_search_baseline(query, k = k)

    top_k_sources: list[str] = []
    for doc in docs:
        source = normalize_source(str(doc.metadata.get("source", "unknown")))
        top_k_sources.append(source)
    return top_k_sources


# 对单个查询进行评估
def evaluate_query(item: dict[str, Any], k: int, mode: str = "baseline") -> QueryMetrics:
    from rag.query_rewrite import rewrite_query

    query = str(item["query"]).strip()
    rewrite = rewrite_query(query)
    category = str(item["category"]).strip()
    reason = str(item["reason"]).strip()
    expected_sources = [
        normalize_source(str(source))
        for source in item.get("expected_sources", [])
    ]
    if mode == "baseline":
        top_k_sources = get_top_k_sources(query, k)
    else:
        top_k_sources = get_top_k_sources(query, k, mode = mode)

    matched_sources = [
        source for source in expected_sources if source in top_k_sources
    ]

    expected_count = len(expected_sources)
    matched_count = len(matched_sources)
    top_k_count = len(top_k_sources)

    # 计算召回率
    recall_at_k = matched_count / expected_count if expected_count else 0.0
    # 获取准确率
    precision_at_k = matched_count / top_k_count if top_k_count else 0.0

    # 倒数排名
    reciprocal_rank = 0.0
    for idx, source in enumerate(top_k_sources, start = 1):
        if source in expected_sources:
            reciprocal_rank = 1 / idx
            break
    
    return QueryMetrics(
        query = query,
        rewritten_query=rewrite.rewritten_query if mode == "enhanced" else query,
        expected_sources = expected_sources,
        category = category,
        reason = reason,
        matched_sources = matched_sources,
        top_k_sources = top_k_sources,
        recall_at_k = recall_at_k,
        precision_at_k = precision_at_k,
        reciprocal_rank = reciprocal_rank,
    )


def evaluate_compare_query(item: dict[str], k: int) -> CompareMetrics:
    return CompareMetrics(
        baseline = evaluate_query(item, k, mode = "baseline"),
        enhanced = evaluate_query(item, k, mode = "enhanced"),
    )


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

# 按类别进行分组
def group_by_category(metrics: list[QueryMetrics]) -> dict[str, list[QueryMetrics]]:
    grouped: dict[str, list[QueryMetrics]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.category].append(metric)
    return dict(grouped)

# 按类别打印评估结果
def print_categor_metrics(metrics: list[QueryMetrics],k : int) -> None:
    print("按 category 汇总:")
    for category, category_metrics in sorted(group_by_category(metrics).items()):
        recall = average([m.recall_at_k for m in category_metrics])
        precision = average([m.precision_at_k for m in category_metrics])
        mrr = average([m.reciprocal_rank for m in category_metrics])
        print(
            f"- {category}: 样本数={len(category_metrics)},"
            f"Recall@{k}={recall:.4f},"
            f"Precision@{k}={precision:.4f},"
            f"MRR@{k}={mrr:.4f}"
        )
    print("")


# 打印失败的样本
def print_failed_samples(metrics: list[QueryMetrics],k : int) -> None:
    failed_metrics = [metrics for metrics in metrics if metrics.recall_at_k < 1.0]
    if not failed_metrics:
        print("失败样本: 无")
        print("")
        return
    
    print(f"失败样本:{len(failed_metrics)}")
    for metric in failed_metrics:
        missing_sources = [
            source for source in metric.expected_sources 
            if source not in metric.top_k_sources
        ]
        print(f"- Query: {metric.query}")
        print(f"  Category: {metric.category}")
        print(f"  Reason: {metric.reason}")
        print(f"  Missing: {missing_sources}")
        print(f"  Retrieved: {metric.top_k_sources}")
        print(f"  Recall@{k}={metric.recall_at_k:.4f}")
    print("")


# 打印评估结果摘要
def print_metrics_summary(
        metrics: list[QueryMetrics],
        k: int,
        label: str = "",
) -> tuple[float, float, float]:
    if label:
        print(label)

    mean_recall = average([m.recall_at_k for m in metrics])
    mean_precision = average([m.precision_at_k for m in metrics])
    mean_mrr = average([m.reciprocal_rank for m in metrics])

    print(f"Recall@{k}: {mean_recall:.4f}")
    print(f"Precision@{k}: {mean_precision:.4f}")
    print(f"MRR@{k}: {mean_mrr:.4f}")
    print("")
    return mean_recall, mean_precision, mean_mrr

# 评估结果对比
def print_compare_summary(compare_metrics: list[CompareMetrics],k : int) -> float:
    baseline_metrics = [metric.baseline for metric in compare_metrics]
    enhanced_metrics = [metric.enhanced for metric in compare_metrics]

    baseline_recall, baseline_precision, baseline_mrr = print_metrics_summary(
        baseline_metrics, k, label = "Baseline"
    )
    enhanced_recall, enhanced_precision, enhanced_mrr = print_metrics_summary(
        enhanced_metrics, k, label = "Enhanced"
    )

    improved = 0
    regressed = 0
    unchanged = 0
    for metric in compare_metrics:
        delta = metric.enhanced.recall_at_k - metric.baseline.recall_at_k
        if delta > 0:
            improved += 1
        elif delta < 0:
            regressed += 1
        else:
            unchanged += 1

    print("Compare")
    print(f"Recall@{k} delta: {enhanced_recall - baseline_recall:+.4f}")
    print(f"Precision@{k} delta: {enhanced_precision - baseline_precision:+.4f}")
    print(f"MRR@{k} delta: {enhanced_mrr - baseline_mrr:+.4f}")
    print(f"Improved samples: {improved}")
    print(f"Regressed samples: {regressed}")
    print(f"Unchanged samples: {unchanged}")
    print("")

    failed_metrics = [
        metric for metric in compare_metrics
        if metric.baseline.recall_at_k < 1.0 or metric.enhanced.recall_at_k < 1.0
    ]
    if failed_metrics:
        print(f"对比失败样本: {len(failed_metrics)}")
        for metric in failed_metrics:
            missing_sources = [
                source for source in metric.enhanced.expected_sources 
                if source not in metric.enhanced.top_k_sources
            ]
            print(f"- Query: {metric.baseline.query}")
            print(f"  Rewritten: {metric.enhanced.rewritten_query}")
            print(f"  Baseline retrieved: {metric.baseline.top_k_sources}")
            print(f"  Enhanced retrieved: {metric.enhanced.top_k_sources}")
            print(f"  Missing after enhanced: {missing_sources}")
        print("")

    return enhanced_recall

#  RAG 检索评估脚本的总控流程
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索离线测评")
    parser.add_argument(
        "--evals-set",
        default="data/rag_eval_set.example.json",
        help = "评测集 JSON 文件路径"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=load_default_k(),
        help = "检索 top-k，默认取 config/chroma.yml 中的 k"
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default = None,
        help = "最低平均 Recall@K 阈值；低于该值时以非零退出码结束"
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "enhanced", "compare"),
        default = "baseline",
        help = "评测模式: baseline 原始检索；enhanced 改写+重排；compare 对比两者"
    )
    # 读取命令行传进来的参数，结果放到 args 里
    args = parser.parse_args()
    # 获取绝对路径,Path标准化处理
    eval_path = Path(get_abs_path(args.evals_set))
    eval_set = load_eval_set(eval_path)

    try:
        if args.mode == "compare":
            compare_metrics = [evaluate_compare_query(item, args.k) for item in eval_set]
            metrics = [metrics.enhanced for metrics in compare_metrics]
        else:
            compare_metrics = []
            metrics = [evaluate_query(item, args.k, mode=args.mode) for item in eval_set]
    except Exception as exc:
        print("评测执行失败。")
        print("可能原因:")
        print("1. 当前 Python 环境未安装项目依赖。")
        print("2. DashScope embedding 需要可用的 API Key 和网络。")
        print("3. 代理配置异常，导致无法访问 DashScope。")
        print(f"原始错误: {exc}")
        raise

    print(f"评测集: {eval_path}")
    print(f"样本数: {len(metrics)}")
    print(f"Mode: {args.mode}")
    if args.mode == "compare":
        mean_recall = print_compare_summary(compare_metrics, args.k)
    else:
        mean_recall, _, _ = print_metrics_summary(metrics, args.k)

    print_categor_metrics(metrics, args.k)
    print_failed_samples(metrics, args.k)

    for metric in metrics:
        print(f"Query: {metric.query}")
        if metric.rewritten_query != metric.query:
            print(f"Rewritten: {metric.rewritten_query}")
        print(f"Category: {metric.category}")
        print(f"Reason: {metric.reason}")
        print(f"Expected: {metric.expected_sources}")
        print(f"Retrieved: {metric.top_k_sources}")
        print(f"Matched: {metric.matched_sources}")
        print(
            f"Recall@{args.k}={metric.recall_at_k:.4f}, "
            f"Precision@{args.k}={metric.precision_at_k:.4f}, "
            f"RR={metric.reciprocal_rank:.4f}"
        )
        print("-" * 60)

    if args.min_recall is not None and mean_recall < args.min_recall:
        print(
            f"Recall@{args.k}={mean_recall:.4f} 低于阈值 "
            f"{args.min_recall:.4f},评测未通过。"
        )
        raise SystemExit(1)

if __name__ == "__main__":
    main()
