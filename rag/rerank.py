import re

from langchain_core.documents import Document

from rag.query_rewrite import RewriteResult


STOPWORDS = {
    "扫地",
    "机器",
    "器人",
    "机器人",
    "扫地机",
    "扫地机器人",
    "怎么",
    "什么",
    "应该",
    "优先",
    "可以",
    "需要",
    "处理",
    "功能",
    "哪些",
    "一下",
    "比较",
    "时候",
    "是不是",
    "原因",
}


def _is_stop_token(token: str) -> bool:
    return token in STOPWORDS or any(stopword in token for stopword in STOPWORDS)


def _tokens(text: str) -> list[str]:
    # 要么匹配连续的英文/数字，要么匹配 2 个字及以上的中文连续字符串
    raw_tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}",text.lower())
    tokens: list[str] = []
    fallback_tokens: list[str] = []
    for token in raw_tokens:
        # 保留一份未过滤的 fallback_tokens, 兜底措施
        if token not in fallback_tokens:
            fallback_tokens.append(token)
        # 处理中文
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}",token):
            max_size = min(4,len(token))
            # 尝试暴力切分,滑动切割
            for size in range(2,max_size + 1):
                # 保证切出来的每一段都正好等于 size
                for start in range(0,len(token) - size + 1):
                    piece = token[start: start + size]
                    if _is_stop_token(piece):
                        continue
                    if piece not in tokens:
                        tokens.append(piece)
            continue
        # 针对非长中文词
        if _is_stop_token(token):
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens or fallback_tokens



def _contains_count(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    normalized_terms = _normalize_terms(terms)
    return sum(1 for term in normalized_terms if term in lowered)


# 对 terms 进行标准化、去空、去重、数量截断
def _normalize_terms(terms: list[str], max_terms: int = 60) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for term in terms:
        value = term.strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
        if len(normalized) >= max_terms:
            break

    return normalized


# 计算文档得分
def score_documents(doc: Document, rewrite: RewriteResult) -> float:
    content = doc.page_content or ""
    metadata = doc.metadata or {}
    question = str(metadata.get("question", ""))
    section_title = str(metadata.get("section_title", ""))
    combined = f"{content}\n{question}\n{section_title}"

    query_tokens = _tokens(rewrite.original_query)
    expansion_terms = rewrite.expansion_terms

    score = 0.0
    # 计算原始查询词在全文中出现的个数，每个词计 2.0 分
    score += _contains_count(combined, query_tokens) * 2.0
    # 计算扩展词在全文中出现的个数，每个词计 1.0 分
    score += _contains_count(combined, expansion_terms) * 1.0
    # 计算扩展词在问题字段中出现的个数，每个词计 3.0 分
    score += _contains_count(question, query_tokens) * 3.0
    # 计算扩展词在问题字段中出现的个数，每个词计 1.5 分
    score += _contains_count(question, expansion_terms) * 1.5

    if metadata.get("chunk_type") == "faq":
        # 添加 FAQ 文档的得分 优先推荐这种标准答案格式
        score += 5.0

    return score


def rerank_documents(
        docs: list[Document],
        rewrite: RewriteResult,
        k: int
) -> list[Document]:
    ranked = sorted(
        enumerate(docs),
        # 多级排序 分数越高越靠前 分数相同，原始索引越小的越靠前 降序
        key = lambda item: (score_documents(item[1],rewrite), -item[0]),
        reverse = True
    )
    # 返回前 k 个文档
    return [doc for _, doc in ranked[:k]]


def rerank_documents_with_source_diversity(
        docs: list[Document],
        rewrite: RewriteResult,
        k: int,
) -> list[Document]:
    ranked_docs = rerank_documents(docs, rewrite, len(docs))
    selected: list[Document] = []
    seen_sources: set[str] = set()

    # 优先挑选“新面孔”
    for doc in ranked_docs:
        source = str((doc.metadata or{}).get("source",""))
        if source in seen_sources:
            continue
        selected.append(doc)
        seen_sources.add(source)
        if len(selected) == k:
            return selected
        

    # 补齐名额
    for doc in ranked_docs:
        if doc in selected:
            continue 
        selected.append(doc)
        if len(selected) == k:
            break

    return selected


