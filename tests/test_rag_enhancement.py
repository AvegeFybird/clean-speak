from langchain_core.documents import Document

from rag import rerank as rerank_module
from rag import evaluate_retrieval
from rag.preprocess import clean_text, prepare_documents, split_faq_documents
from rag.query_rewrite import rewrite_query
from rag.rerank import _contains_count, rerank_documents, rerank_documents_with_source_diversity
from rag.rerank import _tokens


# 跳过切分
class DummySplitter:
    def split_documents(self, documents):
        return documents


def test_clean_text_keeps_titles_and_removes_extra_blank_lines():
    text = "一、维护保养\r\n\r\n\r\n  1. 清理滤网\t\t和尘盒  "

    cleaned = clean_text(text)

    assert "一、维护保养" in cleaned
    assert "清理滤网" in cleaned
    assert "\n\n\n" not in cleaned


# 针对 FAQ（常见问题解答），必须保证“问题”与“答案”在语义上不被切断
def test_split_faq_documents_keep_question_and_answer_together():
    doc = Document(
        page_content="问题：吸力变弱怎么办？\n答案：清理尘盒、滤网和主刷。\n\n问题：不出水怎么办？\n答案：检查水箱。",
        metadata={"source": "faq.txt"},
    )

    chunks = split_faq_documents(doc)

    assert len(chunks) == 2
    assert chunks[0].metadata["chunk_type"] == "faq"
    assert chunks[0].metadata["question"] == "吸力变弱怎么办？"
    assert "清理尘盒" in chunks[0].page_content


# 是否能正确地“降级”并使用常规的文本分块方式处理
def test_prepare_documents_falls_back_to_text_splitter_for_normal_text():
    doc = Document(page_content="一、选购指南\n小户型优先关注续航和导航。", metadata={"source": "guide.txt"})

    chunks = prepare_documents([doc], DummySplitter())

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "text"
    assert chunks[0].metadata["question"] == ""
    

def test_rewrite_query_expands_typical_symptom_terms():
    result = rewrite_query("扫地机器人吸力变弱怎么处理")

    assert result.original_query == "扫地机器人吸力变弱怎么处理"
    assert "尘盒" in result.expansion_terms
    assert "滤网" in result.rewritten_query

    
def test_rewrite_query_handles_hair_wording():
    result = rewrite_query("主刷缠了很多头发应该怎么清理")

    assert "毛发" in result.expansion_terms
    assert "防缠绕" in result.rewritten_query


def test_rewrite_query_expands_elder_usage_terms():
    result = rewrite_query("老人用扫地机器人应该优先看什么功能")

    assert "按键简洁" in result.expansion_terms
    assert "操作简单" in result.rewritten_query



def test_rerank_prefers_faq_question_and_expansion_matches():
    rewrite = rewrite_query("吸力变弱怎么处理")
    docs = [
        Document(page_content="这是一段泛泛的说明。", metadata={"source": "a.txt"}),
        Document(
            page_content="答案：清理尘盒、滤网和风道。",
            metadata={
                "source": "faq.txt",
                "chunk_type": "faq",
                "question": "吸力变弱怎么办？",
            },
        ),
    ]

    ranked = rerank_documents(docs, rewrite, 2)

    assert ranked[0].metadata["source"] == "faq.txt"
    

def test_rerank_tokens_filter_generic_question_fragments():
    tokens = _tokens("老人用扫地机器人应该优先看什么功能")

    assert "老人" in tokens
    assert "机器人" not in tokens
    assert "什么功" not in tokens
    assert "应该优" not in tokens


def test_rerank_tokens_fallback_when_stopwords_remove_everything(monkeypatch):
    monkeypatch.setattr(rerank_module, "STOPWORDS", {"abc"})

    assert _tokens("abc") == ["abc"]


def test_rerank_tokens_fallback_uses_raw_tokens(monkeypatch):
    monkeypatch.setattr(rerank_module, "STOPWORDS", {"ab", "bc", "abc"})

    assert _tokens("abc") == ["abc"]

def test_contains_count_deduplicates_and_caps_terms():
    terms = ["abc", " ABC", "def"] + [f"noise_{idx}" for idx in range(80)] + ["overflow"]

    assert _contains_count("abc def overflow", terms) == 2



def test_rerank_keeps_source_diversity_before_duplicates():
    rewrite = rewrite_query("吸力变弱怎么处理")
    docs = [
        Document(page_content="吸力变弱 清理尘盒", metadata={"source": "a.txt"}),
        Document(page_content="吸力变弱 清理滤网", metadata={"source": "a.txt"}),
        Document(page_content="吸力变弱 检查风道", metadata={"source": "b.txt"}),
    ]

    ranked = rerank_documents_with_source_diversity(docs, rewrite, k=2)

    assert [doc.metadata["source"] for doc in ranked] == ["a.txt", "b.txt"]


def test_evaluate_compare_query_keeps_baseline_and_enhanced(monkeypatch):
    def fake_sources(query, k, mode="baseline"):
        if mode == "enhanced":
            return ["维护保养.txt", "故障排除.txt"]
        return ["选购指南.txt", "维护保养.txt"]

    # 把生产环境的检索函数替换成这个“假函数”
    monkeypatch.setattr(evaluate_retrieval, "get_top_k_sources", fake_sources)
    
    result = evaluate_retrieval.evaluate_compare_query(
        {
            "query": "吸力变弱怎么处理",
            "expected_sources": ["故障排除.txt"],
            "category": "故障排除",
            "reason": "验证增强检索能召回故障来源",
        },
        k=2,
    )

    assert result.baseline.recall_at_k == 0.0
    assert result.enhanced.recall_at_k == 1.0
    assert "尘盒" in result.enhanced.rewritten_query
