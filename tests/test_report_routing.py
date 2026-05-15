from agent.react_agent import ReactAgent
from agent.conversation_context import ResolveResult
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage



def test_report_intent_keywords_enable_report_mode():
    queries = [
        "生成报告",
        "帮我生成本月报告",
        "查看我的使用情况",
        "给我出一份月报",
    ]
    
    for query in queries:
        assert ReactAgent._should_enable_report_mode(query) is True

    
def test_non_report_query_does_not_enable_report_mode():
    assert ReactAgent._should_enable_report_mode("扫地机器人怎么维护") is False
    assert ReactAgent._should_enable_report_mode("") is False


def test_report_follow_up_query_keeps_report_mode():
    assert ReactAgent._should_enable_report_mode("扫地机器人 报告 和上个月相比呢") is True
    assert ReactAgent._should_enable_report_mode("报告 耗材情况呢") is True


# 验证当用户问"和上个月相比"时，系统能正确使用缓存的上一次报告上下文进行对比
def test_report_compare_follow_up_uses_last_actual_month(monkeypatch):
    agent = ReactAgent()
    agent.last_report_context = {
        "user_id": "1005",
        "actual_month": "2025-12",
        "record": {
            "效率": "当前效率",
            "耗材": "当前耗材",
            "对比": "当前对比",
        },
    }

    monkeypatch.setattr(
        "agent.react_agent.get_external_record",
        lambda user_id, month: {
            "效率": "上月效率",
            "耗材": "上月耗材",
            "对比": "上月对比",
        }
    )

    answer = agent._answer_report_follow_up(
        "和上个月相比呢",
        ResolveResult(
            raw_query="和上个月相比呢",
            resolved_query="扫地机器人 报告 和上个月相比呢",
            is_follow_up=True,
        )
    )

    assert "2025-12" in answer
    assert "2025-11" in answer
    assert "当前效率" in answer
    assert "上月效率" in answer


# 验证当用户明确指定对比月份时，系统能正确解析并获取对应月份的数据
def test_report_compare_follow_up_uses_explicit_month(monkeypatch):
    agent = ReactAgent()
    agent.last_report_context = {
        "user_id": "1005",
        "actual_month": "2025-12",
        "record": {
            "效率": "12月效率",
            "耗材": "12月耗材",
            "对比": "12月对比",
        },
    }

    def fake_get_external_record(user_id, month):
        assert month == "2025-10"
        return {
            "效率": "10月效率",
            "耗材": "10月耗材",
            "对比": "10月对比",
        }
    
    monkeypatch.setattr("agent.react_agent.get_external_record", fake_get_external_record)

    answer = agent._answer_report_follow_up(
        "和10月的相比呢",
        ResolveResult(raw_query="和10月的相比呢", resolved_query="和10月的相比呢", is_follow_up=True),
    )

    assert "2025-12" in answer
    assert "2025-10" in answer
    assert "10月效率" in answer


# 验证当用户问"和上上个月的相比"时，系统能正确使用缓存的上上个月数据进行对比
def test_report_compare_follow_up_uses_month_before_previous(monkeypatch):
    agent = ReactAgent()
    agent.last_report_context = {
        "user_id": "1005",
        "actual_month": "2025-12",
        "record": {
            "效率": "12月效率",
            "耗材": "12月耗材",
            "对比": "12月对比",
        },
    }

    def fake_get_external_record(user_id, month):
        assert month == "2025-10"
        return {
            "效率": "10月效率",
            "耗材": "10月耗材",
            "对比": "10月对比",
        }

    monkeypatch.setattr("agent.react_agent.get_external_record", fake_get_external_record)

    answer = agent._answer_report_follow_up(
        "和上上个月的相比呢",
        ResolveResult(raw_query="和上上个月的相比呢", resolved_query="和上上个月的相比呢", is_follow_up=True),
    )

    assert "2025-10" in answer
    assert "10月效率" in answer


# 验证当用户明确指定对比月份但该月份没有数据时，系统不会自动降级到其他月份，而是直接告诉用户没有找到数据
def test_report_compare_follow_up_does_not_fallback_missing_explicit_month(monkeypatch):
    agent = ReactAgent()
    agent.last_report_context = {
        "user_id": "1005",
        "actual_month": "2025-12",
        "record": {
            "效率": "12月效率",
            "耗材": "12月耗材",
            "对比": "12月对比",
        },
    }

    monkeypatch.setattr("agent.react_agent.get_external_record", lambda user_id, month: None)


    answer = agent._answer_report_follow_up(
        "和9月份相比呢",
        ResolveResult(raw_query="和9月份相比呢", resolved_query="和9月份相比呢", is_follow_up=True),
    )

    assert "2025-09" in answer
    assert "没有找到" in answer
    assert "2025-11" not in answer


# 验证流式输出时，只显示最终的助手回复，不显示中间状态消息（如用户消息、工具消息、工具调用消息)
def test_stream_filter_only_emits_final_ai_messages():
    assert ReactAgent._is_final_assistant_message(HumanMessage(content="用户原问题")) is False
    assert ReactAgent._is_final_assistant_message(ToolMessage(content="工具结果", tool_call_id="1")) is False
    assert ReactAgent._is_final_assistant_message(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_summarize",
                    "args": {"query": "扫地机器人"},
                    "id": "call_1",
                }
            ],
        )
    ) is False
    assert ReactAgent._is_final_assistant_message(AIMessage(content="最终答案")) is True
