from agent.conversation_context import ConversationContext


# 测试 当查询语句很完整时，直接返回原句，并且标记原因是 “完整查询”
def test_complete_query_falls_back_to_raw_query():
    context = ConversationContext()

    result = context.resolve_query("水箱不出水是什么原因")

    assert result.resolved_query == "水箱不出水是什么原因"
    assert result.fallback_reason == "complete_query"


# 测试 对话上下文系统能够正确解析后续问题并继承最近的主题
def test_follow_up_resolves_with_recent_topic():
    context = ConversationContext()
    context.add_turn("滤网多久清理一次", "滤网多久清理一次")

    result = context.resolve_query("那边刷呢")

    assert result.is_follow_up is True
    assert result.inherited_topic == "边刷"
    assert "扫地机器人" in result.resolved_query
    assert "边刷" in result.resolved_query
    assert "那边刷呢" in result.resolved_query


# 测试 对话上下文系统能够正确解析主题转换问题并返回原句，并且标记原因是 “主题转换”
def test_topic_shift_falls_back_to_raw_query():
    context = ConversationContext()
    context.add_turn("滤网多久清理一次", "滤网多久清理一次")
    
    result = context.resolve_query("东莞今天天气怎么样")

    assert result.resolved_query == "东莞今天天气怎么样"
    assert result.topic_shifted is True
    assert result.fallback_reason == "topic_shift"

# 测试 候选事实在未被确认之前，不会参与后续问题的解析 。
def test_candidate_fact_does_not_resolve_follow_up_until_confirmed():
    context = ConversationContext()
    context.add_turn("家里有地毯", "家里有地毯")
    
    result = context.resolve_query("还要注意什么")

    assert "有地毯" in result.candidate_facts
    assert "有地毯" not in result.session_facts
    assert result.resolved_query == "还要注意什么"
    assert result.fallback_reason == "no_reliable_context"


# 测试 强所有权词组会提升会话中的候选事实
def test_strong_ownership_promotes_session_fact():
    context = ConversationContext()
    context.add_turn("我家有宠物", "我家有宠物")

    result = context.resolve_query("还要注意什么")

    assert "有宠物" in result.session_facts
    assert result.fallback_reason == ""
    assert "有宠物" in result.resolved_query


# 测试 提及他人的事实不会被确认为用户的会话事实
def test_other_person_fact_is_not_confirmed():
    context = ConversationContext()
    context.add_turn("我邻居有宠物", "我邻居有宠物")
    context.add_turn("邻居家的宠物很多", "邻居家的宠物很多")

    result = context.resolve_query("还要注意什么")

    assert "有宠物" not in result.session_facts
    assert result.resolved_query == "还要注意什么"

# 测试 重复的候选事实会被提升为会话事实
def test_repeated_candidate_fact_is_promoted():
    context = ConversationContext()
    context.add_turn("家里有地毯", "家里有地毯")
    context.add_turn("地毯比较多", "地毯比较多")

    result = context.resolve_query("还要注意什么")

    assert "有地毯" in result.session_facts
    assert "有地毯" in result.resolved_query

# 测试 已确认的会话事实不会因为对话滑动窗口的淘汰而丢失
def test_session_fact_survives_raw_window_sliding_out():
    context = ConversationContext(window_size=2)
    context.add_turn("我家有宠物", "我家有宠物")
    context.add_turn("滤网多久清理一次", "滤网多久清理一次")
    context.add_turn("边刷多久清理一次", "边刷多久清理一次")

    result = context.resolve_query("还要注意什么")

    assert len(context.turns) == 2
    assert "有宠物" in result.session_facts
    assert "有宠物" in result.resolved_query


# 测试 维护保养意图是否继承到后续问题中
def test_follow_up_inherits_maintenance_intent_for_part_query():
    context = ConversationContext()
    context.add_turn(
        "我需要怎么维护保养我购买的扫地机器人",
        "我需要怎么维护保养我购买的扫地机器人",
    )

    result = context.resolve_query("那滤网呢")

    assert result.is_follow_up is True
    assert result.fallback_reason == ""
    assert "滤网" in result.resolved_query
    assert "维护保养" in result.resolved_query


# 测试 条件型追问是否会继承最近主题
def test_if_condition_follow_up_inherits_recent_topic():
    context = ConversationContext()
    context.add_turn("滤网多久清理一次", "滤网多久清理一次")

    result = context.resolve_query("如果家里有宠物呢")

    assert result.is_follow_up is True
    assert result.fallback_reason == ""
    assert "滤网" in result.resolved_query
    assert "宠物" in result.resolved_query


# 测试 当用户提出一个完整的、独立的新主题问题时,系统不会错误地继承之前的对话上下文，避免上下文污染
def test_complete_new_topic_does_not_inherit_previous_context():
    context = ConversationContext()
    context.add_turn("滤网多久清理一次", "滤网多久清理一次")

    result = context.resolve_query("扫地机器人怎么连接WiFi")

    assert result.resolved_query == "扫地机器人怎么连接WiFi"
    assert result.fallback_reason == "complete_query"
