from dataclasses import dataclass,field


FOLLOW_UP_MARKERS = (
    "那",
    "这个",
    "它",
    "还有",
    "如果",
    "多久",
    "怎么处理",
    "怎么办",
    "怎么清洗",
    "怎么维护",
    "怎么保养",
    "注意什么",
    "比呢",
    "对比",
)

TOPIC_SHIFT_MARKERS = (
    "天气",
    "气温",
    "下雨",
    "生成报告",
    "使用报告",
    "本月报告",
    "月度报告",
)


REPORT_MARKERS =(
    "报告",
    "本月",
    "上个月",
    "耗材",
    "清洁效率",
    "使用记录",
)


OTHER_PERSON_MARKERS = (
    "邻居",
    "朋友",
    "别人",
    "同事",
    "亲戚",
)


# 会话事实别名
FACT_ALIASES = {
    "有宠物": ("我家有宠物", "家里有宠物", "养宠物", "有猫", "有狗", "宠物家庭"),
    "木地板": ("我家是木地板", "家里是木地板", "木地板"),
    "有地毯": ("我家有地毯", "家里有地毯", "地毯"),
    "老人使用": ("老人用", "给老人用", "买给老人", "老人使用"),
    "自动集尘": ("自动集尘", "集尘款", "集尘袋"),
    "防缠绕": ("防缠绕", "毛发缠绕", "头发缠绕"),
}


# 短期话题别名
TOPIC_ALIASES = {
    "滤网": ("滤网", "hepa"),
    "边刷": ("边刷",),
    "主刷": ("主刷", "滚刷"),
    "维护保养": ("维护", "保养", "清理", "清洁", "更换", "多久换", "多久清理"),
    "故障排查": ("坏了", "故障", "异常", "无法", "不能", "不动", "报错", "怎么办"),
    "网络连接": ("wifi", "wi-fi", "联网", "连接网络"),
    "水箱": ("水箱", "不出水", "出水"),
    "吸力变弱": ("吸力变弱", "吸力下降", "吸不干净"),
    "自动集尘": ("自动集尘", "集尘袋"),
    "宠物家庭": ("宠物", "猫", "狗", "毛发"),
    "木地板": ("木地板",),
    "报告": REPORT_MARKERS,
}


@dataclass
class ConversationTurn:
    raw_query: str
    resolved_query: str
    assistant_response: str = ""


@dataclass
class ResolveResult:
    raw_query: str
    resolved_query: str 
    is_follow_up: bool
    inherited_topic: str = ""
    candidate_facts: list[str] = field(default_factory=list)
    session_facts: list[str] = field(default_factory=list)
    topic_shifted: bool = False
    fallback_reason: str = ""
    reason: str = ""
    

class ConversationContext:
    def __init__(
            self,
            window_size: int = 5,
            max_session_facts: int = 12,
            enable_session_facts: bool = True,
            enable_candidate_facts: bool = True,
            enable_topic_shift_guard: bool = True,
    ):
        self.window_size = max(1,int(window_size))
        self.max_session_facts = max(1,int(max_session_facts))
        self.enable_session_facts = enable_session_facts
        self.enable_candidate_facts = enable_candidate_facts
        self.enable_topic_shift_guard = enable_topic_shift_guard
        self.turns: list[ConversationTurn] = []
        self.candidate_fact_counts: dict[str, int] = {}
        self.session_facts: list[str] = []

    def resolve_query(self, raw_query: str) -> ResolveResult:
        query = raw_query.strip()
        if not  query:
            return ResolveResult(
                raw_query=raw_query,
                resolved_query=raw_query,
                is_follow_up=False,
                fallback_reason="empty_query",
                reason="empty query falls back to raw query"
            )

        topic_shifted = self._is_topic_shift(query)
        if topic_shifted:
            return self._fallback(query, "topic_shift", topic_shifted=True)
        
        is_follow_up = self._is_follow_up(query)
        if self._has_clear_topic(query) and not is_follow_up:
            return self._fallback(query, "complete_query")

        if not is_follow_up:
            return self._fallback(query, "not_follow_up")

        inherited_topic = self._build_inherited_topic(query)
        comfirmed_facts = self._facts_for_query(query)

        if not inherited_topic and not comfirmed_facts:
            return self._fallback(query, "no_reliable_context", is_follow_up=True)

        resolved_query = self._build_resolved_query(query, inherited_topic, comfirmed_facts)
        if len(resolved_query) < len(query) or resolved_query == query:
            return self._fallback(query, "invalid_resolved_query", is_follow_up=True)

        return ResolveResult(
            raw_query=raw_query,
            resolved_query=resolved_query,
            is_follow_up=True,
            inherited_topic=inherited_topic,
            candidate_facts = self._candidate_facts(),
            session_facts=self.session_facts,
            reason="follow-up resolved with reliable context"
        )

    # 记录对话轮次
    def add_turn(self, raw_query: str, resolved_query: str, assistant_response: str = "") -> None: 
        #  从用户输入和助手回复中提取事实
        self._update_facts_from_user(raw_query)
        self._update_facts_from_assistant(assistant_response)

        # 创建对话轮次对象并添加到历史列表
        self.turns.append(
            ConversationTurn(
                raw_query=raw_query.strip(),
                resolved_query=resolved_query.strip(),
                assistant_response=assistant_response.strip(),
            )
        )
        # 保持历史记录不超过窗口大小
        if len(self.turns) > self.window_size:
            self.turns = self.turns[-self.window_size:]
    
    # 返回降级结果
    def _fallback(
            self,
            query: str,
            fallback_reason: str,
            is_follow_up: bool = False,
            topic_shifted: bool = False,
        ) -> ResolveResult:
        return ResolveResult(
            raw_query=query,
            resolved_query=query,
            is_follow_up=is_follow_up,
            candidate_facts=self._candidate_facts(),
            session_facts=list(self.session_facts),
            topic_shifted=topic_shifted,
            fallback_reason=fallback_reason,
            reason="resolver fell back to raw query",
        )

    def _is_follow_up(self, query: str) -> bool:
        compact_query = query.strip()
        if len(compact_query) <= 12 and any(marker in compact_query for marker in FOLLOW_UP_MARKERS):
            return True
        return any(marker in compact_query for marker in ("那", "这个", "它", "还有"))

    def _is_topic_shift(self, query: str) -> bool:
        if not self.enable_topic_shift_guard:
            return False
        return any(marker in query for marker in TOPIC_SHIFT_MARKERS) and not self._is_report_follow_up(query) 

    def _is_report_follow_up(self, query: str) -> bool:
        return "报告" in self._latest_topic() and any(marker in query for marker in REPORT_MARKERS)

    def _has_clear_topic(self, query: str) -> bool:
        return self._extract_topic(query) != ""
    
    # 获取历史最新话题
    def _latest_topic(self) -> str:
        for turn in reversed(self.turns):
            if self._mentions_other_person(turn.raw_query):
                continue
            topic = self._extract_topic(turn.raw_query) or self._extract_topic(turn.resolved_query)
            if  topic:
                return topic
        return ""
    
    # 构建继承话题
    def _build_inherited_topic(self, query: str) -> str:
        current_topic = self._extract_topic(query)
        latest_topic = self._latest_topic()
        should_merge = (
            latest_topic in ("维护保养", "故障排查")
            or any(marker in query for marker in  ("如果", "家里", "宠物"))
        )
        if current_topic and latest_topic and current_topic != latest_topic and should_merge:
            return " ".join(self._deduplicate([current_topic, latest_topic]))
        return current_topic or latest_topic


    # 从文本中提取话题
    @staticmethod
    def _extract_topic(text: str) -> str:
        lowered = text.lower()
        for topic, aliases in TOPIC_ALIASES.items():
            if any(alias.lower() in lowered for alias in aliases):
                return topic
        return ""
    
    # 获取当前查询可用的会话事实
    def _facts_for_query(self,  query: str) -> list[str]:
        if not self.enable_session_facts:
            return []
        if self._is_topic_shift(query):
            return []
        return list(self.session_facts)
    
    
    # 构建解析后的完整查询
    def _build_resolved_query(self, query: str, inherited_topic: str, facts: list[str]) -> str:
        parts = ["扫地机器人"]
        if facts:
            parts.extend(facts)
        if inherited_topic:
            parts.append(inherited_topic)
        parts.append(query)
        return " ".join(self._deduplicate(parts))
    
    # 从用户输入中提取并更新会话事实
    def _update_facts_from_user(self, text: str) -> None:
        if not self.enable_session_facts or not text.strip(): 
            return 
        for  fact, aliases in FACT_ALIASES.items():
            if not any(alias.lower() in text.lower() for alias in aliases):
                continue
            if self._mentions_other_person(text):
                continue
            if self._has_strong_ownership(text):
                self._confirm_fact(fact)
                continue
            if self.enable_candidate_facts:
                self.candidate_fact_counts[fact] = self.candidate_fact_counts.get(fact, 0) + 1
                if self.candidate_fact_counts[fact] >= 2:
                    self._confirm_fact(fact)


    def _update_facts_from_assistant(self, text: str) -> None:
        if not self.enable_session_facts or not text.strip(): 
            return
        for fact in self._candidate_facts():
            aliases = FACT_ALIASES.get(fact, ())
            if fact in text or any (alias in text for alias in aliases):
                self._confirm_fact(fact)

    
    def _confirm_fact(self, fact: str) -> None:
        if fact  in self.session_facts:
            return
        self.session_facts.append(fact)
        if len(self.session_facts) > self.max_session_facts:
            self.session_facts = self.session_facts[-self.max_session_facts:]

    def _candidate_facts(self) -> list[str]:
        if not self.enable_candidate_facts:
            return []
        return [
            fact for fact in self.candidate_fact_counts
            if fact not in self.session_facts
        ]

    
    # 过滤他人相关内容
    @staticmethod
    def _mentions_other_person(text: str) -> bool:
        return any(marker in text for marker in OTHER_PERSON_MARKERS)

    # 识别明确的所有权表述
    @staticmethod
    def _has_strong_ownership(text: str) -> bool:
        return any(marker in text for marker in ("我家", "我的", "我想", "买给"))
    
    # 去重
    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        deduplicated: list[str] = []
        for value in values:
            if value and value not in deduplicated:
                deduplicated.append(value)
        return deduplicated


            