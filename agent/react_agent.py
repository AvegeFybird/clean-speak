import re
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from model.factory import chat_model

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts

from agent.conversation_context import ConversationContext, ResolveResult 
from agent.tools.agent_tools import(
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_external_record,
    get_runtime_state,
    get_user_id,
    get_user_location,
    get_weather,
    mark_report_mode,
    rag_summarize,
    reset_runtime_state,
)
                                    
from agent.tools.middleware import monitor_tool, log_before_model , report_prompt_switch



class ReactAgent:

    REPORT_STRONG_KEYWORDS = (
        "生成报告",
        "使用报告",
        "本月报告",
        "月度报告",
        "使用记录",
        "生成我的报告",
        "我的使用情况",
        "我的机器人报告",
    )

    action_keywords = ("生成", "查看", "总结", "出一份", "出个")
    report_keywords = ("报告", "月报", "使用情况")
    report_follow_up_keywords = ("上个月", "上上个月", "相比", "对比", "耗材", "清洁效率")
    report_compare_keywords = ("上个月", "上上个月", "相比", "对比")
    
    def __init__(self):#self一直在
        self.conversation_context = ConversationContext(
            window_size=agent_conf.get("conversation_context_window", 5),
            max_session_facts=agent_conf.get("max_session_facts", 12),
            enable_session_facts=agent_conf.get("enable_session_facts", True),
            enable_candidate_facts=agent_conf.get("enable_candidate_facts", True),
            enable_topic_shift_guard=agent_conf.get("enable_topic_shift_guard", True),
        )
        self.agent = create_agent(
            model = chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                   rag_summarize,
                   get_weather,
                   get_user_location,
                   get_user_id,
                   get_current_month,
                   fetch_external_data,
                   fill_context_for_report,
                   ],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
        )
        self.last_run_info = {
            "sources": [],#存放RAG过程中引用的原始文档来源
            "report_mode": False,
            "conversation": {},
        }
        self.last_report_context: dict[str, object] = {}


    @classmethod #类方法,不需要实例化对象就能调用
    def _should_enable_report_mode(cls, query: str) -> bool:
        normalized_query = query.strip()
        if not normalized_query:
            return False
        
        #强匹配
        if any(keyword in normalized_query for keyword in cls.REPORT_STRONG_KEYWORDS):
            return True
        
        if (
            any(keyword in normalized_query for keyword in cls.report_keywords)
            and any(keyword in normalized_query for keyword in cls.report_follow_up_keywords)
        ):
            return True
        
        #弱匹配 
        return (
                any(keyword in normalized_query for keyword in cls.action_keywords) 
            and any(keyword in normalized_query for keyword in cls.report_keywords)
        )



    def execute_stream(self,query:str):
        reset_runtime_state()
        resolve_result = self._resolve_query(query)
        direct_report_answer = self._answer_report_follow_up(query, resolve_result)
        if direct_report_answer:
            self.conversation_context.add_turn(
                raw_query=query,
                resolved_query=resolve_result.resolved_query,
                assistant_response=direct_report_answer,
            )
            self.last_run_info = {
                "sources": [],
                "report_mode": True,
                "conversation": self._resolve_result_to_dict(resolve_result),
            }
            yield direct_report_answer
            return

        effective_query = resolve_result.resolved_query
        if agent_conf.get("enable_resolved_query_log", True):
            logger.info(
                "[agent] raw_query=%s resolved_query=%s follow_up=%s fallback=%s topic_shifted=%s facts=%s",
                resolve_result.raw_query,
                resolve_result.resolved_query,
                resolve_result.is_follow_up,
                resolve_result.fallback_reason,
                resolve_result.topic_shifted,
                resolve_result.session_facts,
            )
        should_enable_report_mode = self._should_enable_report_mode(effective_query)
        if should_enable_report_mode:
            mark_report_mode()
            logger.info("[agent] 由查询路由触发,报告模式预开启")

        input_dict = {
            "messages": [
                {"role": "user","content": effective_query},
            ]
        }
        
        #只吐新增的那一部分
        last_emitted_content = ""
        #第三个参数context就是上下文runtime中的信息,就是我们做提示词切换的标记
        #"value"当前状态的完整副本
        #stream每产生一点新信息,就立刻向外吐出一个数据块
        for chunk in self.agent.stream(
            input_dict, 
            stream_mode="values",
            context={"report": should_enable_report_mode},
        ):
            latest_message = chunk["messages"][-1]
            if not self._is_final_assistant_message(latest_message):
                continue

            content = latest_message.content
                            #是字符串类型否则跳过
            if not isinstance(content,str) or not content:
                continue

            if content.startswith(last_emitted_content):#判断前缀
                delta = content[len(last_emitted_content):]#切片,真正的流式输出
            else:
                delta = content
            
            last_emitted_content = content
            if delta:
                yield delta


        runtime_state = get_runtime_state()
        if runtime_state["report_mode"] and runtime_state.get("last_external_data"):
            self.last_report_context = dict(runtime_state["last_external_data"])
        self.conversation_context.add_turn(
            raw_query=query,
            resolved_query=effective_query,
            assistant_response=last_emitted_content,
        )
        self.last_run_info = {
            "sources": runtime_state["last_rag_sources"],
            "report_mode": runtime_state["report_mode"],
            "conversation": self._resolve_result_to_dict(resolve_result),
        }

    def get_last_run_info(self) -> dict[str, object]:
        return {
            "sources" : list(self.last_run_info["sources"]),
            "report_mode" : bool(self.last_run_info["report_mode"]),
            "conversation": dict(self.last_run_info.get("conversation", {})),
        }
    
    def _resolve_query(self, query: str) -> ResolveResult:
        if not agent_conf.get("enable_conversation_context", True):
            return ResolveResult(
                raw_query=query,
                resolved_query=query,
                is_follow_up=False,
                fallback_reason="conversation_context_disabled",
                reason="conversation context disabled",
            )
        return self.conversation_context.resolve_query(query)
    
    
    def _answer_report_follow_up(self, query: str, resolve_result: ResolveResult) -> str:
        if not self.last_report_context:
            return ""
        if not any(keyword in query for keyword in self.report_compare_keywords):
            return ""
        
        user_id = str(self.last_report_context.get("user_id", ""))
        actual_month = str(str(self.last_report_context.get("actual_month", "")))
        current_record = self.last_report_context.get("record")
        if not user_id or not actual_month or not isinstance(current_record, dict):
            return ""
        
        compare_month = self._resolve_compare_month(query, actual_month)
        if not compare_month:
            return ""
        
        previous_record = get_external_record(user_id, compare_month)
        if not previous_record:
            return f"当前报告实际采用的是 {actual_month}，但没有找到 {compare_month} 的记录，暂时无法做对比。"

        logger.info(
            "[agent] direct report compare raw_query=%s actual_month=%s compare_month=%s",
            resolve_result.raw_query,
            actual_month,
            compare_month,
        )
        return self._format_report_compare(actual_month, current_record, compare_month, previous_record)


    @classmethod
    def _resolve_compare_month(cls, query: str, actual_month: str) -> str | None:
        parsed_actual = cls._parse_year_month(actual_month)
        if not parsed_actual:
            return None
        
        explicit_month = cls._resolve_explicit_compare_month(query, parsed_actual[0])
        if explicit_month:
            return explicit_month
        
        if "上上个月" in query:
            return cls._shift_month(actual_month, -2)
        if "上个月" in query:
            return cls._shift_month(actual_month, -1)
        return None
    @staticmethod
    def _resolve_explicit_compare_month(query: str, default_year: int) -> str | None:
        year_month_match = re.search(r"(?P<year>20\d{2})\s*年\s*(?P<month>1[0-2]|0?[1-9])\s*月", query)
        if year_month_match:
            year = int(year_month_match.group("year"))
            month = int(year_month_match.group("month"))
            return f"{year:04d}-{month:02d}"
        
        # 精准提取 “月份” 数字
        month_match = re.search(r"(?<!\d)(?P<month>1[0-2]|0?[1-9])\s*月(?:份)?", query)
        if month_match:
            month = int(month_match.group("month"))
            return f"{default_year:04d}-{month:02d}"
        
        return None


    @staticmethod
    def _shift_month(month: str, offset: int) -> str | None:
        parsed_month = ReactAgent._parse_year_month(month)
        if not parsed_month:
            return None
        
        year, month_num = parsed_month
        month_index = year *12 + (month_num - 1) + offset
        shifted_year = month_index // 12
        shifted_month = month_index % 12 + 1
        return f"{shifted_year:04d}-{shifted_month:02d}"
        


    @staticmethod
    def _parse_year_month(month: str) -> tuple[int, int] | None:
        try:
            year_text, month_text = month.split("-", maxsplit=1)
            year = int(year_text)
            month_num = int(month_text)
        except ValueError:
            return None
        
        if 1 <= month_num <= 12:
            return year, month_num
        return None

            
    @staticmethod
    def _format_report_compare(
        actual_month:str,
        current_record: dict[str, str],
        previous_month: str,
        previous_record: dict[str, str],
    ) -> str:
        return (
            f"上次报告实际采用的是 `{actual_month}` 的记录；根据你的追问，"
            f"这里对比 `{actual_month}` 和 `{previous_month}`。\n\n"
            f"### {actual_month} vs {previous_month}\n\n"
            f"**清洁效率**\n\n"
            f"- {actual_month}：{current_record.get('效率', '暂无数据')}\n"
            f"- {previous_month}：{previous_record.get('效率', '暂无数据')}\n\n"
            f"**耗材状态**\n\n"
            f"- {actual_month}：{current_record.get('耗材', '暂无数据')}\n"
            f"- {previous_month}：{previous_record.get('耗材', '暂无数据')}\n\n"
            f"**整体对比**\n\n"
            f"- {actual_month}：{current_record.get('对比', '暂无数据')}\n"
            f"- {previous_month}：{previous_record.get('对比', '暂无数据')}\n"
        )


    @staticmethod
    def _resolve_result_to_dict(result: ResolveResult) -> dict[str, object]:
        return {
            "raw_query": result.raw_query,
            "resolved_query": result.resolved_query,
            "is_follow_up": result.is_follow_up,
            "inherited_topic": result.inherited_topic,
            "candidate_facts": list(result.candidate_facts),
            "session_facts": list(result.session_facts),
            "topic_shifted": result.topic_shifted,
            "fallback_reason": result.fallback_reason,
            "reason": result.reason,
        }


    @staticmethod
    def _is_final_assistant_message(message: object) -> bool:
        if not isinstance(message, AIMessage):
            return False
        # 检查是否有工具调用 如果有 tool_calls（非空），说明还在处理中，不是最终回复
        # 如果没有 tool_calls（None 或空），说明是最终回复
        return not bool(getattr(message, "tool_calls", None))

if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的报告"):
        print(chunk,end="",flush=True)