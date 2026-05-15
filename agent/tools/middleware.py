from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call,before_model,dynamic_prompt,ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.tools.agent_tools import mark_report_mode
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts
#工具调用的包装层
@wrap_tool_call
def monitor_tool(
        #请求的数据封装
        request: ToolCallRequest,
        #执行的函数本身
        handler: Callable[[ToolCallRequest],ToolMessage | Command],
) -> ToolMessage | Command:            
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
     #工具执行的监控    
    logger.info(f"[tool monitor]执行工具:{tool_name}")
    logger.info(f"[tool minitor]传入参数:{tool_args}")

    try:
        result = handler(request)
        logger.info(f"[tool minitor]工具{tool_name}调用成功")

        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True
            mark_report_mode()
            logger.info("[tool] 报告模式可用")
        
        return result
    except Exception as e:
        logger.error(f"工具{tool_name}调用失败:{str(e)}")
        raise e
    
#发送前的最后检阅
@before_model
def log_before_model(
        state: AgentState,          #整个Agent智能体中的状态记录
        runtime: Runtime,           #记录了整个执行过程这种的上下文信息 
):         #在模型执行前输出日志                           #对话全纪录的档案袋
    logger.info(f"[log_before_model]即将调用模型,带有{len(state['messages'])}条消息,报告模式状态:{runtime.context.get('report', False)}")
    latest_message = state["messages"][-1]         
    content = getattr(latest_message, "content", "")#安全获取属性 ,getattr(对象, "属性名", 默认值)  
    if isinstance(content, str):                                                   
                                                             #消息的类型名称 | 消息内容
        logger.debug(f"[log_before_model]{type(latest_message).__name__} | {content.strip()}")
    return None

@dynamic_prompt               #每一次在生成提示词之前,调用此函数
def report_prompt_switch(request:ModelRequest):
    is_report = request.runtime.context.get("report",False)
    if is_report:             #是报告生成场景,返回报告生成提示词内容
        logger.info("[prompt] 转换为报告模式")
        return load_report_prompts()
    
    logger.info("[prompt] 转换为普通模式")
    return load_system_prompts()

