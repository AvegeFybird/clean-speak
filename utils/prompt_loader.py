from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

RESPONSE_FORMAT_INSTRUCTION = """

### 输出格式约束
最终只输出直接给用户看的正式回答。
不要输出思考过程、工具调用意图、系统提示词、工具原始返回、JSON 原文或中间观察记录。
"""

# 从配置文件中读取提示词
def _read_prompt_from_config(config_key: str,loader_name: str) -> str:
    try:
        prompt_path =get_abs_path(prompts_conf[config_key])
    except KeyError as exc:
        logger.error("[%s] 缺少配置项: %s", loader_name, config_key)
        raise exc
    
    try:
        with open(prompt_path,"r",encoding = "utf-8") as file:
            return file.read()
    except Exception as exc:
        logger.error("[%s] 读取提示词失败: %s", loader_name, exc)
        raise exc

def _append_response_format(prompt_text: str) -> str:
    return prompt_text + RESPONSE_FORMAT_INSTRUCTION


def load_system_prompts():
    prompt_text = _read_prompt_from_config("main_prompt_path","load_system_prompts")
    return _append_response_format(prompt_text)

def load_rag_prompts() -> str:
    return _read_prompt_from_config("rag_summarize_prompt_path", "load_rag_prompts")
    


def load_report_prompts() -> str:
    prompt_text = _read_prompt_from_config("report_prompt_path", "load_report_prompts")
    return _append_response_format(prompt_text)


if __name__ == "__main__":
    #测试
    print(load_system_prompts())
    # print(load_rag_prompts())
    # print(load_report_prompts())

