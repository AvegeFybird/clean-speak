import os
import time

import streamlit as st

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import get_runtime_banding_diagnostics
from utils.config_handler import agent_conf,chroma_conf
from utils.path_tool import get_abs_path

st.set_page_config(page_title="洁语智能客服",page_icon = "🤖", layout="wide")

#显示思考过程开关
SHOW_THINKING_STREAM = True

THINKING_OPEN_TAG = "<thinking>"
THINKING_CLOSE_TAG = "</thinking>"
FINAL_OPEN_TAG = "<final>"
FINAL_CLOSE_TAG = "</final>"

EXAMPLE_PROMPTS = [
    {
        "title": "知识库问答",
        "tag": "RAG",
        "description": "适合快速体验维护保养、故障排查和选购建议。",
        "prompt": "我需要怎么维护保养我购买的扫地机器人？",
    },
    {
        "title": "天气查询",
        "tag": "Tool",
        "description": "结合天气条件，判断今天是否适合更高频地拖地。",
        "prompt": "东莞今天的天气适合让扫拖机器人多拖几次吗？",
    },
    {
        "title": "个人使用报告",
        "tag": "Report",
        "description": "触发报告模式，查看本月使用情况和养护建议。",
        "prompt": "帮我生成一份我本月的扫地机器人使用报告。",
    },
]
#系统启动自检函数
def build_startup_checks() -> list[str]:
    path_warnings: list[str] = []
    missing_bindings: list[str] = []
    invalid_bindings: list[str] = []

    required_paths = [
        get_abs_path("config/agent.yml"),
        get_abs_path("config/chroma.yml"),
        get_abs_path("config/prompts.yml"),
        get_abs_path("config/rag.yml"),
        get_abs_path(chroma_conf["data_path"]),
        get_abs_path(agent_conf["external_data_path"]),
    ]

    for path in required_paths:
        if not os.path.exists(path):
            path_warnings.append(f"缺少运行所需文件或目录：`{path}`")
    
    if not os.environ.get("DASHSCOPE_API_KEY"):
       missing_bindings.append("环境变量 `DASHSCOPE_API_KEY` 未配置，模型调用时可能失败。")

    binding_info = {
        "user_id": "",
        "city": "",
        "default_month": "",
    }
    available_months = []

    try:
        binding_diagnostics = get_runtime_banding_diagnostics()
        binding_info = binding_diagnostics["binding"]
        available_months = binding_diagnostics["available_months"]
        missing_bindings = binding_diagnostics["missing"]
        invalid_bindings = binding_diagnostics["invalid"]
    except Exception as exc:
        invalid_bindings.append(f"绑定数据校验失败：{exc}")

    return {
        "path_warnings": path_warnings,
        "missing_bindings": missing_bindings,
        "invalid_bindings": invalid_bindings,
        "binding_info": binding_info,
        "available_months": available_months,
    }


def extract_tagged_content(raw_text: str,open_tag: str,close_tag: str) -> str:
    """
    从原始文本中提取指定标签内的内容。
    参数：
    - raw_text: 原始文本。
    - open_tag: 标签的开始标记。
    - close_tag: 标签的结束标记。
    返回：
    - 提取到的内容。
    """
    start = raw_text.find(open_tag)
    if start == -1:
        return ""
    
    content_start = start + len(open_tag)
    end = raw_text.find(close_tag, content_start)
    if end == -1:
        return raw_text[content_start:]
    return raw_text[content_start:end]

def strip_tagged_content(raw_text: str) -> str:
    """
    从原始文本中删除所有标签。
    """
    cleaned = raw_text
    for tag in [THINKING_OPEN_TAG, THINKING_CLOSE_TAG, FINAL_OPEN_TAG, FINAL_CLOSE_TAG]:
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


def parse_stream_sections(raw_text: str) -> tuple[str, str]:
    """
    从原始文本中提取思考过程和最终结果。
    参数：
    - raw_text: 原始文本。
    返回：
    - 思考过程和最终结果。
    """
    thinking_text = extract_tagged_content(raw_text,THINKING_OPEN_TAG,THINKING_CLOSE_TAG).strip()
    final_text = extract_tagged_content(raw_text,FINAL_OPEN_TAG,FINAL_CLOSE_TAG)

    if not final_text and FINAL_OPEN_TAG not in raw_text:
        # 在“没有 final 标签”的兜底情况下，把清洗后的内容放到第一个位置，第二个位置留空,兜底逻辑
        fallback_text = strip_tagged_content(raw_text)
        return fallback_text, ""
    
    return thinking_text, final_text

def build_stream_preview(thinking_text: str,final_text: str) -> str:
    """
    生成预览文本，用于显示在流式响应中。
    参数：
    - thinking_text: 思考过程。
    - final_text: 最终结果。
    返回：
    - 预览文本。
    """
    blocks: list[str] = []
    if SHOW_THINKING_STREAM and thinking_text:
        blocks.append(f"#### 思考过程\n{thinking_text}")
    if final_text:
        blocks.append(f"#### 最终结果\n{final_text}")
    return "\n\n".join(blocks).strip() # 块与块之间空一行

#渲染 AI 助手的回复消息
def render_assistant_message(message: dict) -> None:
    with st.chat_message("assistant"):
        if message.get("report_mode"):
            st.markdown("### 使用报告")
            st.markdown(message["content"])
        else:
            st.write(message["content"])

        sources = message.get("sources") or []
        if sources:
            st.caption("参考来源" + "、".join(sources))



def stream_response(agent: ReactAgent, prompt: str) -> dict:
    raw_response = ""
    with st.spinner("智能客服思考中..."):
        response_stream = agent.execute_stream(prompt)  

        with st.chat_message("assistant"):
            # 先留一个空位置，后续流式输出时不停覆盖更新
            stream_placeholder = st.empty()

            for chunk in response_stream:
                raw_response += chunk
                # 把当前已经累计的完整文本解析成两部分
                think_text, final_text = parse_stream_sections(raw_response)
                preview_text = build_stream_preview(think_text, final_text)
                if preview_text:
                    stream_placeholder.markdown(preview_text)
                time.sleep(0.01)

            # 获取当前运行信息
            run_info = agent.get_last_run_info()
            _, final_text = parse_stream_sections(raw_response)
            # 最终结果为空时，用清洗后的文本填充
            final_content = final_text or strip_tagged_content(raw_response)
                
            # 把前面流式展示用的占位区域清空
            stream_placeholder.empty()
            if run_info["report_mode"]:
                st.markdown("### 使用报告")
                st.markdown(final_content)
            else:
                st.write(final_content)

            sources = run_info["sources"]
            if sources:
                st.caption("参考来源：" + "、".join(sources))
    return {
        "role": "assistant",
        "content": final_content.strip(),
        "sources": list(run_info["sources"]),
        "report_mode": bool(run_info["report_mode"]),
    }

    
def queue_prompt(prompt: str) -> None:
    st.session_state["pending_prompt"] = prompt

#UI样式
st.markdown(
    """
    <style>
    .hero-wrap {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at top right, rgba(255, 200, 120, 0.35), transparent 28%),
            linear-gradient(135deg, #f7f1e3 0%, #eef5ea 48%, #e2f0f7 100%);
        border: 1px solid rgba(120, 140, 120, 0.18);
        margin-bottom: 1.25rem;
    }
    .hero-kicker {
        display: inline-block;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        background: rgba(34, 84, 61, 0.08);
        color: #22543d;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        margin: 0;
        color: #163326;
        font-size: 2.2rem;
        line-height: 1.15;
    }
    .hero-subtitle {
        margin: 0.8rem 0 0 0;
        color: #355646;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 760px;
    }
    .section-title {
        margin-top: 0.3rem;
        margin-bottom: 0.15rem;
        color: #17392b;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .section-desc {
        color: #587062;
        margin-bottom: 1rem;
    }
    .example-card {
        min-height: 220px;
        padding: 1.1rem 1rem 0.9rem 1rem;
        border-radius: 22px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbf8 100%);
        border: 1px solid rgba(86, 110, 95, 0.14);
        box-shadow: 0 14px 34px rgba(24, 42, 33, 0.06);
        margin-bottom: 0.75rem;
    }
    .example-tag {
        display: inline-block;
        padding: 0.2rem 0.58rem;
        border-radius: 999px;
        background: #eef4ea;
        color: #4a6a54;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .example-title {
        color: #1f3c2e;
        font-size: 1.16rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }
    .example-description {
        color: #5e7468;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0.9rem;
        min-height: 3rem;
    }
    .example-label {
        color: #5a695f;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    .example-prompt {
        color: #233a2f;
        background: #f3f7f2;
        border-radius: 14px;
        padding: 0.85rem 0.9rem;
        font-size: 0.95rem;
        line-height: 1.55;
        border: 1px solid rgba(86, 110, 95, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-kicker">Smart Cleaning Assistant</div>
        <h1 class="hero-title">洁语机器人智能客服</h1>
        <p class="hero-subtitle">
            快速解答选购、故障排查、维护保养与使用问题。
            你可以直接提问，也可以先从下面的示例卡片开始体验。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

checks = build_startup_checks()
with st.sidebar:
    binding_info = checks["binding_info"]
    available_months = checks["available_months"]
    month_text = "、".join(available_months) if available_months else "暂无可查询月份"

    st.subheader("当前用户信息")
    st.caption(
        "用户 ID 和城市来自环境变量，默认月份使用系统当前月份。"
    )
    st.caption(f"用户 ID：`{binding_info['user_id'] or '未配置'}`")
    st.caption(f"城市：`{binding_info['city'] or '未配置'}`")
    st.caption(f"默认月份：`{binding_info['default_month'] or '未知'}`")
    st.caption(f"可查询月份：{month_text}")

    st.divider()
    st.subheader("快捷操作")
    if st.button("生成我的使用报告",use_container_width = True):
        queue_prompt("帮我生成一份我本月的扫地机器人使用报告。")
    if st.button("清空会话",use_container_width= True):
        st.session_state["messages"] = []
        st.session_state.pop("pending_prompt",None)
        st.rerun()
    
    if checks["path_warnings"] or checks["missing_bindings"] or checks["invalid_bindings"]:
        st.divider()
        st.subheader("启动提醒")
        for item in checks["path_warnings"]:
            st.warning(item)
        for item in checks["missing_bindings"]:
            st.warning(f"缺少配置: {item}")
        for item in checks["invalid_bindings"]:
            st.warning(f"配置无效: {item}")

st.info(
    "若该月份在记录中不存在，系统会自动回退到最近可查月份。"
)

st.markdown('<div class="section_title">示例问题</div>',unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">每张卡片都会明确展示将要发送的问题，点一下就能直接体验对应能力。</div>',
    unsafe_allow_html=True,
)

example_columns = st.columns(3, gap="large")
for column, item in zip(example_columns, EXAMPLE_PROMPTS):
    with column:
        st.markdown(
            f"""
            <div class="example-card">
                <div class="example-tag">{item["tag"]}</div>
                <div class="example-title">{item["title"]}</div>
                <div class="example-description">{item["description"]}</div>
                <div class="example-label">将发送的问题</div>
                <div class="example-prompt">{item["prompt"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("使用这个示例",key=f'example_{item["title"]}',use_container_width=True):
            queue_prompt(item["prompt"])

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        render_assistant_message(message)
    
input_prompt = st.chat_input("请输入您的问题")
#用户输入提示词
prompt = st.session_state.pop("pending_prompt", None) or input_prompt

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    assistant_message = stream_response(st.session_state["agent"], prompt)
    st.session_state["messages"].append(assistant_message)
    st.rerun()


   
