# 洁语 — 扫地机器人智能客服

基于 LangChain ReAct Agent + Streamlit 构建的智能客服系统，支持知识库问答、实时天气建议和个人使用报告生成。

## 功能

| 能力 | 说明 |
|---|---|
| 知识库问答 | 基于 RAG 从产品文档中检索答案，解答选购、故障排查、维护保养等问题 |
| 天气建议 | 调用 Open-Meteo 免费天气 API，结合天气条件给出扫拖建议 |
| 使用报告 | 拉取用户月度使用记录，生成清洁效率、耗材状态的汇总报告 |
| 报告对比 | 支持追问"上个月"、"上上个月"或指定月份的跨月对比 |
| 多轮对话 | 上下文感知，支持指代消解、话题追踪和 session facts 累积 |
| 流式输出 | 实时展示思考过程与最终回答 |

## 技术栈

- **Agent 框架**: LangChain (`create_agent`, ReAct 模式)
- **前端**: Streamlit
- **LLM**: 通义千问 (DashScope) — `qwen3.6-flash`
- **Embedding**: DashScope `text-embedding-v4`
- **向量库**: ChromaDB
- **天气数据**: Open-Meteo (免费，无需 API Key)
- **地理编码**: Open-Meteo Geocoding API

## 项目结构

```
├── agent/                  # Agent 主体与工具
│   ├── react_agent.py      # ReAct Agent 核心，报告模式路由
│   ├── conversation_context.py  # 多轮对话上下文管理
│   └── tools/
│       ├── agent_tools.py  # Tool 函数 (RAG/天气/报告)
│       └── middleware.py   # 中间件 (工具监控/提示词切换)
├── rag/                    # RAG 检索增强生成
│   ├── vector_store.py     # 向量库入库
│   ├── rag_service.py      # 检索 + LLM 总结
│   ├── query_rewrite.py    # 查询改写
│   ├── rerank.py           # 重排序
│   └── preprocess.py       # 文档预处理
├── model/
│   └── factory.py          # 模型工厂 (ChatTongyi / DashScopeEmbeddings)
├── config/                 # YAML 配置文件
├── prompts/                # 提示词模板
├── utils/                  # 工具函数 (路径/日志/文件)
├── data/                   # 知识库文档 & 外部用户记录 CSV
├── tests/                  # 测试
└── app.py                  # Streamlit 入口
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
pip install streamlit pyyaml python-dotenv pypdf
```

### 2. 配置

```bash
# 复制环境变量模板并填入你的 API Key
cp .env.example .env
```

编辑 `.env`：

```env
DASHSCOPE_API_KEY=你的通义千问API密钥
AGENT_USER_ID=1005
AGENT_USER_CITY="东莞"
```

### 3. 准备数据

- **知识库文档**: 将产品手册、FAQ 等 PDF/TXT 文件放入 `data/` 目录
- **用户记录**: 在 `data/external/records.csv` 中维护用户月度使用数据

### 4. 构建向量库

```bash
python rag/vector_store.py
```

### 5. 启动

```bash
streamlit run app.py
```

## 配置说明

| 文件 | 用途 |
|---|---|
| `config/agent.yml` | 对话上下文窗口、session facts 开关 |
| `config/chroma.yml` | 向量库参数、分块策略、检索 Top-K |
| `config/rag.yml` | 模型名称 (`chat_model_name`, `embedding_model_name`) |
| `config/prompts.yml` | 提示词版本与路径映射 |

## 依赖

核心依赖见 [requirements.txt](requirements.txt)，运行时还需：

- `streamlit` — Web UI
- `pyyaml` — 配置解析
- `python-dotenv` — 环境变量加载
- `pypdf` — PDF 文档解析
- `huggingface_hub` — 模型下载（可选）
