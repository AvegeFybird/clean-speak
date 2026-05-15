import re

from langchain_core.documents import Document


QUESTION_PATTERN = re.compile(r"^\s*(?:Q|问题|问)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)# 匹配问题
ANSWER_PATTERN = re.compile(r"^\s*(?:A|答案|答)\s*[:：]\s*(.*)$", re.IGNORECASE)# 匹配答案
SECTION_PATTERN = re.compile(r"^\s*(?:#{1,6}\s*)?(\d+(?:\.\d+)*[、.．]\s*.+|[一二三四五六七八九十]+[、.．]\s*.+)$")# 匹配章节标题


# 文本清洗
def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")# 换行符统一为\n
    normalized = re.sub(r"[\t\u3000]+", " ",normalized)# 将制表符和中文全角空格替换为单个普通的半角空格
    normalized = re.sub(r"[ ]{2,}", " ", normalized)# 将多个空格替换为一个空格
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)# 将多个换行符替换为两个换行符,即一个空行
    return normalized.strip()


# 获得当前章节标题
def _current_section(lines: list[str]) -> str:
    for line in reversed(lines):
        match = SECTION_PATTERN.match(line)
        if match:
            return match.group(1).strip() 
    return ""

# 识别问题 + 答案，切成 FAQ
def split_faq_documents(doc: Document) -> list[Document]:
    text = clean_text(doc.page_content)
    if not text:
        return []   
    
    # 拆分、去空格、删空行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[Document] = []
    prefix_lines: list[str] = [] # 记录背景信息
    current_question = ""
    current_answer_lines: list[str] = []

    def flush() -> None:
        nonlocal current_question, current_answer_lines
        if not current_question:
            return
        page_content = clean_text(
            f"问题: {current_question}\n答案:{' '.join(current_answer_lines).strip()}"
        )
        metadata = {
            **doc.metadata,
            "chunk_type": "faq",
            "section_title": _current_section(prefix_lines),
            "question": current_question,
        }
        # 添加FAQ
        chunks.append(Document(page_content=page_content, metadata=metadata))
        # 清空当前缓存
        current_question = ""
        current_answer_lines = []

    for line in lines:
        question_match = QUESTION_PATTERN.match(line)
        if question_match:
            flush()
            current_question = question_match.group(1).strip()
            continue

        if current_question:
            answer_match = ANSWER_PATTERN.match(line)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    current_answer_lines.append(answer_text)
            else:
                current_answer_lines.append(line)
        
        prefix_lines.append(line)

    # 处理最后一条
    flush()
    return chunks


def prepare_documents(documents: list[Document],splitter) -> list[Document]:
    prepared: list[Document] = []
    for doc in documents:
        cleaned_doc = Document(
            page_content = clean_text(doc.page_content),
            metadata = {**doc.metadata, "chunk_type": "text"}
        )
        faq_docs = split_faq_documents(cleaned_doc)
        if faq_docs:
            # 一个一个添加FAQ
            prepared.extend(faq_docs)
            continue
            
        # 处理普通文本
        split_docs = splitter.split_documents([cleaned_doc])
        for split_doc in split_docs:
            split_doc.metadata.setdefault("chunk_type", "text")
            split_doc.metadata.setdefault("section_title", "")
            split_doc.metadata.setdefault("question", "")
        prepared.extend(split_docs)
    
    return prepared









