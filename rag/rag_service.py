"""
总结服务类:用户提问 搜索参考资料,将提问和参考资料提交给模型,让模型总结回复
"""
from pathlib import Path
import sys
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser #将AI返回信息转成所需字符串

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger


# def print_prompt(prompt):
#     print("="*20)
#     print(prompt.to_string())
#     print("="*20)
#     print("="*20)
#     return prompt


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model=chat_model
        self.chain=self.prompt_template|self.model|StrOutputParser()
        
    def retriever_docs(self,query:str)->list[Document]:
        docs = self.vector_store.search(query)#开始执行
        logger.info("[rag]query=: %s retrieved_docs= %s",query,len(docs))
        return docs

    @staticmethod#静态方法,只关心做事
    # 数据清洗,将文档提取出来
    def _extract_sources(docs:list[Document]) -> list[str]:
        sources: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source")
            if not source:
                continue
            source_name = Path(str(source)).name#长路径全部切掉,只留文件名
            if source_name not in sources:
                sources.append(source_name)
        return sources
    
    @staticmethod
    # 构建上下文,文书编辑,排版整齐、适合 AI 阅读的参考文稿
    def _build_context(docs:list[Document]) -> str:
        context_parts: list[str] = []
        for idx, doc in enumerate(docs,start = 1):
            context_parts.append(
                f"【参考资料{idx}】:参考资料:{doc.page_content}|参考元数据:{doc.metadata}"
            )
        return "\n".join(context_parts)

    #先检索，再判断，最后生成                  #泛指任何对象
    def rag_summarize(self,query:str)-> dict[str,object]:
        context_docs = self.retriever_docs(query)
        sources = self._extract_sources(context_docs)
        
        # 防止瞎编
        if not context_docs:
            logger.warning("[rag] query=%s no document matched",query)
            return {
                "answer": "知识库里暂时没有检索到直接相关的内容，你可以换个更具体的问题再试试。",
                "sources": [],
            }
        
        answer = self.chain.invoke(
            {
                "input":query,
                "context":self._build_context(context_docs)
            }
        )
        return {
            "answer": answer,
            "sources": sources,
        }

if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪种扫地机器人"))