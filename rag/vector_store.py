import os
import shutil

#先尽量关闭Chroma遥测噪音
os.environ.setdefault("ANONYMIZED_TELEMETRY","False")

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from utils.config_handler import chroma_conf
from model.factory import embed_model
from rag.preprocess import prepare_documents
from rag.query_rewrite import RewriteResult, rewrite_query
from rag.rerank import rerank_documents_with_source_diversity
from utils.path_tool import get_abs_path
from utils.file_handler import (
    txt_loader,
    pdf_loader,
    listdir_with_allowed_type,
    get_file_md5_hex
)
from utils.logger_handler import logger

from dotenv import load_dotenv
load_dotenv()


class VectorStoreService:
    def __init__(self,reset_db: bool=False):
        self.persist_directory = get_abs_path(chroma_conf["persist_directory"])
        self.md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

        #如果旧的Chroma持久化数据和当前版本不兼容,直接重建
        if reset_db and os.path.exists(self.persist_directory):
            logger.warning(f"[向量库]检测到reset_db=True,删除旧持久化目录:{self.persist_directory}")

        #确保持久化目录存在
        os.makedirs(self.persist_directory, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=self.persist_directory,
        )

        self.spilter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"], 
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separator"],
            length_function=len,
            )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf["k"]}
        )

    # 基础搜索    
    def similarity_search_baseline(self, query: str,k: int | None = None ) -> list[Document]:
        search_k = k or chroma_conf["k"]
        return self.vector_store.similarity_search(query, k=search_k)
    
    # 基于RAG的搜索
    def similarity_search_enhanced(self, query: str,k: int | None = None) -> list[Document]:
        search_k = k or chroma_conf["k"]
        if chroma_conf.get("enable_query_rewrite", True):
            rewrite = rewrite_query(query)
        else:
            # 没有重写就手动创建一个“假”的改写结果
            rewrite = RewriteResult(
                original_query=query.strip(),
                rewritten_query=query.strip(),
                expension_terms=[],
            )
        # 宁可多捞，不可错过
        candidate_k = chroma_conf.get("rerank_top_k", search_k * 3)
        candidate_k = max(search_k, int(candidate_k))
        docs = self.vector_store.similarity_search(rewrite.original_query, k=candidate_k)
        # 智能过滤器
        if rewrite.rewritten_query != rewrite.original_query:
            rewritten_docs = self.vector_store.similarity_search(
                rewrite.rewritten_query,
                k = candidate_k,
            )
            docs = self._merge_documents(docs, rewritten_docs)
        
        if chroma_conf.get("enable_rerank", True):
            docs = rerank_documents_with_source_diversity(docs, rewrite, search_k)
        else:
            docs = docs[:search_k]

        logger.info(
            "[rag] enhanced_search original_query=%s rewritten_query=%s docs=%s",
            rewrite.original_query,
            rewrite.rewritten_query,
            len(docs),
        )
        return docs
    

    @staticmethod
    def _merge_documents(primary_docs: list[Document],secondary_docs: list[Document]) -> list[Document]:
        merged: list[Document] = []
        seen_keys: set[tuple[str, str]] = set()

        # 解包语法 [*...] 将两个列表拼成一个大列表
        for doc in [*primary_docs, *secondary_docs]:
            key = (
                str((doc.metadata or {}).get("source", "")),
                doc.page_content
            )
            if key in seen_keys:
                continue
            merged.append(doc)
            seen_keys.add(key)
        
        return merged
    

    # 根据配置文件，决定本次搜索是走“普通路线”还是“增强路线”
    def search(self, query: str, k: int | None = None) -> list[Document]:
        if chroma_conf.get("enable_query_rewrite", True) or chroma_conf.get("enable_rerank", True):
            return self.similarity_search_enhanced(query, k)
        return self.similarity_search_baseline(query, k)


    def _check_md5_hex(self,md5_for_check:str)->bool:
        md5_dir = os.path.dirname(self.md5_store_path)
        if md5_dir:
            os.makedirs(md5_dir, exist_ok=True)

        if not os.path.exists(self.md5_store_path):
            with open(self.md5_store_path, "w", encoding="utf-8"):
                pass
            return False
        
        with open(self.md5_store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_for_check:
                    return True
        return False
        
    def _save_md5_hex(self,md5_for_check:str):
        md5_dir = os.path.dirname(self.md5_store_path)
        if md5_dir:
            os.makedirs(md5_dir, exist_ok=True)

        with open(self.md5_store_path, "a", encoding="utf-8") as f:
            f.write(md5_for_check + "\n")

    @staticmethod
    def _get_file_documents(read_path:str)->list[Document]:
        lower_path = read_path.lower()

        if lower_path.endswith(".txt"):
            return txt_loader(read_path)
        if lower_path.endswith(".pdf"):
            return pdf_loader(read_path)
        
        return []
    
    def load_document(self):
        """
        从数据文件内读取数据文件,转为向量存入向量库
        同时计算文件内的MD5做去重
        """
        allowed_files_path:list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        if not allowed_files_path: #没有文件
            logger.warning(f"[加载知识库]没有找到文件,请检查配置文件:{chroma_conf['read_path']}")
            return
        
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if self._check_md5_hex(md5_hex): #已经处理过
                logger.info(f"[加载知识库]文件{path}已经存在知识库内,跳过")
                continue

            try:
                documents:list[Document] = self._get_file_documents(path)

                if not documents: #没有内容
                    logger.warning(f"[加载知识库]文件 {path} 没有有效文本内容,跳过")
                    continue
                
                split_documents: list[Document] = prepare_documents(documents)

                if not split_documents: #没有内容
                    logger.warning(f"[加载知识库]文件 {path} 分片后没有有效文本内容,跳过")
                    continue

                self.vector_store.add_documents(split_documents)
                self._save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]文件 {path} 内容加载成功,共{len(split_documents)}个分片")

            except Exception as e:
                logger.error(f"[加载知识库]文件 {path} 加载失败,{str(e)}",exc_info=True)
                continue


if __name__ == "__main__":
    #第一次修库时,先设为True,重建向量库
    vs = VectorStoreService(reset_db=False)

    vs.load_document()
    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("="*20)

