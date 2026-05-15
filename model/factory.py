from abc import ABC, abstractmethod
from typing import Optional,Union
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
from utils.config_handler import rag_conf

load_dotenv()


class BaseFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass

class ChatTongyiModelFactory(BaseFactory):
    def generate(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        return ChatTongyi(model = rag_conf["chat_model_name"])

class EmbeddingsModelFactory(BaseFactory):
    def generate(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        return DashScopeEmbeddings(model = rag_conf["embedding_model_name"])
    
chat_model = ChatTongyiModelFactory().generate()
embed_model = EmbeddingsModelFactory().generate()