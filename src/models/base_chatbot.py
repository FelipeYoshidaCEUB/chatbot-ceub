from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from ..utils.config import Config, ModelType


class BaseChatbot(ABC):
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.index_path = Config.get_index_path(model_type)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.vectordb: Optional[FAISS] = None
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
    
    @abstractmethod
    def initialize_embeddings(self):
        pass
    
    @abstractmethod
    def initialize_llm(self):
        pass
    
    @abstractmethod
    def load_or_create_index(self, force_recreate: bool = False):
        pass
    
    @abstractmethod
    def create_qa_chain(self):
        pass
    
    def query(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")
        
        result = self.qa_chain.invoke({"question": question})
        return result
    
    def get_retriever(self, k: int = 6):
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call load_or_create_index() first.")
        return self.vectordb.as_retriever(search_kwargs={"k": k})

