import os
import warnings
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from .base_chatbot import BaseChatbot
from ..utils.config import Config, ModelType
from ..utils.document_processor import DocumentProcessor

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


class OpenAIChatbot(BaseChatbot):
    def __init__(self):
        super().__init__(ModelType.OPENAI)
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.llm: Optional[ChatOpenAI] = None
        self.initialize_embeddings()
        self.initialize_llm()
    
    def initialize_embeddings(self):
        self.embeddings = OpenAIEmbeddings(model=Config.OPENAI_EMBED_MODEL)
    
    def initialize_llm(self):
        self.llm = ChatOpenAI(model_name=Config.OPENAI_CHAT_MODEL, temperature=0.4)
    
    def load_or_create_index(self, force_recreate: bool = False):
        if os.path.exists(self.index_path) and not force_recreate:
            try:
                self.vectordb = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return
            except Exception as e:
                warnings.warn(f"Error loading index: {e}. Creating new index.")
        
        chunk_size = Config.get_chunk_size(ModelType.OPENAI)
        chunk_overlap = Config.get_chunk_overlap(ModelType.OPENAI)
        
        all_chunks = DocumentProcessor.load_and_split_documents(
            Config.PDF_DIR,
            chunk_size,
            chunk_overlap
        )
        
        if not all_chunks:
            raise ValueError("No documents were processed. Check PDF_DIR path and files.")
        
        self.vectordb = FAISS.from_documents(all_chunks, self.embeddings)
        self.vectordb.save_local(str(self.index_path))
    
    def create_qa_chain(self):
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call load_or_create_index() first.")
        
        if self.llm is None:
            raise ValueError("LLM not initialized.")
        
        document_prompt = PromptTemplate.from_template("{page_content}\n\n[Fonte: {source}]")
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", Config.SYSTEM_PROMPT),
            ("human", "Pergunta: {question}\n\nContexto extraído dos documentos:\n{context}\n\nResposta detalhada:")
        ])
        
        retriever = self.get_retriever(k=6)
        
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="answer"
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            output_key="answer",
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": chat_prompt,
                "document_variable_name": "context",
                "document_prompt": document_prompt,
            },
            return_source_documents=True,
        )
        
        return self.qa_chain
    
    def add_documents(self, documents: List):
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call load_or_create_index() first.")
        
        new_vectordb = FAISS.from_documents(documents, self.embeddings)
        self.vectordb.merge_from(new_vectordb)
        self.vectordb.save_local(str(self.index_path))

