"""
Módulo de implementação do chatbot utilizando modelos da OpenAI.

Este módulo implementa um chatbot RAG usando embeddings e modelos de linguagem
da OpenAI (GPT-4o-mini e text-embedding-3-small).
"""

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
    """
    Implementação de chatbot RAG utilizando modelos da OpenAI.
    
    Esta classe utiliza:
    - Embeddings: text-embedding-3-small
    - Modelo de linguagem: gpt-4o-mini
    - Chunk size: 1500 caracteres
    - Chunk overlap: 200 caracteres
    
    Attributes:
        embeddings (OpenAIEmbeddings): Modelo de embeddings da OpenAI
        llm (ChatOpenAI): Modelo de linguagem da OpenAI
    """
    def __init__(self):
        """
        Inicializa o chatbot OpenAI.
        
        Configura automaticamente os embeddings e o modelo de linguagem.
        Requer a variável de ambiente OPENAI_API_KEY configurada.
        """
        super().__init__(ModelType.OPENAI)
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.llm: Optional[ChatOpenAI] = None
        self.initialize_embeddings()
        self.initialize_llm()
    
    def initialize_embeddings(self):
        """Inicializa o modelo de embeddings da OpenAI."""
        self.embeddings = OpenAIEmbeddings(model=Config.OPENAI_EMBED_MODEL)
    
    def initialize_llm(self):
        """
        Inicializa o modelo de linguagem da OpenAI.
        
        Utiliza o modelo gpt-4o-mini com temperatura de 0.4 para respostas
        balanceadas entre criatividade e consistência.
        """
        self.llm = ChatOpenAI(model_name=Config.OPENAI_CHAT_MODEL, temperature=0.4)
    
    def load_or_create_index(self, force_recreate: bool = False):
        """
        Carrega um índice FAISS existente ou cria um novo.
        
        Se o índice já existir e force_recreate for False, carrega o índice existente.
        Caso contrário, processa todos os PDFs na pasta data/ e cria um novo índice.
        
        Args:
            force_recreate: Se True, força a recriação do índice mesmo se já existir
        
        Raises:
            ValueError: Se não houver documentos PDF na pasta data/
        """
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
        """
        Cria a cadeia de perguntas e respostas usando OpenAI.
        
        Configura o retriever para buscar 6 documentos similares e utiliza
        memória de conversa para manter o contexto da conversa.
        
        Returns:
            ConversationalRetrievalChain: Cadeia configurada para perguntas e respostas
        
        Raises:
            ValueError: Se o banco vetorial ou LLM não estiverem inicializados
        """
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
        """
        Adiciona novos documentos ao índice FAISS existente.
        
        Processa os documentos fornecidos e os adiciona ao índice atual,
        mantendo os documentos anteriores.
        
        Args:
            documents: Lista de documentos LangChain a serem adicionados
        
        Raises:
            ValueError: Se o banco vetorial não estiver inicializado
        """
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call load_or_create_index() first.")
        
        new_vectordb = FAISS.from_documents(documents, self.embeddings)
        self.vectordb.merge_from(new_vectordb)
        self.vectordb.save_local(str(self.index_path))

