"""
Módulo base para implementação de chatbots RAG.

Este módulo define a classe abstrata BaseChatbot que serve como base
para todas as implementações de chatbots com recuperação aumentada por geração (RAG).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from ..utils.config import Config, ModelType


class BaseChatbot(ABC):
    """
    Classe base abstrata para implementação de chatbots RAG.
    
    Esta classe define a interface comum para todos os chatbots que utilizam
    Retrieval-Augmented Generation (RAG) para responder perguntas baseadas em documentos.
    
    Attributes:
        model_type (ModelType): Tipo do modelo utilizado (OpenAI ou HuggingFace)
        index_path (Path): Caminho para o diretório do índice FAISS
        vectordb (Optional[FAISS]): Banco de dados vetorial FAISS
        qa_chain (Optional[ConversationalRetrievalChain]): Cadeia de perguntas e respostas
    
    Methods:
        initialize_embeddings(): Inicializa o modelo de embeddings
        initialize_llm(): Inicializa o modelo de linguagem
        load_or_create_index(): Carrega ou cria o índice FAISS
        create_qa_chain(): Cria a cadeia de perguntas e respostas
        query(): Processa uma pergunta e retorna a resposta
        get_retriever(): Obtém o retriever para busca de documentos similares
    """
    def __init__(self, model_type: ModelType):
        """
        Inicializa a classe base do chatbot.
        
        Args:
            model_type: Tipo do modelo a ser utilizado (OPENAI ou HUGGINGFACE)
        """
        self.model_type = model_type
        self.index_path = Config.get_index_path(model_type)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.vectordb: Optional[FAISS] = None
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
    
    @abstractmethod
    def initialize_embeddings(self):
        """
        Inicializa o modelo de embeddings para conversão de texto em vetores.
        
        Este método deve ser implementado pelas classes filhas para configurar
        o modelo de embeddings específico (OpenAI ou HuggingFace).
        """
        pass
    
    @abstractmethod
    def initialize_llm(self):
        """
        Inicializa o modelo de linguagem (LLM) para geração de respostas.
        
        Este método deve ser implementado pelas classes filhas para configurar
        o modelo de linguagem específico (OpenAI ou HuggingFace).
        """
        pass
    
    @abstractmethod
    def load_or_create_index(self, force_recreate: bool = False):
        """
        Carrega um índice FAISS existente ou cria um novo a partir dos documentos PDF.
        
        Args:
            force_recreate: Se True, força a recriação do índice mesmo se já existir
        
        Raises:
            ValueError: Se não houver documentos para processar
        """
        pass
    
    @abstractmethod
    def create_qa_chain(self):
        """
        Cria a cadeia de perguntas e respostas (QA chain).
        
        Esta cadeia combina o retriever com o LLM para gerar respostas
        baseadas nos documentos recuperados.
        
        Returns:
            ConversationalRetrievalChain: Cadeia configurada para perguntas e respostas
        
        Raises:
            ValueError: Se o banco vetorial ou LLM não estiverem inicializados
        """
        pass
    
    def query(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Processa uma pergunta e retorna a resposta com informações adicionais.
        
        Args:
            question: Pergunta do usuário
            chat_history: Histórico opcional da conversa (não utilizado atualmente)
        
        Returns:
            Dict contendo a resposta e documentos fonte
        
        Raises:
            ValueError: Se a cadeia QA não estiver inicializada
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")
        
        result = self.qa_chain.invoke({"question": question})
        return result
    
    def get_retriever(self, k: int = 6):
        """
        Obtém o retriever para busca de documentos similares.
        
        Args:
            k: Número de documentos similares a retornar (padrão: 6)
        
        Returns:
            Retriever configurado para buscar k documentos mais similares
        
        Raises:
            ValueError: Se o banco vetorial não estiver inicializado
        """
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call load_or_create_index() first.")
        return self.vectordb.as_retriever(search_kwargs={"k": k})

