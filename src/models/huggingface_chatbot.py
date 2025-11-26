"""
Módulo de implementação do chatbot utilizando modelos do HuggingFace.

Este módulo implementa um chatbot RAG usando modelos open-source do HuggingFace,
incluindo embeddings multilíngues e modelos de linguagem para geração de texto.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .base_chatbot import BaseChatbot
from ..utils.config import Config, ModelType
from ..utils.document_processor import DocumentProcessor

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


class HuggingFaceChatbot(BaseChatbot):
    """
    Implementação de chatbot RAG utilizando modelos do HuggingFace.
    
    Esta classe utiliza:
    - Embeddings: intfloat/multilingual-e5-large-instruct
    - Modelo de linguagem: configurável via parâmetro ou Config.HUGGINGFACE_CHAT_MODEL
    - Chunk size: definido em Config para ModelType.HUGGINGFACE
    - Chunk overlap: definido em Config para ModelType.HUGGINGFACE
    
    Attributes:
        embeddings (HuggingFaceEmbeddings): Modelo de embeddings do HuggingFace
        llm (HuggingFacePipeline): Pipeline do modelo de linguagem
        hf_token (str): Token de autenticação do HuggingFace
        chat_model_name (str): Nome do modelo de linguagem do HuggingFace a ser usado
    """
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        """
        Inicializa o chatbot HuggingFace.
        
        Configura automaticamente a autenticação, embeddings e modelo de linguagem.
        Requer a variável de ambiente HUGGINGFACEHUB_API_TOKEN configurada.
        
        Args:
            model_name: nome do modelo de linguagem no HuggingFace Hub
                        (ex.: "Qwen/Qwen2.5-1.5B-Instruct").
                        Se None, usa Config.HUGGINGFACE_CHAT_MODEL.
            device: dispositivo para rodar embeddings/LLM ("cpu", "cuda", etc.).
                    Atualmente usado nos embeddings; o modelo de linguagem segue
                    a configuração padrão do transformers/pipeline.
        """
        super().__init__(ModelType.HUGGINGFACE)
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[HuggingFacePipeline] = None
        self.hf_token: Optional[str] = None

        # Nome do modelo de linguagem a ser usado
        self.chat_model_name: str = model_name or Config.HUGGINGFACE_CHAT_MODEL
        self.device: str = device

        self._authenticate()
        self.initialize_embeddings()
        self.initialize_llm()
    
    def _authenticate(self):
        """
        Autentica no HuggingFace Hub usando o token da API.
        
        Raises:
            ValueError: Se o token não estiver configurado no arquivo .env
        """
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.hf_token:
            raise ValueError("Token do Hugging Face não encontrado no .env!")
        login(token=self.hf_token)
    
    def initialize_embeddings(self):
        """
        Inicializa o modelo de embeddings multilíngue do HuggingFace.
        
        Utiliza o modelo multilingual-e5-large-instruct que suporta múltiplos idiomas
        e normaliza os embeddings para melhor performance na busca vetorial.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.HUGGINGFACE_EMBED_MODEL,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    def initialize_llm(self):
        """
        Inicializa o modelo de linguagem do HuggingFace.
        
        Carrega o modelo definido em self.chat_model_name e configura o pipeline de geração
        de texto com parâmetros otimizados para respostas determinísticas.
        
        Raises:
            ValueError: Se o token do HuggingFace não estiver configurado
        """
        if self.hf_token is None:
            raise ValueError("HuggingFace token not set. Call _authenticate() first.")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.chat_model_name,
            token=self.hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.chat_model_name,
            token=self.hf_token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
            truncation=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
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
        
        chunk_size = Config.get_chunk_size(ModelType.HUGGINGFACE)
        chunk_overlap = Config.get_chunk_overlap(ModelType.HUGGINGFACE)
        
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
        Cria a cadeia de perguntas e respostas usando HuggingFace.
        
        Configura o retriever para buscar 3 documentos similares e utiliza
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
        
        # Cada chunk vem com o conteúdo e a fonte no formato desejado
        document_prompt = PromptTemplate.from_template(
            "{page_content}\n\n[Fonte: {source}]"
        )
        
        # Prompt geral, usando o SYSTEM_PROMPT com {context} e {question}
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                Config.SYSTEM_PROMPT
                + "\n\n### Pergunta do usuário:\n{question}\n\n"
                + "### Resposta detalhada e completa, seguindo todas as regras acima:\n"
            ),
        )
        
        retriever = self.get_retriever(k=3)
        
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
                "prompt": prompt_template,
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
