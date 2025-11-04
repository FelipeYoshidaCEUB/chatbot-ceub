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
    def __init__(self):
        super().__init__(ModelType.HUGGINGFACE)
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[HuggingFacePipeline] = None
        self.hf_token: Optional[str] = None
        self._authenticate()
        self.initialize_embeddings()
        self.initialize_llm()
    
    def _authenticate(self):
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.hf_token:
            raise ValueError("Token do Hugging Face não encontrado no .env!")
        login(token=self.hf_token)
    
    def initialize_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.HUGGINGFACE_EMBED_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def initialize_llm(self):
        if self.hf_token is None:
            raise ValueError("HuggingFace token not set. Call _authenticate() first.")
        
        tokenizer = AutoTokenizer.from_pretrained(
            Config.HUGGINGFACE_CHAT_MODEL,
            token=self.hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            Config.HUGGINGFACE_CHAT_MODEL,
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
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call load_or_create_index() first.")
        
        if self.llm is None:
            raise ValueError("LLM not initialized.")
        
        document_prompt = PromptTemplate.from_template("{page_content}\n\n[Fonte: {source}]")
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "### Instruções:\n"
                + Config.SYSTEM_PROMPT
                + "\n\n### Contexto relevante dos documentos:\n{context}\n"
                + "\n### Pergunta do usuário:\n{question}\n\n"
                + "### Resposta detalhada e completa:\n"
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
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call load_or_create_index() first.")
        
        new_vectordb = FAISS.from_documents(documents, self.embeddings)
        self.vectordb.merge_from(new_vectordb)
        self.vectordb.save_local(str(self.index_path))

