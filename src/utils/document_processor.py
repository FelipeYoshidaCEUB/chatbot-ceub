"""
Módulo de processamento de documentos PDF.

Este módulo fornece utilitários para carregar e processar documentos PDF,
dividindo-os em chunks apropriados para indexação vetorial.
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import Config, ModelType


class DocumentProcessor:
    """
    Classe para processamento de documentos PDF.
    
    Fornece métodos estáticos para carregar PDFs de diretórios ou arquivos
    individuais e dividi-los em chunks com metadados apropriados.
    """
    @staticmethod
    def load_and_split_documents(
        pdf_dir: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        """
        Carrega todos os PDFs de um diretório e os divide em chunks.
        
        Args:
            pdf_dir: Caminho para o diretório contendo os PDFs
            chunk_size: Tamanho máximo de cada chunk em caracteres
            chunk_overlap: Sobreposição entre chunks em caracteres
        
        Returns:
            Lista de documentos LangChain com metadados de fonte e página
        
        Raises:
            FileNotFoundError: Se o diretório não existir
            ValueError: Se não houver arquivos PDF no diretório
        """
        all_chunks = []
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory {pdf_dir} not found")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            
            for doc in chunks:
                page = doc.metadata.get("page", 0) + 1
                doc.metadata["source"] = f"{pdf_file.name} p.{page}"
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    @staticmethod
    def process_uploaded_file(file_content: bytes, filename: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """
        Processa um arquivo PDF enviado via upload e o divide em chunks.
        
        O arquivo é salvo temporariamente, processado e depois removido.
        
        Args:
            file_content: Conteúdo binário do arquivo PDF
            filename: Nome do arquivo original
            chunk_size: Tamanho máximo de cada chunk em caracteres
            chunk_overlap: Sobreposição entre chunks em caracteres
        
        Returns:
            Lista de documentos LangChain com metadados de fonte e página
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(docs)
            
            for doc in chunks:
                page = doc.metadata.get("page", 0) + 1
                doc.metadata["source"] = f"{filename} p.{page}"
            
            return chunks
        finally:
            os.unlink(tmp_path)

