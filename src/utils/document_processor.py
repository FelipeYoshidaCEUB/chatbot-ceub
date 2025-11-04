from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import Config, ModelType


class DocumentProcessor:
    @staticmethod
    def load_and_split_documents(
        pdf_dir: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
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

