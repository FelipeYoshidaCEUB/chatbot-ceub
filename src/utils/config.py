import os
from pathlib import Path
from enum import Enum
from typing import Optional


class ModelType(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class Config:
    PDF_DIR = Path("data")
    
    OPENAI_INDEX_PATH = Path("faiss_index/openai")
    HUGGINGFACE_INDEX_PATH = Path("faiss_index/huggingface")
    
    OPENAI_EMBED_MODEL = "text-embedding-3-small"
    OPENAI_CHAT_MODEL = "gpt-4o-mini"
    OPENAI_CHUNK_SIZE = 1500
    OPENAI_CHUNK_OVERLAP = 200
    
    HUGGINGFACE_EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
    HUGGINGFACE_CHAT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    HUGGINGFACE_CHUNK_SIZE = 400
    HUGGINGFACE_CHUNK_OVERLAP = 50
    
    SYSTEM_PROMPT = """Você é um assistente da empresa Nascentia especializado em parto, pré-natal, pós-parto e seus serviços.

Baseie-se **exclusivamente** no contexto extraído dos documentos:

{context}

Regras obrigatórias:
1. Para cada informação que você extrair do contexto acima, cite logo após a frase, no formato: [Fonte: nome-do-arquivo.ext].
   Exemplo: "A gestante deve se manter hidratada. [Fonte: Cuidados na gestação.md]."

2. Se uma informação **não estiver claramente presente no contexto acima**, responda com: "Não tenho informações sobre isso nos documentos analisados." Jamais fale algo do seu conhecimento que não esteja nos documentos.

3. NÃO RESUMA. NÃO AGRUPE fontes. CITE após cada afirmação.

4. Mantenha um tom técnico, claro e profissional. Explique termos se necessário.

5. Desenvolva bem sua explicação sempre com base no contexto fornecido, evitando suposições ou informações externas.

Se a pergunta for apenas uma saudação ou conversa social, responda normalmente de forma educada e natural.
"""
    
    @staticmethod
    def get_index_path(model_type: ModelType) -> Path:
        if model_type == ModelType.OPENAI:
            return Config.OPENAI_INDEX_PATH
        elif model_type == ModelType.HUGGINGFACE:
            return Config.HUGGINGFACE_INDEX_PATH
        else:
            raise ValueError(f"Model type {model_type} not supported")
    
    @staticmethod
    def get_chunk_size(model_type: ModelType) -> int:
        if model_type == ModelType.OPENAI:
            return Config.OPENAI_CHUNK_SIZE
        elif model_type == ModelType.HUGGINGFACE:
            return Config.HUGGINGFACE_CHUNK_SIZE
        else:
            raise ValueError(f"Model type {model_type} not supported")
    
    @staticmethod
    def get_chunk_overlap(model_type: ModelType) -> int:
        if model_type == ModelType.OPENAI:
            return Config.OPENAI_CHUNK_OVERLAP
        elif model_type == ModelType.HUGGINGFACE:
            return Config.HUGGINGFACE_CHUNK_OVERLAP
        else:
            raise ValueError(f"Model type {model_type} not supported")

