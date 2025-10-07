"""
Chatbot RAG com múltiplos PDFs + Memória nativa LangChain + FAISS persistente.
– Responde perguntas com base nos documentos
– Cita [Fonte: nome_do_pdf.pdf p.X] se usar
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ---------------------------------------------------------------
# 0. Configuração inicial e autenticação
# ---------------------------------------------------------------
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

# Lê o token do Hugging Face do .env
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Token do Hugging Face não encontrado no .env! \
Certifique-se de ter 'HUGGINGFACEHUB_API_TOKEN=hf_seu_token_aqui' no arquivo.")

# Faz login com o token
login(token=hf_token)

# ---------------------------------------------------------------
# 1. Caminhos e parâmetros principais
# ---------------------------------------------------------------
PDF_DIR = Path("data")
INDEX_PATH = "faiss_index"
EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
CHAT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SYSTEM_PROMPT = """
Você é um assistente da empresa Nascentia. Responda baseado no contexto:

{context}

Cite a fonte: [Fonte: arquivo.pdf]
"""

# ---------------------------------------------------------------
# 2. Criar ou carregar o índice FAISS
# ---------------------------------------------------------------
print("Carregando modelo de embeddings Hugging Face...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

if os.path.exists(INDEX_PATH):
    print("Carregando índice FAISS existente...")
    vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Criando novo índice FAISS...")
    all_chunks = []
    
    if not PDF_DIR.exists():
        print(f"ERRO: Diretório {PDF_DIR} não encontrado!")
        exit(1)
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERRO: Nenhum arquivo PDF encontrado em {PDF_DIR}")
        exit(1)
    
    print(f"Encontrados {len(pdf_files)} arquivos PDF para processar...")
    
    for pdf_file in pdf_files:
        print(f"Processando: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)

        for d in chunks:
            page = d.metadata.get("page", 0) + 1
            d.metadata["source"] = f"{pdf_file.name} p.{page}"

        all_chunks.extend(chunks)

    if not all_chunks:
        print("ERRO: Nenhum chunk foi criado dos documentos PDF!")
        exit(1)
    
    print(f"Criando índice FAISS com {len(all_chunks)} chunks...")
    vectordb = FAISS.from_documents(all_chunks, embeddings)
    vectordb.save_local(INDEX_PATH)
    print("Índice FAISS salvo com sucesso!")

# ---------------------------------------------------------------
# 3. Criar memória e prompt
# ---------------------------------------------------------------
memory = ConversationBufferMemory(return_messages=True)

document_prompt = PromptTemplate.from_template("{page_content}\n\n[Fonte: {source}]")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Pergunta: {question}\n\nContexto: {context}\n\nResposta:")
])

# ---------------------------------------------------------------
# 4. Carregar modelo de chat Hugging Face
# ---------------------------------------------------------------
print("Carregando modelo de chat Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, token=hf_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.4,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)
print("✅ Modelos Hugging Face carregados com sucesso!")

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="question",
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
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

# ---------------------------------------------------------------
# 5. Loop de conversa
# ---------------------------------------------------------------
print("\n🤖 Chatbot RAG com Hugging Face e memória pronta! (Digite 'sair' para encerrar)\n")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in {"sair", "exit", "quit"}:
        print("Até logo! 👋")
        break

    result = qa_chain.invoke({"question": pergunta})
    resposta = result["answer"].strip()

    print("\nAssistente:", resposta, "\n")
