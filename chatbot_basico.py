"""
Chatbot RAG com múltiplos PDFs.
– Responde perguntas com base nos documentos
– Cita a fonte [Fonte: nome_do_pdf.pdf p.X] apenas se usar
– Ignora citações quando não for necessário
"""

# Mais pra frente adicionar o conversation buffer memory


import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------
# 0. Configurações
# ---------------------------------------------------------------
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

PDF_DIR = Path("data/")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = (
    "Você é um assistente especializado em parto, pré-natal, enfermagem e na empresa de enfermagem. "
    "Use exclusivamente o conteúdo dos documentos fornecidos para responder perguntas sobre os temas. "
    "Caso o contexto não contenha informações relevantes à pergunta, informe isso de forma clara e respeitosa ao usuário. Nunca forneça informações que não estejam no conteúdo fornecido, e jamais invente ou suponha dados."
    "Utilize o conteúdo com precisão. Sempre que utilizar alguma informação do contexto, cite separadamente a fonte correspondente. Ao final de cada resposta, apresente uma lista individualizada com o nome dos arquivos ou referências utilizadas, sob o título: Fontes consultadas."
    "Se a informação não estiver nos documentos, diga exatamente: "
    "\"Não encontrei essa informação no material fornecido.\" "
    "Adote um tom profissional e objetivo. Explique termos técnicos de forma acessível, sempre que necessário, mantendo clareza e precisão nas respostas."
)

# ---------------------------------------------------------------
# 1. Carregar todos os PDFs da pasta / dividir em chunks
# ---------------------------------------------------------------
all_chunks = []

for pdf_file in PDF_DIR.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    for d in chunks:
        page = d.metadata.get("page", 0) + 1
        d.metadata["source"] = f"{pdf_file.name} p.{page}"

    all_chunks.extend(chunks)

# ---------------------------------------------------------------
# 2. Gerar embeddings e indexar no FAISS
# ---------------------------------------------------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectordb = FAISS.from_documents(all_chunks, embeddings)

# ---------------------------------------------------------------
# 3. Prompt + Cadeia de Pergunta-Resposta
# ---------------------------------------------------------------
document_prompt = PromptTemplate.from_template(
    "{page_content}\n\n[Fonte: {source}]"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Pergunta: {question}\n\n"
     "Contexto extraído dos documentos:\n{context}\n\n"
     "Resposta detalhada:")
])

llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.2)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 4}
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={
        "prompt": chat_prompt,
        "document_variable_name": "context",
        "document_prompt": document_prompt,
    },
    return_source_documents=True,
)

# ---------------------------------------------------------------
# 4. Loop de Conversa no Terminal
# ---------------------------------------------------------------
chat_history = []

print("✅ Chatbot RAG pronto! (Digite 'sair' para encerrar)\n")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in {"sair", "exit", "quit"}:
        print("👋 Até logo!")
        break

    chat_history.append(HumanMessage(content=pergunta))

    result = qa_chain.invoke({"question": pergunta, "chat_history": chat_history})
    resposta = result["answer"].strip()

    chat_history.append(AIMessage(content=resposta))

    print("\nAssistente:", resposta, "\n")

