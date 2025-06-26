"""
Chatbot RAG com múltiplos PDFs + Memória nativa LangChain + FAISS persistente.
– Responde perguntas com base nos documentos
– Cita [Fonte: nome_do_pdf.pdf p.X] se usar
– Marca [Sem fonte] se não usar
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

PDF_DIR = Path("data")
INDEX_PATH = "faiss_index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = """
Você é um assistente da empresa Nascentia especializado em parto, pré-natal, pós-parto e seus serviços.

Baseie-se **exclusivamente** no contexto extraído dos documentos:

{context}

Regras obrigatórias:
1. Para cada informação que você extrair do contexto acima, cite logo após a frase, no formato: [Fonte: nome-do-arquivo.ext].
   Exemplo: "A gestante deve se manter hidratada. [Fonte: Cuidados na gestação.md]."

2. Se uma informação **não estiver claramente presente no contexto acima**, responda com: "Não tenho informações sobre isso nos documentos analisados."

3. NÃO RESUMA. NÃO AGRUPE fontes. CITE após cada afirmação.

4. Mantenha um tom técnico, claro e profissional. Explique termos se necessário.

5. Desenvolva bem sua explicação sempre com base no contexto fornecido, evitando suposições ou informações externas.

Se a pergunta for apenas uma saudação ou conversa social, responda normalmente de forma educada e natural.
"""


# ---------------------------------------------------------------
# 1. Carregar ou criar índice FAISS
# ---------------------------------------------------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

if os.path.exists(INDEX_PATH):
    vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
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

    vectordb = FAISS.from_documents(all_chunks, embeddings)
    vectordb.save_local(INDEX_PATH)

# ---------------------------------------------------------------
# 2. Criar memória de conversa
# ---------------------------------------------------------------
memory = ConversationBufferMemory(return_messages=True)

# ---------------------------------------------------------------
# 3. Criar Prompt + Cadeia RAG
# ---------------------------------------------------------------
document_prompt = PromptTemplate.from_template("{page_content}\n\n[Fonte: {source}]")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Pergunta: {question}\n\nContexto extraído dos documentos:\n{context}\n\nResposta detalhada:")
])

llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.4)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

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
# 4. Loop de conversa no terminal
# ---------------------------------------------------------------
print("\n✅ Chatbot RAG com memória pronta! (Digite 'sair' para encerrar)\n")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in {"sair", "exit", "quit"}:
        print("👋 Até logo!")
        break

    result = qa_chain.invoke({"question": pergunta})
    resposta = result["answer"].strip()

    print("\n")
    print("\nAssistente:", resposta, "\n")
