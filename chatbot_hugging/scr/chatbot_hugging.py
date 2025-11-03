"""
Chatbot RAG com múltiplos PDFs + Memória nativa LangChain + FAISS persistente.
– Responde perguntas com base nos documentos
– Cita [Fonte: nome_do_pdf.pdf p.X] se usar
"""

# =============================================================================
# IMPORTS E DEPENDÊNCIAS
# =============================================================================
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =============================================================================
# CONFIGURAÇÃO INICIAL E AUTENTICAÇÃO
# =============================================================================
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Token do Hugging Face não encontrado no .env!")

login(token=hf_token)

# =============================================================================
# PARÂMETROS PRINCIPAIS
# =============================================================================
PDF_DIR = Path("data")
INDEX_PATH = "faiss_index"
EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
CHAT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

SYSTEM_PROMPT = """
Você é um assistente técnico da empresa Nascentia, especializado em parto, pré-natal, pós-parto e seus serviços.

Baseie-se **exclusivamente** no contexto extraído dos documentos abaixo.

Regras obrigatórias:
1. Cite a fonte após cada informação no formato: [Fonte: nome-do-arquivo.ext p.X].
2. Se a informação **não estiver presente nos documentos**, diga:
   "Não tenho informações sobre isso nos documentos analisados."
3. Evite repetições ou explicações muito extensas.
4. Mantenha tom técnico, claro e **objetivo**.
5. Organize as respostas em **parágrafos curtos e diretos**.
"""


# =============================================================================
# EMBEDDINGS E ÍNDICE FAISS
# =============================================================================
print("Carregando modelo de embeddings...")

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
    if not PDF_DIR.exists():
        print(f"ERRO: Diretório {PDF_DIR} não encontrado!")
        exit(1)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERRO: Nenhum arquivo PDF encontrado em {PDF_DIR}")
        exit(1)

    all_chunks = []
    for pdf_file in pdf_files:
        print(f"Processando: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        for d in chunks:
            page = d.metadata.get("page", 0) + 1
            d.metadata["source"] = f"{pdf_file.name} p.{page}"
        all_chunks.extend(chunks)

    if not all_chunks:
        print("ERRO: Nenhum chunk criado!")
        exit(1)

    vectordb = FAISS.from_documents(all_chunks, embeddings)
    vectordb.save_local(INDEX_PATH)
    print("✅ Índice FAISS salvo com sucesso!")

# =============================================================================
# MEMÓRIA E PROMPTS
# =============================================================================
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="question",
    output_key="answer"
)

document_prompt = PromptTemplate.from_template("{page_content}\n\n[Fonte: {source}]")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "### Instruções:\n"
        + SYSTEM_PROMPT
        + "\n\n### Contexto relevante dos documentos:\n{context}\n"
        + "\n### Pergunta do usuário:\n{question}\n\n"
        + "### Resposta detalhada e completa:\n"
    ),
)

# =============================================================================
# MODELO DE CHAT
# =============================================================================
print("Carregando modelo de linguagem...")

tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, token=hf_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.4,
    do_sample=False,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,  # <-- impede que retorne o input junto
)

llm = HuggingFacePipeline(pipeline=pipe)
print("✅ Modelo carregado com sucesso!")

# =============================================================================
# CADEIA RAG
# =============================================================================
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
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

# =============================================================================
# LOOP DE CONVERSA
# =============================================================================
print("\n🤖 Chatbot RAG pronto! (Digite 'sair' para encerrar)\n")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in {"sair", "exit", "quit"}:
        print("Até logo! 👋")
        break

    result = qa_chain.invoke({"question": pergunta})
    resposta = result["answer"].strip()
    print("\nAssistente:", resposta, "\n")
