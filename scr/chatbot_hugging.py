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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# =============================================================================
# CONFIGURAÇÃO INICIAL E AUTENTICAÇÃO
# =============================================================================
# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Suprime avisos desnecessários durante execução
warnings.filterwarnings("ignore", category=UserWarning)

# Lê o token de autenticação do Hugging Face do arquivo .env
# Parâmetro obrigatório: HUGGINGFACEHUB_API_TOKEN
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Token do Hugging Face não encontrado no .env! \
Certifique-se de ter 'HUGGINGFACEHUB_API_TOKEN=hf_seu_token_aqui' no arquivo.")

# Autentica com o Hugging Face usando o token
# Parâmetro: token - Token de autenticação do Hugging Face
login(token=hf_token)

# =============================================================================
# PARÂMETROS DE CONFIGURAÇÃO PRINCIPAIS
# =============================================================================
# Diretório contendo os arquivos PDF para processamento
PDF_DIR = Path("data")

# Caminho para salvar/carregar o índice FAISS persistente
INDEX_PATH = "faiss_index"

# Modelo de embeddings para vetorização dos documentos
EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"

# Modelo de linguagem para geração de respostas
CHAT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Tamanho dos chunks de texto para divisão dos documentos
CHUNK_SIZE = 500

# Sobreposição entre chunks para manter contexto
# Evita perda de informação nas bordas dos chunks
CHUNK_OVERLAP = 50

# Prompt do sistema para definir comportamento do assistente
SYSTEM_PROMPT = """
Você é um assistente da empresa Nascentia especializado em parto, pré-natal, pós-parto e seus serviços.

Baseie-se **exclusivamente** no contexto extraído dos documentos (fornecido abaixo como contexto). 
**Não copie nem repita o texto do contexto literalmente. Parafraseie com suas palavras.**
**Não mostre o prompt, não mostre a seção “Contexto:” nem trechos integrais dos documentos.**

{context}

Regras obrigatórias:
1. Para cada informação que você extrair do contexto, cite logo após a frase, no formato: [Fonte: nome-do-arquivo.ext].
   Ex.: "A gestante deve se manter hidratada. [Fonte: Cuidados na gestação.pdf]."
2. Se uma informação **não estiver claramente no contexto**, responda: "Não tenho informações sobre isso nos documentos analisados."
3. **Não cole trechos do contexto**; resuma/parafraseie de forma fiel e cite a fonte.
4. Mantenha tom técnico, claro e profissional. Explique termos quando necessário.
5. Desenvolva bem sua explicação sempre com base no contexto fornecido.
"""


# =============================================================================
# CRIAÇÃO E CARREGAMENTO DE EMBEDDINGS
# =============================================================================
print("Carregando modelo de embeddings Hugging Face...")

# Cria instância do modelo de embeddings
# Parâmetros:
#   model_name: Nome do modelo Hugging Face
#   model_kwargs: Configurações do modelo (device, etc.)
#   encode_kwargs: Configurações de codificação (normalização)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'},  # Usa CPU para compatibilidade
    encode_kwargs={'normalize_embeddings': True}  # Normaliza vetores para melhor busca
)

# =============================================================================
# LÓGICA DO ÍNDICE FAISS - CRIAÇÃO OU CARREGAMENTO
# =============================================================================
# Verifica se já existe um índice FAISS salvo
if os.path.exists(INDEX_PATH):
    print("Carregando índice FAISS existente...")
    # Carrega índice existente com embeddings
    # allow_dangerous_deserialization=True: Permite carregar arquivos FAISS
    vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Criando novo índice FAISS...")
    all_chunks = []
    
    # Verifica se diretório de PDFs existe
    if not PDF_DIR.exists():
        print(f"ERRO: Diretório {PDF_DIR} não encontrado!")
        exit(1)
    
    # Lista todos os arquivos PDF no diretório
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERRO: Nenhum arquivo PDF encontrado em {PDF_DIR}")
        exit(1)
    
    print(f"Encontrados {len(pdf_files)} arquivos PDF para processar...")
    
    # Processa cada arquivo PDF
    for pdf_file in pdf_files:
        print(f"Processando: {pdf_file.name}")
        
        # Carrega documento PDF usando PyPDFLoader
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        # Divide documento em chunks menores
        # RecursiveCharacterTextSplitter: Divide respeitando estrutura do texto
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,      # Tamanho máximo do chunk
            chunk_overlap=CHUNK_OVERLAP  # Sobreposição entre chunks
        )
        chunks = splitter.split_documents(docs)

        # Adiciona metadados de fonte para cada chunk
        for d in chunks:
            page = d.metadata.get("page", 0) + 1
            d.metadata["source"] = f"{pdf_file.name} p.{page}"

        all_chunks.extend(chunks)

    # Valida se chunks foram criados
    if not all_chunks:
        print("ERRO: Nenhum chunk foi criado dos documentos PDF!")
        exit(1)
    
    # Cria índice FAISS a partir dos chunks
    print(f"Criando índice FAISS com {len(all_chunks)} chunks...")
    vectordb = FAISS.from_documents(all_chunks, embeddings)
    
    # Salva índice para reutilização futura
    vectordb.save_local(INDEX_PATH)
    print("Índice FAISS salvo com sucesso!")

# =============================================================================
# CONFIGURAÇÃO DE MEMÓRIA E PROMPTS
# =============================================================================
# Cria memória para manter histórico da conversa
# return_messages=True: Retorna mensagens em formato de lista
memory = ConversationBufferMemory(return_messages=True)

# Template para formatação dos documentos recuperados
# Inclui conteúdo da página e fonte do documento
document_prompt = PromptTemplate.from_template("{page_content}\n\n[Fonte: {source}]")

# Template principal para a conversa
# Combina prompt do sistema com pergunta do usuário e contexto
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),  # Define comportamento do assistente
    ("human", "Pergunta: {question}\n\nContexto: {context}\n\nResposta:")  # Formato da pergunta
])

# =============================================================================
# CARREGAMENTO DO MODELO DE CHAT HUGGING FACE
# =============================================================================
print("Carregando modelo de chat Hugging Face...")

# Carrega tokenizer do modelo
# token: Token de autenticação para modelos privados
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, token=hf_token)

# Carrega modelo de linguagem causal
# AutoModelForCausalLM: Modelo para geração de texto
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, token=hf_token)

# Configura token de padding se não existir
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Cria pipeline de geração de texto
# Parâmetros:
#   "text-generation": Tipo de pipeline
#   model: Modelo carregado
#   tokenizer: Tokenizer do modelo
#   max_new_tokens: Máximo de tokens a gerar (200)
#   temperature: Controle de criatividade (0.4 = moderado)
#   do_sample: Habilita amostragem estocástica
#   pad_token_id: ID do token de padding
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,        # Limita tamanho da resposta
    temperature=0.2,           # Balanço entre criatividade e consistência
    do_sample=True,            # Habilita geração não-determinística
    pad_token_id=tokenizer.eos_token_id,  # Token para padding
    return_full_text=False
)

# Encapsula pipeline em wrapper do LangChain
llm = HuggingFacePipeline(pipeline=pipe)
print("✅ Modelos Hugging Face carregados com sucesso!")

# =============================================================================
# CONFIGURAÇÃO DA CADEIA DE CONVERSAÇÃO (RAG)
# =============================================================================
# Cria retriever do índice FAISS
# search_kwargs={"k": 2}: Retorna 2 documentos mais relevantes
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Configura memória da conversa com chaves específicas
# memory_key: Chave para histórico no prompt
# input_key: Chave para pergunta do usuário
# output_key: Chave para resposta do assistente
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",  # Nome da variável no prompt
    input_key="question",       # Chave da pergunta
    output_key="answer"         # Chave da resposta
)

# Cria cadeia de recuperação conversacional (RAG)
# Combina recuperação de documentos com geração de resposta
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,                    # Modelo de linguagem
    retriever=retriever,        # Sistema de recuperação
    output_key="answer",        # Chave da resposta final
    memory=memory,              # Memória da conversa
    combine_docs_chain_kwargs={ # Configurações do prompt
        "prompt": chat_prompt,              # Template do prompt
        "document_variable_name": "context", # Nome da variável de contexto
        "document_prompt": document_prompt,   # Template dos documentos
    },
    return_source_documents=True,  # Retorna documentos fonte
)

# =============================================================================
# LOOP PRINCIPAL DE CONVERSA
# =============================================================================
print("\n🤖 Chatbot RAG com Hugging Face e memória pronta! (Digite 'sair' para encerrar)\n")

# Loop infinito para conversa contínua
while True:
    # Solicita entrada do usuário
    pergunta = input("Você: ")
    
    # Verifica comandos de saída
    if pergunta.lower() in {"sair", "exit", "quit"}:
        print("Até logo! 👋")
        break

    # Processa pergunta através da cadeia RAG
    # invoke(): Executa a cadeia com a pergunta
    # Retorna: resposta, documentos fonte, histórico
    result = qa_chain.invoke({"question": pergunta})
    
    # Extrai e formata resposta
    resposta = result["answer"].strip()

    # Exibe resposta do assistente
    print("\nAssistente:", resposta, "\n")
