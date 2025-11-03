"""
Interface Streamlit para o Chatbot RAG da Nascentia
- Upload de documentos PDF
- Persistência no FAISS
- Visualização de chunks organizados
- Chat interativo
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

# Configurações
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

2. Se uma informação **não estiver claramente presente no contexto acima**, responda com: "Não tenho informações sobre isso nos documentos analisados." Jamais fale algo do seu conhecimento que não esteja nos documentos.

3. NÃO RESUMA. NÃO AGRUPE fontes. CITE após cada afirmação.

4. Mantenha um tom técnico, claro e profissional. Explique termos se necessário.

5. Desenvolva bem sua explicação sempre com base no contexto fornecido, evitando suposições ou informações externas.

Se a pergunta for apenas uma saudação ou conversa social, responda normalmente de forma educada e natural.
"""

def initialize_session_state():
    """Inicializa o estado da sessão"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

def load_or_create_vectordb():
    """Carrega ou cria o banco vetorial FAISS"""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    if os.path.exists(INDEX_PATH):
        try:
            vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            st.success("✅ Índice FAISS carregado com sucesso!")
            return vectordb, embeddings
        except Exception as e:
            st.error(f"❌ Erro ao carregar índice FAISS: {str(e)}")
            return None, embeddings
    else:
        st.info("ℹ️ Nenhum índice FAISS encontrado. Faça upload de documentos para criar um novo índice.")
        return None, embeddings

def process_uploaded_files(uploaded_files: List[Any], embeddings: OpenAIEmbeddings) -> FAISS:
    """Processa arquivos PDF enviados e cria/atualiza o índice FAISS"""
    all_chunks = []
    
    # Criar diretório data se não existir
    PDF_DIR.mkdir(exist_ok=True)
    
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            # Salvar arquivo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Carregar PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Dividir em chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, 
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(docs)
                
                # Adicionar metadados
                for d in chunks:
                    page = d.metadata.get("page", 0) + 1
                    d.metadata["source"] = f"{uploaded_file.name} p.{page}"
                
                all_chunks.extend(chunks)
                st.success(f"✅ Processado: {uploaded_file.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar {uploaded_file.name}: {str(e)}")
            finally:
                # Limpar arquivo temporário
                os.unlink(tmp_path)
        else:
            st.warning(f"⚠️ Arquivo {uploaded_file.name} não é um PDF válido")
    
    if all_chunks:
        # Criar ou atualizar índice FAISS
        if st.session_state.vectordb is not None:
            # Adicionar novos chunks ao índice existente
            new_vectordb = FAISS.from_documents(all_chunks, embeddings)
            st.session_state.vectordb.merge_from(new_vectordb)
            st.success(f"✅ Adicionados {len(all_chunks)} chunks ao índice existente")
        else:
            # Criar novo índice
            st.session_state.vectordb = FAISS.from_documents(all_chunks, embeddings)
            st.success(f"✅ Criado novo índice com {len(all_chunks)} chunks")
        
        # Salvar índice
        st.session_state.vectordb.save_local(INDEX_PATH)
        st.success("💾 Índice salvo com sucesso!")
        
        return st.session_state.vectordb
    else:
        st.error("❌ Nenhum chunk válido foi processado")
        return None

def create_qa_chain(vectordb: FAISS, embeddings: OpenAIEmbeddings):
    """Cria a cadeia de perguntas e respostas"""
    if vectordb is None:
        return None
    
    # Criar prompt
    document_prompt = PromptTemplate.from_template("{page_content}\n\n[Fonte: {source}]")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Pergunta: {question}\n\nContexto extraído dos documentos:\n{context}\n\nResposta detalhada:")
    ])
    
    # Configurar LLM e retriever
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.4)
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})
    
    # Criar cadeia RAG (sem memória integrada - será gerenciada pelo Streamlit)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": chat_prompt,
            "document_variable_name": "context",
            "document_prompt": document_prompt,
        },
        return_source_documents=True,
    )
    
    return qa_chain

def get_chunks_dataframe(vectordb: FAISS) -> pd.DataFrame:
    """Extrai informações dos chunks para visualização"""
    if vectordb is None:
        return pd.DataFrame()
    
    chunks_data = []
    
    # Obter todos os documentos do índice
    try:
        # Buscar documentos similares a uma query vazia para obter todos os chunks
        docs = vectordb.similarity_search("", k=vectordb.index.ntotal)
        
        for i, doc in enumerate(docs):
            chunks_data.append({
                "ID": i + 1,
                "Conteúdo": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "Fonte": doc.metadata.get("source", "Desconhecida"),
                "Página": doc.metadata.get("page", "N/A"),
                "Tamanho": len(doc.page_content)
            })
    except Exception as e:
        st.error(f"Erro ao extrair chunks: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(chunks_data)

def main():
    """Função principal da aplicação Streamlit"""
    st.set_page_config(
        page_title="Chatbot Nascentia",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Chatbot Inteligente - Nascentia")
    st.markdown("---")
    
    # Inicializar estado da sessão
    initialize_session_state()
    
    # Carregar embeddings
    if st.session_state.embeddings is None:
        st.session_state.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    # Carregar ou criar vectordb
    if st.session_state.vectordb is None:
        st.session_state.vectordb, _ = load_or_create_vectordb()
        
        # Se o vectordb foi carregado com sucesso, criar a cadeia QA automaticamente
        if st.session_state.vectordb is not None and st.session_state.qa_chain is None:
            st.session_state.qa_chain = create_qa_chain(st.session_state.vectordb, st.session_state.embeddings)
    
    # Sidebar para upload de documentos
    with st.sidebar:
        st.header("📁 Upload de Documentos")
        
        # Upload de arquivos
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF:",
            type=["pdf"],
            accept_multiple_files=True,
            help="Faça upload de documentos PDF que serão processados e adicionados ao índice FAISS"
        )
        
        if uploaded_files:
            if st.button("🔄 Processar Documentos", type="primary", width='stretch'):
                with st.spinner("Processando documentos..."):
                    vectordb = process_uploaded_files(uploaded_files, st.session_state.embeddings)
                    if vectordb is not None:
                        st.session_state.vectordb = vectordb
                        st.session_state.qa_chain = create_qa_chain(vectordb, st.session_state.embeddings)
                        st.rerun()
        
        st.markdown("---")
        
        # Informações do índice
        if st.session_state.vectordb is not None:
            st.success("✅ Índice FAISS ativo")
            try:
                total_docs = st.session_state.vectordb.index.ntotal
                st.info(f"📊 Total de chunks: {total_docs}")
            except:
                st.info("📊 Índice carregado")
        else:
            st.warning("⚠️ Nenhum índice disponível")
        
        st.markdown("---")
        
        # Instruções
        st.markdown("### 💡 Como usar:")
        if st.session_state.vectordb is None:
            st.markdown("1. Faça upload de PDFs")
            st.markdown("2. Clique em 'Processar'")
            st.markdown("3. Use o chat para perguntas")
            st.markdown("4. Veja os chunks na aba 'Visualização'")
        else:
            st.markdown("✅ **Chat ativo!** O índice já está carregado.")
            st.markdown("1. Use o chat para perguntas")
            st.markdown("2. Veja os chunks na aba 'Visualização'")
            st.markdown("3. Faça upload de novos PDFs se necessário")
    
    # Criar abas principais
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Visualização de Chunks"])
    
    with tab1:
        st.header("💬 Chat com o Assistente")
        
        if st.session_state.qa_chain is None:
            if st.session_state.vectordb is None:
                st.warning("⚠️ Faça upload de documentos na barra lateral para ativar o chat.")
            else:
                st.info("ℹ️ Criando cadeia de perguntas e respostas...")
                # Tentar criar a cadeia QA se o vectordb existe mas a cadeia não foi criada
                st.session_state.qa_chain = create_qa_chain(st.session_state.vectordb, st.session_state.embeddings)
                st.rerun()
        else:
            # Exibir histórico de mensagens
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Input do usuário
            if prompt := st.chat_input("Digite sua pergunta aqui..."):
                # Adicionar mensagem do usuário
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Gerar resposta
                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            # Construir contexto da conversa para o prompt
                            chat_history = ""
                            for msg in st.session_state.messages[-6:]:  # Últimas 6 mensagens
                                if msg["role"] == "user":
                                    chat_history += f"Usuário: {msg['content']}\n"
                                else:
                                    chat_history += f"Assistente: {msg['content']}\n"
                            
                            # Adicionar histórico ao prompt se existir
                            if chat_history:
                                contextual_prompt = f"Histórico da conversa:\n{chat_history}\n\nPergunta atual: {prompt}"
                            else:
                                contextual_prompt = prompt
                            
                            result = st.session_state.qa_chain.invoke({"question": contextual_prompt})
                            response = result["answer"].strip()
                            
                            st.markdown(response)
                            
                            # Adicionar resposta ao histórico
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao processar pergunta: {str(e)}")
            
            # Botão para limpar histórico
            if st.button("🗑️ Limpar Histórico"):
                st.session_state.messages = []
                st.rerun()
    
    with tab2:
        st.header("📊 Visualização de Chunks")
        
        if st.session_state.vectordb is None:
            st.warning("⚠️ Nenhum índice FAISS disponível. Faça upload de documentos na barra lateral primeiro.")
        else:
            st.subheader("📋 Chunks Organizados no Índice FAISS")
            
            # Obter dados dos chunks
            chunks_df = get_chunks_dataframe(st.session_state.vectordb)
            
            if not chunks_df.empty:
                # Filtros
                col1, col2 = st.columns(2)
                
                with col1:
                    fonte_filter = st.selectbox(
                        "Filtrar por fonte:",
                        ["Todas"] + list(chunks_df["Fonte"].unique())
                    )
                
                with col2:
                    tamanho_min = st.slider(
                        "Tamanho mínimo do chunk:",
                        min_value=0,
                        max_value=int(chunks_df["Tamanho"].max()),
                        value=0
                    )
                
                # Aplicar filtros
                filtered_df = chunks_df.copy()
                if fonte_filter != "Todas":
                    filtered_df = filtered_df[filtered_df["Fonte"] == fonte_filter]
                filtered_df = filtered_df[filtered_df["Tamanho"] >= tamanho_min]
                
                # Estatísticas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total de Chunks", len(filtered_df))
                with col2:
                    st.metric("Fontes Únicas", filtered_df["Fonte"].nunique())
                with col3:
                    st.metric("Tamanho Médio", f"{filtered_df['Tamanho'].mean():.0f} chars")
                with col4:
                    st.metric("Tamanho Total", f"{filtered_df['Tamanho'].sum():,} chars")
                
                # Tabela de chunks
                st.subheader("📄 Lista de Chunks")
                
                # Paginação
                page_size = st.selectbox("Chunks por página:", [10, 25, 50, 100], index=1)
                total_pages = (len(filtered_df) - 1) // page_size + 1
                
                if total_pages > 1:
                    page = st.selectbox("Página:", range(1, total_pages + 1))
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    display_df = filtered_df.iloc[start_idx:end_idx]
                else:
                    display_df = filtered_df
                
                # Exibir tabela
                st.dataframe(
                    display_df,
                    width='stretch',
                    height=400
                )
                
                # Download dos dados
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar dados como CSV",
                    data=csv,
                    file_name="chunks_data.csv",
                    mime="text/csv"
                )
                
                
            else:
                st.error("❌ Não foi possível extrair dados dos chunks")

if __name__ == "__main__":
    main()
