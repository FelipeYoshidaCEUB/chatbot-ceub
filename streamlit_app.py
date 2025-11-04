"""
Interface Streamlit para o Chatbot RAG da Nascentia.

Este módulo fornece uma interface web moderna para interagir com os chatbots,
permitindo escolher entre diferentes modelos (OpenAI ou HuggingFace), fazer upload
de documentos PDF e visualizar os chunks indexados.

Funcionalidades:
- Seletor de modelo (OpenAI ou HuggingFace)
- Upload e processamento de documentos PDF
- Chat interativo com histórico de conversa
- Visualização de chunks indexados
- Estatísticas do índice FAISS
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from src.models.openai_chatbot import OpenAIChatbot
from src.models.huggingface_chatbot import HuggingFaceChatbot
from src.utils.config import Config, ModelType
from src.utils.document_processor import DocumentProcessor

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


def initialize_session_state():
    """
    Inicializa o estado da sessão do Streamlit.
    
    Configura as variáveis de estado necessárias para manter o contexto
    da aplicação durante a sessão do usuário.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = ModelType.OPENAI
    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False


def initialize_chatbot(model_type: ModelType) -> Optional[Any]:
    """
    Inicializa um chatbot do tipo especificado.
    
    Args:
        model_type: Tipo do modelo a ser inicializado (OPENAI ou HUGGINGFACE)
    
    Returns:
        Instância do chatbot inicializada ou None em caso de erro
    """
    try:
        if model_type == ModelType.OPENAI:
            chatbot = OpenAIChatbot()
        elif model_type == ModelType.HUGGINGFACE:
            chatbot = HuggingFaceChatbot()
        else:
            st.error(f"Model type {model_type} not supported")
            return None
        
        chatbot.load_or_create_index()
        chatbot.create_qa_chain()
        
        st.session_state.index_loaded = True
        return chatbot
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None


def process_uploaded_files(uploaded_files: List[Any], chatbot: Any) -> bool:
    """
    Processa arquivos PDF enviados via upload e os adiciona ao índice.
    
    Args:
        uploaded_files: Lista de arquivos enviados pelo usuário
        chatbot: Instância do chatbot ativo
    
    Returns:
        True se os arquivos foram processados com sucesso, False caso contrário
    """
    if chatbot is None:
        st.error("Chatbot not initialized")
        return False
    
    all_chunks = []
    
    chunk_size = Config.get_chunk_size(chatbot.model_type)
    chunk_overlap = Config.get_chunk_overlap(chatbot.model_type)
    
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            try:
                chunks = DocumentProcessor.process_uploaded_file(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    chunk_size,
                    chunk_overlap
                )
                all_chunks.extend(chunks)
                st.success(f"Processed: {uploaded_file.name} ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        else:
            st.warning(f"File {uploaded_file.name} is not a valid PDF")
    
    if all_chunks:
        try:
            chatbot.add_documents(all_chunks)
            st.success(f"Added {len(all_chunks)} chunks to index")
            return True
        except Exception as e:
            st.error(f"Error adding documents to index: {str(e)}")
            return False
    
    return False


def get_chunks_dataframe(chatbot: Any) -> pd.DataFrame:
    """
    Extrai informações dos chunks do índice FAISS para visualização.
    
    Args:
        chatbot: Instância do chatbot ativo
    
    Returns:
        DataFrame pandas com informações dos chunks (ID, Conteúdo, Fonte, Página, Tamanho)
    """
    if chatbot is None or chatbot.vectordb is None:
        return pd.DataFrame()
    
    chunks_data = []
    
    try:
        docs = chatbot.vectordb.similarity_search("", k=chatbot.vectordb.index.ntotal)
        
        for i, doc in enumerate(docs):
            chunks_data.append({
                "ID": i + 1,
                "Conteúdo": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "Fonte": doc.metadata.get("source", "Desconhecida"),
                "Página": doc.metadata.get("page", "N/A"),
                "Tamanho": len(doc.page_content)
            })
    except Exception as e:
        st.error(f"Error extracting chunks: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(chunks_data)


def main():
    st.set_page_config(
        page_title="Chatbot Nascentia",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Chatbot Inteligente - Nascentia")
    st.markdown("---")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        model_options = {
            "OpenAI (GPT-4o-mini)": ModelType.OPENAI,
            "HuggingFace (Qwen2.5-1.5B)": ModelType.HUGGINGFACE
        }
        
        selected_model_name = st.selectbox(
            "Escolha o modelo:",
            list(model_options.keys()),
            index=0 if st.session_state.model_type == ModelType.OPENAI else 1
        )
        
        selected_model_type = model_options[selected_model_name]
        
        if selected_model_type != st.session_state.model_type:
            st.session_state.model_type = selected_model_type
            st.session_state.chatbot = None
            st.session_state.index_loaded = False
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🔄 Carregar/Recarregar Modelo", type="primary"):
            with st.spinner("Inicializando chatbot..."):
                chatbot = initialize_chatbot(st.session_state.model_type)
                if chatbot:
                    st.session_state.chatbot = chatbot
                    st.success("✅ Modelo carregado com sucesso!")
                    st.rerun()
        
        st.markdown("---")
        st.header("📁 Upload de Documentos")
        
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF:",
            type=["pdf"],
            accept_multiple_files=True,
            help="Faça upload de documentos PDF que serão processados e adicionados ao índice FAISS"
        )
        
        if uploaded_files and st.session_state.chatbot:
            if st.button("🔄 Processar Documentos", type="primary"):
                with st.spinner("Processando documentos..."):
                    if process_uploaded_files(uploaded_files, st.session_state.chatbot):
                        st.rerun()
        
        st.markdown("---")
        
        if st.session_state.chatbot and st.session_state.chatbot.vectordb is not None:
            st.success("✅ Índice FAISS ativo")
            try:
                total_docs = st.session_state.chatbot.vectordb.index.ntotal
                st.info(f"📊 Total de chunks: {total_docs}")
            except:
                st.info("📊 Índice carregado")
        else:
            st.warning("⚠️ Nenhum índice disponível")
        
        st.markdown("---")
        
        st.markdown("### 💡 Como usar:")
        if st.session_state.chatbot is None:
            st.markdown("1. Escolha o modelo")
            st.markdown("2. Clique em 'Carregar/Recarregar Modelo'")
            st.markdown("3. Faça upload de PDFs (opcional)")
            st.markdown("4. Use o chat para perguntas")
        else:
            st.markdown("✅ **Chat ativo!**")
            st.markdown("1. Use o chat para perguntas")
            st.markdown("2. Veja os chunks na aba 'Visualização'")
            st.markdown("3. Faça upload de novos PDFs se necessário")
    
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Visualização de Chunks"])
    
    with tab1:
        st.header("💬 Chat com o Assistente")
        
        if st.session_state.chatbot is None:
            st.warning("⚠️ Por favor, carregue um modelo na barra lateral primeiro.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Digite sua pergunta aqui..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            chat_history = ""
                            for msg in st.session_state.messages[-6:]:
                                if msg["role"] == "user":
                                    chat_history += f"Usuário: {msg['content']}\n"
                                else:
                                    chat_history += f"Assistente: {msg['content']}\n"
                            
                            if chat_history:
                                contextual_prompt = f"Histórico da conversa:\n{chat_history}\n\nPergunta atual: {prompt}"
                            else:
                                contextual_prompt = prompt
                            
                            result = st.session_state.chatbot.query(contextual_prompt)
                            response = result["answer"].strip()
                            
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao processar pergunta: {str(e)}")
            
            if st.button("🗑️ Limpar Histórico"):
                st.session_state.messages = []
                st.rerun()
    
    with tab2:
        st.header("📊 Visualização de Chunks")
        
        if st.session_state.chatbot is None or st.session_state.chatbot.vectordb is None:
            st.warning("⚠️ Nenhum índice FAISS disponível. Carregue um modelo primeiro.")
        else:
            st.subheader("📋 Chunks Organizados no Índice FAISS")
            
            chunks_df = get_chunks_dataframe(st.session_state.chatbot)
            
            if not chunks_df.empty:
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
                        max_value=int(chunks_df["Tamanho"].max()) if len(chunks_df) > 0 else 0,
                        value=0
                    )
                
                filtered_df = chunks_df.copy()
                if fonte_filter != "Todas":
                    filtered_df = filtered_df[filtered_df["Fonte"] == fonte_filter]
                filtered_df = filtered_df[filtered_df["Tamanho"] >= tamanho_min]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total de Chunks", len(filtered_df))
                with col2:
                    st.metric("Fontes Únicas", filtered_df["Fonte"].nunique())
                with col3:
                    st.metric("Tamanho Médio", f"{filtered_df['Tamanho'].mean():.0f} chars")
                with col4:
                    st.metric("Tamanho Total", f"{filtered_df['Tamanho'].sum():,} chars")
                
                st.subheader("📄 Lista de Chunks")
                
                page_size = st.selectbox("Chunks por página:", [10, 25, 50, 100], index=1)
                total_pages = (len(filtered_df) - 1) // page_size + 1
                
                if total_pages > 1:
                    page = st.selectbox("Página:", range(1, total_pages + 1))
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    display_df = filtered_df.iloc[start_idx:end_idx]
                else:
                    display_df = filtered_df
                
                st.dataframe(
                    display_df,
                    width='stretch',
                    height=400
                )
                
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

