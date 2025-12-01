# Chatbot final 

import base64
import warnings
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from markdown2 import markdown

# Modelo fixo OpenAI
from src.models.openai_chatbot import OpenAIChatbot

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

# Caminhos das logos
COMPANY_LOGO_PATH = "Docs\images\Logo Nascentia 2.png"
CHATBOT_AVATAR_PATH = "Docs\images\Logo chatbot.png"


def _image_to_base64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


CHATBOT_AVATAR_B64 = _image_to_base64(CHATBOT_AVATAR_PATH)


# =========================================================
#                       CSS
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

/* ------------------ BASE GERAL ------------------ */

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* fundo geral */
.stApp {
    background-color: #ECE2F4 !important;  /* fundo lilás suave */
}

/* força camadas internas para lilás claro ( evita fundo preto no dark mode ) */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stBottom"],
[data-testid="stBottom"] div,
div[data-testid="stToolbar"] + div,
section[data-testid="stSidebar"] + div {
    background-color: #ECE2F4 !important;
}

/* barra superior */
header[data-testid="stHeader"] {
    background-color: #2E1A38 !important;
    color: #E7D9F4 !important;
    height: 3rem;
    border-bottom: 1px solid #3B2346;
}

/* espaçamento do conteúdo */
div.block-container {
    padding-top: 3.4rem !important;
}

/* ------------------ TÍTULOS / TOPO ------------------ */

/* Títulos gerais */
h1, h2, h3 {
    color: #5A357A !important;
    font-weight: 600 !important;
}

/* Topo com logo + nome NascentIA */
.top-bar {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 1.8rem;
}

.app-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #5A357A;
    margin: 0;
}

.app-subtitle {
    font-size: 0.95rem;
    color: #7A5C9B;
    margin-top: 0.25rem;
}

.title-ia {
    color: #B764D6;
    font-weight: 700;
}

/* ------------------ INPUT (ST.CHAT_INPUT) ------------------ */

/* faixa inferior */
[data-testid="stChatInput"] {
    background-color: #ECE2F4 !important;
    border-top: none !important;
    padding: 0.6rem 0.5rem !important;
}

/* caixa branca que envolve campo + botão */
[data-testid="stChatInput"] > div:first-child {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    border: 2px solid #D7C6EE !important;
    display: flex;
    align-items: center;
}

/* campo de texto */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] div[role="textbox"],
div[data-baseweb="textarea"] textarea,
div[data-baseweb="textarea"] {
    background-color: #FFFFFF !important;
    color: #3A2B4D !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    caret-color: #7B4BA8 !important;
}

/* Placeholder do campo de input do chat */
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
    color: #7A649A !important;   /* lilás médio */
    opacity: 1 !important;       /* garante que apareça */
    font-weight: 400;
}

/* botão enviar */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #7B4BA8, #5C3284) !important;
    border-radius: 12px !important;
    color: white !important;
    width: 56px !important;
    min-width: 56px !important;
    border: none !important;
}

[data-testid="stChatInput"] button:hover {
    background: #552D7A !important;
}

/* ------------------ CHAT WRAPPER ------------------ */

.chat-wrapper {
    background: #FFFFFF;
    border-radius: 22px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
    border: 1px solid #E6D8F3;
    margin-bottom: 2rem;
}

/* NÃO é flex aqui, para não brigar com wrappers do Streamlit */
.chat-scroll {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 8px;
}

/* ------------------ USER (DIREITA) ------------------ */

/* bolha do usuário alinhada à direita, com tamanho da mensagem */
.user-msg {
    background: linear-gradient(135deg, #6C409A, #4E2A70);
    color: white;
    padding: 10px 14px;
    border-radius: 14px 14px 4px 14px;

    /* truque para empurrar à direita */
    display: block;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
    margin-right: 6px;

    font-size: 0.95rem;
}

/* ------------------ BOT (ESQUERDA) ------------------ */

.bot-msg-wrapper {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    max-width: 80%;
    margin-top: 6px;
}

.bot-avatar-img {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 2px 4px rgba(0,0,0,0.18);
    flex-shrink: 0;
}

.bot-msg {
    background: #DCC8F0;
    color: #4A2C62;
    padding: 10px 14px;
    border-radius: 14px 14px 14px 4px;
    width: auto;
    max-width: 75%;
    font-size: 0.95rem;
    display: inline-block;
}

/* ------------------ DIGITANDO ------------------ */

.typing-bubble {
    background: #DCC8F0;
    color: #4A2C62;
    padding: 8px 12px;
    border-radius: 14px;
    display: flex;
    gap: 6px;
    max-width: 120px;
}

.typing-dot {
    width: 6px;
    height: 6px;
    background: #7B4BA8;
    border-radius: 50%;
    animation: typingBlink 1.2s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingBlink {
    0%   { opacity: 0.2; transform: translateY(0px); }
    50%  { opacity: 1;   transform: translateY(-2px); }
    100% { opacity: 0.2; transform: translateY(0px); }
}

</style>
""",
    unsafe_allow_html=True,
)



# =========================================================
#                ESTADO DA SESSÃO
# =========================================================

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Olá! Eu sou a **NascentIA**. Como posso ajudar você hoje?"
            }
        ]

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = OpenAIChatbot()
        st.session_state.chatbot.load_or_create_index()
        st.session_state.chatbot.create_qa_chain()



# =========================================================
#                        APP
# =========================================================

def main():
    st.set_page_config(page_title="NascentIA", page_icon="🤰", layout="wide")
    initialize_session_state()

    # TOPO ------------------------------------------
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(COMPANY_LOGO_PATH, width=140)

    with col_title:
        st.markdown(
            """
            <h1 class="app-title">Nascent<span class="title-ia">IA</span></h1>
            <div class="app-subtitle">Assistente virtual especializada em parto, pré-natal e pós-parto.</div>
            """,
            unsafe_allow_html=True,
        )

    # CHAT CONTAINER -------------------------------
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    st.subheader("💬 Chat com a NascentIA")

    # LISTA DE MENSAGENS (SEMPRE MOSTRA ANTES DO INPUT)
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'''
                <div class="message-row user-row">
                    <div class="user-msg">{msg["content"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:
            avatar = f'<img src="data:image/png;base64,{CHATBOT_AVATAR_B64}" class="bot-avatar-img" />'
            html_msg = markdown(msg["content"])

            st.markdown(
                f'''
                <div class="message-row bot-row">
                    <div class="bot-msg-wrapper">
                        {avatar}
                        <div class="bot-msg">{html_msg}</div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )


    st.markdown("</div>", unsafe_allow_html=True)

    # INDICADOR "DIGITANDO..."
    typing_placeholder = st.empty()

    # INPUT DO USUÁRIO
    prompt = st.chat_input("Digite sua pergunta aqui...", key="unique_chat_input")

    if prompt:
        # Mostra IMEDIATAMENTE a mensagem na tela
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # SE A ÚLTIMA MENSAGEM FOR DO USUÁRIO → IA DEVE RESPONDER
    if st.session_state.messages[-1]["role"] == "user":
        user_msg = st.session_state.messages[-1]["content"]

        # Mostra bolha "digitando..."
        avatar = f'<img src="data:image/png;base64,{CHATBOT_AVATAR_B64}" class="bot-avatar-img" />'
        typing_placeholder.markdown(
            f"""
            <div class="bot-msg-wrapper">
                {avatar}
                <div class="typing-bubble">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Consulta o modelo
        try:
            result = st.session_state.chatbot.query(user_msg)
            response = result["answer"].strip()

            # remove animação
            typing_placeholder.empty()

            # adiciona resposta
            st.session_state.messages.append({"role": "assistant", "content": response})

            st.rerun()

        except Exception as e:
            typing_placeholder.empty()
            st.error(f"❌ Erro ao responder: {e}")

    st.markdown("</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
