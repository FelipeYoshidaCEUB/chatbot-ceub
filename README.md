# Chatbot Nascentia - Projeto Unificado

Chatbot RAG (Retrieval-Augmented Generation) da Nascentia com suporte para múltiplos modelos de linguagem (OpenAI e HuggingFace).

## 🚀 Características

- **Múltiplos Modelos**: Suporte para OpenAI GPT-4o-mini e HuggingFace Qwen2.5-1.5B
- **Interface Streamlit**: Interface web moderna e intuitiva
- **Processamento de PDFs**: Carrega e processa múltiplos documentos PDF automaticamente
- **Memória de Conversa**: Mantém contexto da conversa durante a sessão
- **Citações Automáticas**: Inclui referências às fontes dos documentos
- **Índices FAISS Persistentes**: Salva e carrega índices separados para cada modelo
- **Seletor de Modelo**: Escolha entre diferentes modelos na interface

## 📋 Pré-requisitos

1. **Python 3.8+**
2. **Tokens de API**:
   - Para OpenAI: `OPENAI_API_KEY` no arquivo `.env`
   - Para HuggingFace: `HUGGINGFACEHUB_API_TOKEN` no arquivo `.env`
3. **Dependências**: Instalar as dependências do arquivo `requirements.txt`

## 🔧 Configuração

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com:

```env
OPENAI_API_KEY=sk-seu_token_openai_aqui
HUGGINGFACEHUB_API_TOKEN=hf_seu_token_huggingface_aqui
```

### 3. Preparar Documentos

1. Coloque seus arquivos PDF na pasta `data/`
2. O chatbot processará automaticamente todos os PDFs encontrados ao carregar o modelo

### 4. Executar a Aplicação

```bash
streamlit run streamlit_app.py
```

A aplicação estará disponível em `http://localhost:8501`

## 🤖 Modelos Disponíveis

### OpenAI (GPT-4o-mini)
- **Embeddings**: `text-embedding-3-small`
- **Chat**: `gpt-4o-mini`
- **Chunk Size**: 1500 caracteres
- **Chunk Overlap**: 200 caracteres

### HuggingFace (Qwen2.5-1.5B)
- **Embeddings**: `intfloat/multilingual-e5-large-instruct`
- **Chat**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Chunk Size**: 400 caracteres
- **Chunk Overlap**: 50 caracteres

## 📁 Estrutura de Arquivos

```
chatbot-ceub/
├── src/
│   ├── models/
│   │   ├── base_chatbot.py          # Classe base abstrata
│   │   ├── openai_chatbot.py        # Implementação OpenAI
│   │   └── huggingface_chatbot.py   # Implementação HuggingFace
│   └── utils/
│       ├── config.py                 # Configurações centralizadas
│       └── document_processor.py     # Processamento de documentos
├── data/                              # Diretório para PDFs
├── faiss_index/                       # Índices FAISS persistentes
│   ├── openai/                       # Índice para modelo OpenAI
│   └── huggingface/                   # Índice para modelo HuggingFace
├── streamlit_app.py                   # Interface Streamlit
├── requirements.txt                   # Dependências
├── .env                               # Variáveis de ambiente (criar)
└── README.md                          # Este arquivo
```

## 💻 Uso

1. **Iniciar a aplicação**: Execute `streamlit run streamlit_app.py`
2. **Selecionar modelo**: Na barra lateral, escolha entre OpenAI ou HuggingFace
3. **Carregar modelo**: Clique em "Carregar/Recarregar Modelo"
4. **Fazer upload de documentos** (opcional): Se ainda não houver índice, faça upload de PDFs
5. **Conversar**: Use a aba "Chat" para fazer perguntas
6. **Visualizar chunks**: Use a aba "Visualização" para ver os chunks indexados

## 🔄 Migração de Índices Existentes

Se você já tinha índices FAISS dos projetos anteriores:

- **OpenAI**: Copie os arquivos de `chatbot_openIA/faiss_index/` para `faiss_index/openai/`
- **HuggingFace**: Copie os arquivos de `chatbot_hugging/faiss_index/` para `faiss_index/huggingface/`

## 📝 Notas

- Cada modelo mantém seu próprio índice FAISS separado
- Os índices são criados automaticamente na primeira execução
- Documentos podem ser adicionados via upload na interface
- O histórico de conversa é mantido durante a sessão

## 👥 Integrantes

- Rafael Martins
- Felipe Yoshida
- Matheus Alves
- Mateus Bitar
- José Muller
- João Pedro Borges

## 📄 Licença

Este projeto é parte do Projeto Integrador III do curso de Ciência de Dados e Machine Learning – CEUB.

