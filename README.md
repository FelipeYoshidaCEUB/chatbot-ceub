# Chatbot Nascentia 

Chatbot RAG (Retrieval-Augmented Generation) da Nascentia especializado em parto, pré-natal e pós-parto.

## 📦 Versões Disponíveis

O projeto possui duas versões:

### 🎯 Versão Final (`chatbot.py`)
- **Modelo**: OpenAI GPT-4o-mini (fixo, único modelo disponível)
- **Interface**: Interface web moderna e personalizada com tema da Nascentia
- **Uso**: Versão de produção, recomendada para uso final
- **Características**: 
  - Interface otimizada com design customizado da marca Nascentia
  - Experiência de usuário aprimorada
  - Chat interativo com histórico de conversa
  - Processamento automático de documentos PDF da pasta `data/`
  - Índices FAISS persistentes

### 🧪 Versão de Desenvolvimento (`chatbot (dev).py`)
- **Modelos**: Suporte para OpenAI e múltiplos modelos HuggingFace
- **Interface**: Interface completa com funcionalidades de teste e desenvolvimento
- **Uso**: Versão para testes, comparação de modelos e desenvolvimento
- **Características**: 
  - Seletor de modelos (OpenAI ou HuggingFace)
  - Múltiplos modelos HuggingFace disponíveis (leves, médios e pesados)
  - Visualização de chunks indexados
  - Upload de documentos PDF via interface
  - Estatísticas do índice FAISS
  - Análise e exportação de dados dos chunks

## 🚀 Características

- **Interface Streamlit**: Interface web moderna e intuitiva
- **Processamento de PDFs**: Carrega e processa múltiplos documentos PDF automaticamente
- **Memória de Conversa**: Mantém contexto da conversa durante a sessão
- **Citações Automáticas**: Inclui referências às fontes dos documentos
- **Índices FAISS Persistentes**: Salva e carrega índices separados para cada modelo
- **Design Personalizado**: Interface customizada com tema da marca Nascentia (versão final)

## 📋 Pré-requisitos

1. **Python 3.8+**
2. **Tokens de API**:
   - Para OpenAI: `OPENAI_API_KEY` no arquivo `.env`
   - Para HuggingFace: `HUGGINGFACEHUB_API_TOKEN` no arquivo `.env`
3. **Dependências**: Instalar as dependências do arquivo `requirements.txt`

## 🔧 Configuração

### 1. Criar e Ativar Ambiente Virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com:

```env
OPENAI_API_KEY=sk-seu_token_openai_aqui
HUGGINGFACEHUB_API_TOKEN=hf_seu_token_huggingface_aqui
```

### 4. Preparar Documentos

1. Coloque seus arquivos PDF na pasta `data/`
2. O chatbot processará automaticamente todos os PDFs encontrados ao carregar o modelo

### 5. Executar a Aplicação

**Versão Final (Produção):**
```bash
streamlit run chatbot.py
```

**Versão de Desenvolvimento:**
```bash
streamlit run "chatbot (dev).py"
```

A aplicação estará disponível em `http://localhost:8501`

## 🤖 Modelos Disponíveis

### Versão Final (`chatbot.py`)
- **Modelo**: OpenAI GPT-4o-mini (único modelo disponível)
  - **Embeddings**: `text-embedding-3-small`
  - **Chat**: `gpt-4o-mini`
  - **Chunk Size**: 1500 caracteres
  - **Chunk Overlap**: 200 caracteres

### Versão de Desenvolvimento (`chatbot (dev).py`)

#### OpenAI
- **Embeddings**: `text-embedding-3-small`
- **Chat**: `gpt-4o-mini`
- **Chunk Size**: 1500 caracteres
- **Chunk Overlap**: 200 caracteres

#### HuggingFace (Múltiplos modelos disponíveis)

**Modelos Leves:**
- `Qwen/Qwen2.5-0.5B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-2-2b-it`

**Modelos Médios:**
- `Qwen/Qwen2.5-1.5B-Instruct` (padrão)
- `Qwen/Qwen2.5-3B-Instruct`

**Modelos Pesados:**
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen2.5-7B-Instruct`

**Configuração padrão HuggingFace:**
- **Embeddings**: `intfloat/multilingual-e5-large-instruct`
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
├── chatbot.py                         # Versão final (produção) - OpenAI apenas
├── chatbot (dev).py                   # Versão de desenvolvimento - OpenAI + HuggingFace
├── requirements.txt                   # Dependências
├── .env                               # Variáveis de ambiente (criar)
└── README.md                          # Este arquivo
```

## 💻 Uso

### Versão Final (`chatbot.py`)

1. **Iniciar a aplicação**: Execute `streamlit run chatbot.py`
2. **Conversar**: O modelo OpenAI já está carregado automaticamente
3. **Documentos**: Coloque os PDFs na pasta `data/` antes de iniciar (serão processados automaticamente)

### Versão de Desenvolvimento (`chatbot (dev).py`)

1. **Iniciar a aplicação**: Execute `streamlit run "chatbot (dev).py"`
2. **Selecionar modelo**: Na barra lateral, escolha entre OpenAI ou HuggingFace
3. **Selecionar modelo HuggingFace** (se aplicável): Escolha entre os modelos disponíveis (leves, médios ou pesados)
4. **Carregar modelo**: Clique em "Carregar/Recarregar Modelo"
5. **Fazer upload de documentos** (opcional): Se ainda não houver índice, faça upload de PDFs
6. **Conversar**: Use a aba "Chat" para fazer perguntas
7. **Visualizar chunks**: Use a aba "Visualização" para ver os chunks indexados

## 🔄 Migração de Índices Existentes

Se você já tinha índices FAISS dos projetos anteriores:

- **OpenAI**: Copie os arquivos de `chatbot_openIA/faiss_index/` para `faiss_index/openai/`
- **HuggingFace**: Copie os arquivos de `chatbot_hugging/faiss_index/` para `faiss_index/huggingface/`

## 📝 Notas

- Cada modelo mantém seu próprio índice FAISS separado
- Os índices são criados automaticamente na primeira execução
- **Versão Final**: Documentos devem ser colocados na pasta `data/` antes de iniciar
- **Versão de Desenvolvimento**: Documentos podem ser adicionados via upload na interface
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

