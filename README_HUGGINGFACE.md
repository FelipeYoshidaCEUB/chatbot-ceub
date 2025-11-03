# Chatbot Nascentia - HuggingFace

Chatbot RAG (Retrieval-Augmented Generation) da Nascentia utilizando modelos do HuggingFace para processamento de documentos PDF e conversação inteligente.

## 🚀 Características

- **Modelos HuggingFace**: Utiliza modelos de embeddings e chat do HuggingFace
- **Processamento de PDFs**: Carrega e processa múltiplos documentos PDF automaticamente
- **Memória de Conversa**: Mantém contexto da conversa durante a sessão
- **Citações Automáticas**: Inclui referências às fontes dos documentos
- **Índice FAISS Persistente**: Salva e carrega índices para reutilização
- **Interface de Linha de Comando**: Simples e direta para uso em terminal

## 📋 Pré-requisitos

1. **Python 3.8+**
2. **Token do HuggingFace**: Necessário para acessar os modelos
3. **Dependências**: Instalar as dependências do arquivo `requirements_huggingface.txt`
4. **Documentos PDF**: Colocar arquivos PDF na pasta `data/`

## 🔧 Configuração

### 1. Instalar Dependências

```bash
pip install -r requirements_huggingface.txt
```

### 2. Configurar Token do HuggingFace

1. Crie um arquivo `.env` na raiz do projeto
2. Adicione seu token do HuggingFace:

```env
HUGGINGFACEHUB_API_TOKEN=hf_seu_token_aqui
```

### 3. Preparar Documentos

1. Crie a pasta `data/` na raiz do projeto
2. Coloque seus arquivos PDF na pasta `data/`
3. O chatbot processará automaticamente todos os PDFs encontrados

### 4. Executar a Aplicação

```bash
python scr/chatbot_hugging.py
```

## 🤖 Modelos Utilizados

- **Embeddings**: `intfloat/multilingual-e5-large-instruct`
- **Chat**: `meta-llama/Llama-3.2-1B-Instruct`

## 📁 Estrutura de Arquivos

```
chatbot_hugging/
├── scr/
│   ├── chatbot_hugging.py         # Aplicação principal
│   └── faiss_index/              # Índice FAISS persistente
├── data/                         # Diretório para PDFs
├── requirements_huggingface.txt  # Dependências
├── README_HUGGINGFACE.md        # Este arquivo
└── .env                         # Configurações (criar)
```

## 💡 Como Usar

### 1. Preparação
- Coloque seus PDFs na pasta `data/`
- Configure o token do HuggingFace no arquivo `.env`
- Execute o script

### 2. Processamento Automático
- O chatbot carrega automaticamente todos os PDFs da pasta `data/`
- Cria chunks de texto com sobreposição
- Gera embeddings e salva o índice FAISS

### 3. Conversação
- Digite suas perguntas no terminal
- O chatbot responde baseado nos documentos
- Use `sair`, `exit` ou `quit` para encerrar

### 4. Citações
- Todas as respostas incluem referências às fontes
- Formato: `[Fonte: nome_do_arquivo.pdf p.X]`

## ⚙️ Configurações

Você pode modificar as configurações no arquivo `chatbot_hugging.py`:

```python
# Modelos
EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
CHAT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Parâmetros de chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Número de chunks recuperados
search_kwargs={"k": 2}
```

## 🔍 Funcionalidades

### Processamento de Documentos
- Carregamento automático de PDFs
- Chunking inteligente com sobreposição
- Metadados de fonte e página
- Índice FAISS persistente

### Chat Inteligente
- Memória de conversa nativa do LangChain
- Citações automáticas das fontes
- Respostas baseadas exclusivamente nos documentos
- Interface de linha de comando simples

### Persistência
- Índice FAISS salvo automaticamente
- Reutilização de índices existentes
- Atualização incremental de documentos

## 🚨 Limitações

- Requer conexão com internet para carregar modelos
- Primeira execução pode ser lenta (download dos modelos)
- Modelos podem consumir memória RAM
- Token do HuggingFace necessário
- Interface apenas de linha de comando

## 🛠️ Solução de Problemas

### Erro de Token
```
❌ Token do Hugging Face não encontrado no .env!
```
**Solução**: Verifique se o arquivo `.env` existe e contém o token correto.

### Erro de Diretório
```
ERRO: Diretório data não encontrado!
```
**Solução**: Crie a pasta `data/` e coloque seus PDFs nela.

### Erro de PDFs
```
ERRO: Nenhum arquivo PDF encontrado em data
```
**Solução**: Adicione arquivos PDF na pasta `data/`.

### Erro de Memória
```
CUDA out of memory
```
**Solução**: Os modelos estão configurados para CPU. Para GPU, modifique as configurações.

## 📊 Exemplo de Uso

```bash
$ python scr/chatbot_hugging.py

Carregando modelo de embeddings Hugging Face...
Carregando índice FAISS existente...
Carregando modelo de chat Hugging Face...
✅ Modelos Hugging Face carregados com sucesso!

🤖 Chatbot RAG com Hugging Face e memória pronta! (Digite 'sair' para encerrar)

Você: O que é a Nascentia?
Assistente: A Nascentia é uma empresa especializada em cuidados de parto, pré-natal e pós-parto. [Fonte: O que é a Nascentia.pdf p.1]

Você: Quais são os cuidados na gestação?
Assistente: Durante a gestação, é importante manter uma alimentação equilibrada, praticar exercícios adequados e fazer acompanhamento médico regular. [Fonte: Cuidados na gestação.pdf p.2]

Você: sair
Até logo! 👋
```

## 📞 Suporte

Para dúvidas ou problemas, verifique:
1. Se todas as dependências estão instaladas
2. Se o token do HuggingFace está correto
3. Se há arquivos PDF na pasta `data/`
4. Se a conexão com internet está funcionando
5. Se há espaço suficiente em disco para os modelos