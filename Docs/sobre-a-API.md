# Sobre a API:

***OpenAI ChatGPT***

**Ferramentas de versionamento:**

- Git
- GitHub

**Integração com a API OpenAI ChatGPT:**

- Utiliza o endpoint: https://api.openai.com/v1/chat/completions.
- Envia mensagens (input do usuário) e recebe respostas do modelo.
- O back-end faz essa comunicação e repassa a resposta para o front-end.


## Como a API da OpenAI será usada no Chatbot da Nascentia
 - A API do ChatGPT vai gerar as respostas automáticas do chatbot.

- Quando o usuário enviar uma dúvida no site, o sistema:

*1- Envia a pergunta para a API da OpenAI.*

*2- Recebe uma resposta inteligente e em português.*

*3- Exibe essa resposta na interface do chatbot.*

- O chatbot será treinado com as FAQs fornecidas pela Nascentia.

- A comunicação será feita por um back-end em Python (Flask ou FastAPI).


