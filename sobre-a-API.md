# Sobre a API:

***OpenAI ChatGPT***

**Hospedagem/Back-end:**
Pode ser feita em:

- AWS – robusto e escalável (ideal para produção).
- Heroku – fácil de configurar (ótimo para testes e MVPs).
- Servidor próprio (VPS) – mais controle, porém exige manutenção (ex: DigitalOcean).
  
**Frameworks auxiliares:**
*Back-end (API):*

- Flask – simples, ideal para projetos pequenos.
- FastAPI – moderno, rápido, ideal para APIs mais complexas e assíncronas.

**Front-end (Integração com usuário):**

- HTML / CSS / JavaScript.
- Ou frameworks modernos como React.js ou Next.js, se quiser algo mais dinâmico.

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


