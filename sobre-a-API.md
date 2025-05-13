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
