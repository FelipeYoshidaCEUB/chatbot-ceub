
# CONTEXTO TÉCNICO E ACADÊMICO DO PROJETO

--------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Fundamentos Técnicos

Este projeto aplica conceitos modernos de inteligência artificial aplicada à educação e saúde, por meio de um chatbot construído com a arquitetura RAG (Retrieval-Augmented Generation). Essa abordagem permite que o modelo recupere informações de documentos PDF fornecidos pela equipe Nascentia e responda perguntas com base nesse conteúdo, utilizando a API da OpenAI para gerar respostas em linguagem natural.

LangChain foi utilizado para orquestrar os fluxos entre documentos, embeddings vetoriais (FAISS) e o modelo de linguagem da OpenAI. O FAISS é responsável pela recuperação eficiente de trechos relevantes com base em similaridade semântica. As variáveis sensíveis como chaves de API são armazenadas em arquivo .env, seguindo boas práticas de segurança.

--------------------------------------------------------------------------------------------------------------------------------------------------------------

2. Tecnologias Empregadas

Linguagem de Programação: Python

Framework de IA: LangChain

Armazenamento vetorial: FAISS

Geração de texto: OpenAI GPT (via API)

Carregamento de documentos: PyPDFLoader

Gerenciamento de ambiente: dotenv (.env)

--------------------------------------------------------------------------------------------------------------------------------------------------------------

3. Relevância Acadêmica e Social

O chatbot visa atender de forma automatizada e empática dúvidas frequentes de gestantes e alunos das plataformas da Nascentia. A proposta está alinhada a três frentes:

1. Educação a Distância
O projeto responde à crescente demanda por soluções automatizadas de suporte educacional, garantindo disponibilidade 24/7 sem aumento de carga humana.
Referência: ALMEIDA, M. E. B. Educação a distância na era digital. Loyola, 2011.

2. Promoção da Saúde com Tecnologias Digitais
Contribui com o acesso a informações seguras sobre saúde gestacional, dentro das recomendações da OMS para uso de tecnologia em cuidados primários.
Referência: WHO. Digital technologies: shaping the future of primary health care. Geneva, 2018.

3. Interação Humano-Computador (IHC)
Chatbots representam um avanço em acessibilidade e usabilidade, promovendo inclusão digital e engajamento dos usuários.

--------------------------------------------------------------------------------------------------------------------------------------------------------------

4. Contribuição Esperada

A proposta técnica viabiliza um sistema que utiliza inteligência artificial para resolver um problema real de comunicação, com base em dados documentais fornecidos por especialistas da área da saúde. A aplicação combina eficiência, acessibilidade e alinhamento com princípios éticos e educacionais.
