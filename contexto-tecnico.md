
# Contexto Técnico e Acadêmico

Este projeto aplica conceitos modernos de **Inteligência Artificial aplicada à educação e saúde**, por meio de um chatbot construído com a arquitetura **RAG (Retrieval-Augmented Generation)**. Essa abordagem permite que o modelo recupere informações de documentos PDF fornecidos pela equipe Nascentia e responda perguntas com base nesse conteúdo, utilizando a API da OpenAI para gerar respostas em linguagem natural.

## 🔹 Fundamentos Técnicos

- **LangChain Framework**: Utilizado para orquestrar os fluxos entre documentos, embeddings vetoriais (via FAISS) e o modelo de linguagem da OpenAI.
- **FAISS (Facebook AI Similarity Search)**: Responsável pela recuperação rápida de trechos relevantes dos documentos, com base na similaridade semântica.
- **OpenAI GPT**: Responsável pela geração de texto, utilizando os dados contextualizados recuperados dos documentos.
- **Boas Práticas de Segurança**: Variáveis sensíveis como chaves de API são gerenciadas por meio do arquivo `.env`, evitando exposição em repositórios públicos.

## 🔹 Relevância Acadêmica e Social

O chatbot visa atender de forma automatizada e empática dúvidas frequentes de gestantes e alunos das plataformas da Nascentia. A proposta está alinhada a três frentes:

1. **Educação a Distância**  
   O projeto responde a uma crescente demanda por soluções automatizadas de suporte educacional, garantindo disponibilidade 24/7 sem aumento de carga humana.  
   > *Referência: ALMEIDA, M. E. B. Educação a distância na era digital. Loyola, 2011.*

2. **Promoção da Saúde com Tecnologias Digitais**  
   Contribui com o acesso a informações seguras sobre saúde gestacional, dentro das recomendações da OMS para uso de tecnologia em cuidados primários.  
   > *Referência: WHO, Digital technologies: shaping the future of primary health care, 2018.*

3. **Interação Humano-Computador (IHC)**  
   Chatbots representam um avanço significativo em acessibilidade e usabilidade, promovendo inclusão digital e engajamento dos usuários.
