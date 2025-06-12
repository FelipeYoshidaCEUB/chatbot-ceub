# Código simples para testar a API se está funcionando

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def perguntar_llm(pergunta):
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "Você é um assistente simpático."},
                {"role": "user", "content": pergunta}
            ]
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro ao consultar API: {e}"

if __name__ == "__main__":
    print("Chatbot iniciado. Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando chatbot. Até mais!")
            break
        resposta = perguntar_llm(pergunta)
        print(f"Assistente: {resposta}")
