import time
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

def responder_com_llm(pergunta, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4o-mini")

    with get_openai_callback() as cb:
        inicio = time.time()
        resposta = llm.invoke(pergunta)
        fim = time.time()

        tempo = fim - inicio
        tokens = cb.total_tokens
        custo = cb.total_cost

    return resposta.content, [], tempo, tokens, custo

