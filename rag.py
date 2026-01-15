import time
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.manager import get_openai_callback

def extrair_numero_portaria(texto):
    """
    Busca por padrões como:
    - Portaria 123
    - Portaria nº 1234
    - Portaria n. 1.234
    Retorna apenas a string numérica limpa (ex: "1234").
    """
    # Regex explicada:
    # portaria\s+       -> palavra "portaria" seguida de espaço
    # (?:n[º°oNn]\.?\s*)? -> grupo opcional para "nº", "n°", "n.", "n "
    # ([\d\.]+)         -> captura dígitos e pontos (ex: 1.200)
    match = re.search(r"portaria\s+(?:n[º°oNn]\.?\s*)?([\d\.]+)", texto, re.IGNORECASE)
    
    if match:
        # Remove pontos de milhar (ex: "1.234" vira "1234") para bater com o banco
        return match.group(1).replace(".", "").lstrip("0")
    return None

def responder_com_rag(pergunta, retriever, openai_api_key):
    
    
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0)

    # --- Configuração do Prompt (SEU PROMPT ORIGINAL MANTIDO) ---
    prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado na análise de documentos institucionais públicos. Responda sempre em **português**, com base **exclusivamente no contexto abaixo**.

Seu papel é **encontrar respostas claras, diretas e com evidência textual**.
Se a pergunta pedir a data da publicação utilize a data por extenso, por exemplo, ao invés de 04/11/2025, use 04 de novembro de 2025. Se a pergunta pedir o número de uma portaria ou outro identificador, **forneça esse número explicitamente**, desde que esteja no contexto. Responda à pergunta do usuário usando documentos fornecidos no contexto. Se os documentos mencionarem 'Anexo', mas o texto do anexo não estiver presente, diga que não tem acesso ao conteúdo do anexo. Sempre faça referência ao ID do documento (entre colchetes, por exemplo [0],[1]) do documento que foi usado para fazer uma consulta. Perguntas sobre prazos, procure ao longo do texto, bem como a data da publicação. Perguntas sobre consequências procure no final do documento os últimos incisos dentro de disposições finais ou disposições gerais, pode ser que encontre deverá. Perguntas sobre fundamentação, procure as bases legais citadas no início do documento, normalmente iniciadas por considerando. Para público alvo procure discentes, docentes, servidores, estudantes, unidades/órgãos internos, câmpus, comunidade acadêmica, etc. Para perguntas sobre procedimentos procure por procedimentos, processos, regras, medidas, recomendações, orientações, cálculos ou fluxos. Para responsabilidades procure competências, responsabilidades, atribuições, deveres, obrigações, etc. Para síntese do documento, tente falar dos objetivos, público alvo, responsabilidades, procedimentos, prazos, se revoga ou altera outra portaria, consequencias, etc.
Se a pergunta mencionar uma portaria específica (número/ano), verifique se o contexto contém DOCUMENTO: Portaria nº X/ANO. Se não contiver, responda que não há evidência suficiente e peça para o sistema recuperar novamente.
Não invente nada. Se não encontrar a resposta no texto, responda: "Informação não encontrada nos documentos."

Contexto:
{context}

Pergunta:
{input}
""")

    # Criação das Chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # ============================
    # LÓGICA DE FILTRO DINÂMICO ✅
    # ============================
    
    # 1. Identifica número na pergunta
    numero_identificado = extrair_numero_portaria(pergunta)
    filtro = {"numero": numero_identificado} if numero_identificado else None

    

    # 2. Identifica qual objeto retriever manipular (Normal vs Comprimido/Self-RAG)
    retriever_alvo = None
    original_kwargs = {}

    if hasattr(retriever, "vectorstore"):
        # Caso 1: Retriever Padrão
        retriever_alvo = retriever
    elif hasattr(retriever, "base_retriever") and hasattr(retriever.base_retriever, "vectorstore"):
        # Caso 2: Self-RAG (Wrapper) - O filtro deve ir no base_retriever
        retriever_alvo = retriever.base_retriever

    # 3. Salva estado original (se formos aplicar filtro)
    if retriever_alvo and filtro:
        original_kwargs = retriever_alvo.search_kwargs.copy() if retriever_alvo.search_kwargs else {}

    try:
        # 4. Aplica o filtro temporariamente
        if retriever_alvo and filtro:
            new_kwargs = original_kwargs.copy()
            new_kwargs["filter"] = filtro
            retriever_alvo.search_kwargs = new_kwargs

        # 5. Execução (Invoke)
        with get_openai_callback() as cb:
            inicio = time.time()
            
            response = rag_chain.invoke({"input": pergunta})
            
            fim = time.time()

            resposta = response["answer"]
            fontes = response["context"]
                  
            tempo = fim - inicio
            tokens_totais = cb.total_tokens
            custo = cb.total_cost

        return resposta, fontes, tempo, tokens_totais, custo

    finally:
        # 6. Restaura estado original (limpeza)
        # Isso garante que a próxima pergunta não herde o filtro desta
        if retriever_alvo and filtro:
            # print("🧹 Limpando filtro...") 
            retriever_alvo.search_kwargs = original_kwargs