import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback

# Imports para Re-ranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# Importa a função RAG e o extrator de número (do arquivo anterior)
from rag import responder_com_rag, extrair_numero_portaria

def responder_com_advancedrag(pergunta, retriever, openai_api_key):
    llm = ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-4o-mini")
    
    # --- 0. Configuração do Re-ranking (Flashrank) ---
    compressor = FlashrankRerank(top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    # --- Lógica de Filtro (Pré-Recuperação) ---
    # Precisamos filtrar AQUI também, para que o Grader veja os docs certos.
    numero_identificado = extrair_numero_portaria(pergunta)
    filtro = {"numero": numero_identificado} if numero_identificado else None
    
    # Salva kwargs originais para restaurar no final (try/finally)
    original_kwargs = {}
    base_retriever_ref = None

    # Tenta acessar o retriever base para aplicar o filtro
    if hasattr(retriever, "vectorstore") and filtro:
        base_retriever_ref = retriever
        original_kwargs = base_retriever_ref.search_kwargs.copy()
        new_kwargs = original_kwargs.copy()
        new_kwargs["filter"] = filtro
        base_retriever_ref.search_kwargs = new_kwargs
    
    try:
        # --- 1. Recuperação Inicial (Com Re-ranking + Filtro) ---
        # Busca docs para o Grader avaliar
        docs = compression_retriever.invoke(pergunta)
        
        if not docs:
            primeiro_doc_conteudo = "Nenhum documento encontrado."
        else:
            primeiro_doc_conteudo = docs[0].page_content

        # --- 2. GRADER: Avalia relevância ---
        grader_prompt = PromptTemplate(
            template="""Você é um avaliador de relevância. O documento abaixo responde à pergunta "{pergunta}"? 
            Responda apenas 'SIM' ou 'NAO'.
            
            Documento: {documento}
            """,
            input_variables=["pergunta", "documento"]
        )
        chain_grader = grader_prompt | llm | StrOutputParser()
        
        # Avalia o top-1
        score = chain_grader.invoke({"pergunta": pergunta, "documento": primeiro_doc_conteudo})
        
        custo_extra = 0
        trace = f"Pergunta Original: {pergunta}\n"
        trace += f"Filtro Portaria: {numero_identificado if numero_identificado else 'Nenhum'}\n"
        trace += f"Retrieval (Flashrank): {len(docs)} docs recuperados.\n"
        trace += f"Decisão do Grader (baseado no Top-1): {score}\n"
        
        pergunta_final = pergunta
        
        # --- 3. Lógica de Correção (Self-Correction) ---
        if "NAO" in score.upper() or not docs:
            trace += ">> Documentos irrelevantes (ou ausentes). Reescrevendo pergunta...\n"
            
            rewrite_prompt = PromptTemplate.from_template(
                "Reescreva a pergunta para torná-la mais eficaz na busca vetorial por similaridade. Mantenha números de portarias se existirem. Pergunta: {pergunta}"
            )
            chain_rewrite = rewrite_prompt | llm | StrOutputParser()
            
            with get_openai_callback() as cb_rewrite:
                pergunta_final = chain_rewrite.invoke({"pergunta": pergunta})
                custo_extra = cb_rewrite.total_cost
                
            trace += f">> Nova Pergunta Gerada: {pergunta_final}\n"
        else:
            trace += ">> Documentos relevantes. Mantendo pergunta original.\n"

        # --- 4. Geração Final ---
        # Chamamos o responder_com_rag. 
        # IMPORTANTE: Ele aplicará o filtro novamente internamente (o que é seguro).
        resposta, fontes, tempo_rag, tokens_rag, custo_rag = responder_com_rag(
            pergunta_final, compression_retriever, openai_api_key
        )
        
        return resposta, fontes, tempo_rag, tokens_rag, custo_rag + custo_extra, trace

    finally:
        # Restaura o retriever ao estado original para não afetar próximas execuções
        if base_retriever_ref and original_kwargs:
            base_retriever_ref.search_kwargs = original_kwargs