import os
import csv
import platform
import subprocess
from datetime import datetime
import streamlit as st
import pandas as pd  # Para exibir DataFrame

# === Seus módulos ===
from dados import carregar_dados
from rag import responder_com_rag
from llm import responder_com_llm
from advanced_rag import responder_com_advancedrag
from download import baixar_e_processar_portarias  
from limpar import gerar_portarias_limpas
# ------------------------------------------------------------
# CONFIGURAÇÕES BÁSICAS
# ------------------------------------------------------------
st.set_page_config(page_title="PerguntAI", layout="wide")

# API Key (por enquanto fixa aqui para dev ou via env)
# Idealmente, carregue de st.secrets ou variável de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-proj-...") 

# ------------------------------------------------------------
# SIDEBAR - CONFIGURAÇÕES E SCRAPING
# ------------------------------------------------------------

# 1. Diretório
destino_dir = os.path.abspath("dados_documentos")
os.makedirs(destino_dir, exist_ok=True)

persist_dir = os.path.join(
    destino_dir,
    "chroma_app",
    "openai_emb3small_cs1500_co600"
)
os.makedirs(persist_dir, exist_ok=True)


# 2. Web Scraping (NOVA SEÇÃO)
st.sidebar.header("🕷️ Coleta de Dados")
url_html = st.sidebar.text_input("URL das Portarias")

if st.sidebar.button("1. Executar Download"):
    if not url_html:
        st.sidebar.error("Por favor, informe uma URL.")
    else:
        with st.spinner("Iniciando Download... (Isso pode demorar)"):
            try:
                # Passamos 'st' como container para os logs aparecerem na tela principal
                caminho_gerado = baixar_e_processar_portarias(
                    destino_dir=destino_dir, 
                    url_html=url_html, 
                    st_container=st
                )
                st.sidebar.success("Download concluído!")
                st.success(f"Dados salvos em: {caminho_gerado}")
            except Exception as e:
                st.error(f"Erro no scraping: {e}")

st.sidebar.divider()

# 3. Utilitários
if st.sidebar.button("Abrir pasta dos dados"):
    dir_abs_path = os.path.abspath(destino_dir)
    try:
        sistema = platform.system()
        if sistema == "Windows":
            subprocess.Popen(f'explorer "{dir_abs_path}"')
        elif sistema == "Darwin":
            subprocess.Popen(["open", dir_abs_path])
        else:
            subprocess.Popen(["xdg-open", dir_abs_path])
    except Exception as e:
        st.sidebar.error(f"Erro ao abrir diretório: {e}")

# ------------------------------------------------------------
# ESTADO GLOBAL
# ------------------------------------------------------------
st.title("PerguntAI")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "df_portarias" not in st.session_state:
    st.session_state.df_portarias = None

# ------------------------------------------------------------
# ÁREA PRINCIPAL - INDEXAÇÃO E CHAT
# ------------------------------------------------------------

# Botão de Indexação (Passo 2)
st.markdown("### Processamento")
if st.button("2. Indexar/Processar Dados (Criar Base Vetorial)"):
    # Verifica se existe o arquivo gerado pelo passo 1 (Scraping) ou se já existia
    caminho_json_esperado = os.path.join(destino_dir, "portarias.json") 
    # Nota: O seu script de download gera 'portarias.json', mas o carregar_dados
    # original pedia 'portarias_limpas.json'. Se o download gerar o json pronto, 
    # certifique-se que o nome bate. 
    # Ajuste aqui conforme seu fluxo: se o download já limpa, use o mesmo nome.
    
    if not os.path.exists(caminho_json_esperado):
        st.warning(f"Arquivo `{caminho_json_esperado}` não encontrado.")
        st.info("Execute o Download na barra lateral primeiro ou verifique o nome do arquivo.")
    else:
        try:
            with st.spinner("Indexando documentos no ChromaDB..."):
                # Ajuste os parametros conforme sua função carregar_dados espera
                # Ex: assumindo que carregar_dados leia 'portarias.json' ou você renomeie
                caminho_json_limpo=gerar_portarias_limpas(caminho_json_esperado, destino_dir)
                df, retriever = carregar_dados(
                  diretorio_dados=destino_dir,
                  persist_dir=persist_dir,
                  openai_api_key=OPENAI_API_KEY,
                  embedding_backend="openai",
                  openai_embedding_model="text-embedding-3-small",
                  chunk_size=1500,
                  chunk_overlap=600,
                  k=10,
                  reuse_existing_index=True
                )

                
                st.session_state.df_portarias = df
                st.session_state.retriever = retriever
            st.success("Base vetorial atualizada e carregada na memória!")
        except Exception as e:
            st.error(f"Erro ao indexar dados: {e}")

st.divider()

# ------------------------------------------------------------
# BLOCO DE PERGUNTAS (Q&A)
# ------------------------------------------------------------
if st.session_state.retriever is None:
    st.info("⚠️ O sistema ainda não foi carregado. Execute o passo **2. Indexar/Processar Dados** acima.")
else:
    modo = st.radio(
        "Selecione o modo de resposta:",
        ["LLM (Puro)", "RAG Tradicional", "Advanced RAG"],
        horizontal=True,
    )

    st.markdown("### Faça sua pergunta")
    pergunta = st.text_input("Digite sua pergunta sobre as portarias:", placeholder="Ex: Qual portaria trata de...")

    if st.button("Enviar Pergunta"):
        if not pergunta.strip():
            st.warning("Digite uma pergunta válida.")
        else:
            trace = ""
            try:
                if "LLM" in modo:
                    resposta, fontes, tempo, tokens, custo = responder_com_llm(
                        pergunta, OPENAI_API_KEY
                    )

                elif "Advanced RAG" in modo:
                    resposta, fontes, tempo, tokens, custo, trace = responder_com_advancedrag(
                        pergunta, st.session_state.retriever, OPENAI_API_KEY
                    )

                else: # RAG Tradicional
                    # Debug visual opcional
                    with st.expander("🔍 Debug: Documentos recuperados (Raw Retrieval)"):
                        st.write(f"Consultando retriever para: '{pergunta}'")
                        try:
                            docs_debug = st.session_state.retriever.invoke(pergunta)
                            for i, doc in enumerate(docs_debug):
                                st.markdown(f"**Doc {i+1}** ({doc.metadata.get('source', '')})")
                                st.caption(doc.page_content[:300] + "...")
                        except Exception as e:
                            st.error(f"Erro no debug do retriever: {e}")

                    resposta, fontes, tempo, tokens, custo = responder_com_rag(
                        pergunta, st.session_state.retriever, OPENAI_API_KEY
                    )

                # --- Exibição da resposta ---
                st.success("🤖 Resposta:")
                st.markdown(resposta)

                # Trace / raciocínio (apenas Advanced)
                if trace:
                    with st.expander("🧠 Raciocínio (Self-Correction Trace)"):
                        st.text_area("Log", value=trace, height=200)

                # Fontes (RAGs)
                if fontes:
                    with st.expander("📚 Fontes utilizadas"):
                        for i, doc in enumerate(fontes, 1):
                            st.markdown(f"**Fonte {i}:** {doc.metadata.get('identificador', 'Doc')}")
                            st.caption(f"Trecho: {(doc.page_content or '')[:400]}...")

                # Métricas
                st.caption(f"⏱️ Tempo: {tempo:.2f}s | 🪙 Custo Est.: ${custo:.6f} | 🔠 Tokens: {tokens}")

                # Logging (Opcional)
                log_file = os.path.join(destino_dir, "log_respostas.csv")
                # (Lógica de log mantida simples aqui)
                
            except Exception as e:
                st.error(f"Erro ao processar resposta: {e}")