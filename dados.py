import os
import json
import pandas as pd
import traceback

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


def _mask(s: str, keep: int = 6) -> str:
    if not s:
        return "<vazio>"
    s = str(s)
    return s[:keep] + "…" + f"({len(s)} chars)"


def _has_chroma_index(persist_dir: str) -> bool:
    """Heurística simples para detectar índice persistido do Chroma."""
    try:
        if not persist_dir or not os.path.exists(persist_dir):
            return False
        if not os.path.isdir(persist_dir):
            return False
        files = os.listdir(persist_dir)
        if not files:
            return False
        # Marcadores comuns (podem variar por versão)
        markers = {"chroma.sqlite3", "index", "collections", "embeddings_queue"}
        if any(m in files for m in markers):
            return True
        # Se tem qualquer arquivo sqlite / bin, é provável que exista índice
        if any(name.endswith((".sqlite3", ".bin", ".parquet")) for name in files):
            return True
        # Direto e simples: pasta não vazia
        return True
    except Exception:
        return False


def _get_embeddings(
    backend: str,
    openai_api_key: str | None = None,
    openai_model: str = "text-embedding-3-small",
    hf_model_name: str = "intfloat/multilingual-e5-small",
    hf_device: str | None = None,
):
    backend = (backend or "openai").lower().strip()

    if backend == "hf":
        model_kwargs = {}
        if hf_device:
            model_kwargs["device"] = hf_device

        # normalize_embeddings ajuda bastante na busca por cosseno
        encode_kwargs = {"normalize_embeddings": True}

        return HuggingFaceEmbeddings(
            model_name=hf_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    # default: openai
    key = (openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY ausente para embeddings OpenAI.")
    return OpenAIEmbeddings(api_key=key, model=openai_model)

def _normalize_numero(numero: str | int | None, ano: int | str | None):
    if numero is None:
        return None
    num = str(numero).strip()
    try:
        a = int(ano) if ano is not None else None
    except Exception:
        a = None

    # Heurística: se ano>=2021 e número veio como 6 dígitos começando com "20",
    # provavelmente era "%20" + "NNNN" -> "20NNNN"
    if a is not None and a >= 2021 and len(num) == 6 and num.startswith("20"):
        return num[2:]
    return num

def carregar_dados(
    diretorio_dados: str,
    persist_dir: str,
    openai_api_key: str | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 600,
    k: int = 10,
    embedding_backend: str = "openai",  # "openai" ou "hf"
    hf_model_name: str = "intfloat/multilingual-e5-small",
    hf_device: str | None = None,
    openai_embedding_model: str = "text-embedding-3-small",
    reuse_existing_index: bool = True,
):
    """
    Carrega portarias_limpas.json, faz chunking + injeção de cabeçalho em cada chunk,
    cria/abre Chroma e retorna (df, retriever).

    Otimização:
      - Se reuse_existing_index=True e persist_dir já tiver índice, apenas ABRE o índice
        (não reprocessa nem reindexa), o que economiza bastante tempo no grid.

    Para reduzir custo no grid:
      - embedding_backend="hf" (e mantém OpenAI só para geração nas funções de resposta)
    """

    # 0) Caminho do JSON
    diretorio_abs = os.path.abspath(diretorio_dados)
    caminho_json = os.path.join(diretorio_abs, "portarias_limpas.json")
    print(f"📄 Carregando JSON de: {os.path.abspath(caminho_json)}")

    if not os.path.exists(caminho_json):
        raise FileNotFoundError(f"Arquivo {caminho_json} não encontrado.")

    with open(caminho_json, "r", encoding="utf-8") as f:
        portarias = json.load(f)

    df = pd.DataFrame(portarias)
    print(f"📊 {len(portarias)} documentos carregados do JSON.")

    # 1) API key (para geração) — não é necessária para HF embeddings
    key = (openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    print(f"🔐 OPENAI_API_KEY presente? {bool(key)} | valor(mascarado): {_mask(key)}")

    # 2) Versões
    try:
        import langchain, langchain_core, langchain_openai, chromadb
        print(
            "📦 versions:",
            "langchain", getattr(langchain, "__version__", "?"),
            "| core", getattr(langchain_core, "__version__", "?"),
            "| openai", getattr(langchain_openai, "__version__", "?"),
            "| chromadb", getattr(chromadb, "__version__", "?"),
        )
    except Exception as _e:
        print(f"⚠️ Falha ao obter versões: {_e}")

    # 3) Embeddings (HF ou OpenAI)
    try:
        print(f"🧠 Embeddings backend: {embedding_backend}")
        embeddings = _get_embeddings(
            backend=embedding_backend,
            openai_api_key=key,
            openai_model=openai_embedding_model,
            hf_model_name=hf_model_name,
            hf_device=hf_device,
        )

        # Probe opcional (só OpenAI; HF pode demorar na primeira carga e é ok)
        if (embedding_backend or "").lower().strip() == "openai":
            print("Realizando teste (probe) com embeddings OpenAI...")
            probe = embeddings.embed_query("ping")
            print(f"✅ probe_len: {len(probe)} | exemplo primeiros 5: {probe[:5]}")
            if not probe:
                raise RuntimeError("Embedder retornou vetor vazio no probe.")

    except Exception:
        print("❌ Falha ao inicializar embeddings:")
        traceback.print_exc()
        raise

    # 4) Reuso do índice (economiza MUITO tempo em re-runs)
    os.makedirs(persist_dir, exist_ok=True)
    if reuse_existing_index and _has_chroma_index(persist_dir):
        print(f"♻️ Reutilizando índice existente em: {os.path.abspath(persist_dir)}")
        try:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": int(k)})
            print(f"✅ Índice carregado | persist_dir: {os.path.abspath(persist_dir)} | k={k}")
            return df, retriever
        except Exception:
            print("⚠️ Falha ao abrir índice existente; vou recriar.")
            traceback.print_exc()

    # 5) Documentos brutos (para indexação nova)
    docs_brutos: list[Document] = []

    for item in portarias:
      numero = _normalize_numero(item.get("numero"), item.get("ano"))
      ano = item.get("ano")
      data = item.get("data")
      ementa = item.get("ementa")
      orgao = item.get("orgao")
      url_pdf = item.get("url_pdf")
      texto = item.get("conteudo_limpo") or ""

      source = (
        f"Portaria nº {numero}/{ano}"
        if numero and ano
        else (item.get("identificador") or "Portaria (id desconhecido)")
      )

      docs_brutos.append(
        Document(
            page_content=texto,
            metadata={
                "source": source,
                "numero": numero,
                "ano": ano,
                "data": data,
                "ementa": ementa,
                "orgao": orgao,
                "url_pdf": url_pdf,
                "texto_item": item.get("texto_item"),
                "arquivo": item.get("arquivo"),
            },
        )
      )

    # 6) Chunking (usa os parâmetros recebidos)
    print(f"Dividindo documentos em chunks | chunk_size={chunk_size} overlap={chunk_overlap} ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["Art.", "Seção", "O REITOR", "A REITORA", "\n\n", ". "],
    )
    docs = splitter.split_documents(docs_brutos)

    # 7) Injeção de cabeçalho em cada chunk
    print(f"Injetando metadados de contexto em {len(docs)} chunks...")
    for chunk in docs:
        meta = chunk.metadata or {}
        cabecalho = (
          f"DOCUMENTO: {meta.get('source', 'N/A')} | "
          f"ÓRGÃO: {meta.get('orgao', 'N/A')} | "
          f"DATA: {meta.get('data', 'S/D')} | "
          f"EMENTA: {meta.get('ementa', 'N/A')} | "
          f"CONTEÚDO: "
        )


        chunk.page_content = cabecalho + (chunk.page_content or "")

    if not docs:
        raise ValueError("Nenhum documento após o split — verifique portarias_limpas.json.")

    # 8) Indexação (criação nova) em lotes para não estourar limite do OpenAI
    print(f"Indexando {len(docs)} chunks no ChromaDB em lotes...")

    try:
      vectordb = Chroma(
        collection_name="portarias",
        persist_directory=persist_dir,
        embedding_function=embeddings,  # importante: embedding_function aqui
      )

      batch_size = 64  # ajuste seguro (32/64/128). Quanto maior, mais rápido; quanto menor, mais seguro.
      for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        vectordb.add_documents(batch)
        if i == 0 or (i + batch_size) % (batch_size * 10) == 0:
            print(f"  -> {min(i+batch_size, len(docs))}/{len(docs)} chunks adicionados")

      try:
        vectordb.persist()
      except Exception as e:
        print(f"⚠️ vectordb.persist() falhou/indisponível: {e}")

      retriever = vectordb.as_retriever(search_kwargs={"k": k})

      print(f"✅ Indexação OK | persist_dir: {os.path.abspath(persist_dir)} | k={k}")
      return df, retriever

    except Exception:
      print("❌ Erro no Chroma.add_documents / persist:")
      traceback.print_exc()
      raise
