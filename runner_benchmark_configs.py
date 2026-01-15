import os
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict
import random

from dados import carregar_dados
from avaliador_benchmarking_offline import avaliar_tecnologias, carregar_golden_set


# Remove proxies (evita bugs de cliente http)
for k in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy","OPENAI_PROXY","openai_proxy"]:
    os.environ.pop(k, None)

# =========================================
# CONFIG BÁSICA
# =========================================
DIRETORIO_DADOS = os.path.abspath("dados_documentos")

# Embeddings para o grid: HF (grátis)
#EMBEDDING_BACKEND = "hf"
#HF_MODEL_NAME = "intfloat/multilingual-e5-small"
#HF_DEVICE = None  # "cuda" se tiver GPU

EMBEDDING_BACKEND = "openai"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # ou large
HF_MODEL_NAME = None
HF_DEVICE = None

# Pasta base dos índices
#PERSIST_BASE = os.path.join(DIRETORIO_DADOS, "chroma_grid")

PERSIST_BASE = os.path.join(DIRETORIO_DADOS, "chroma_grid_openai_emb3small")

# CSV consolidado
OUTPUT_CSV = os.path.join(DIRETORIO_DADOS, "benchmark_consolidado.csv")

# OpenAI key (geração das respostas)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# =========================================
# MÉTRICAS EXTRA (opcional)
# =========================================
COMPUTE_BERTSCORE = True

# =========================================
# AMOSTRAGEM DO GOLDEN SET (estratificada)
# =========================================
TIPOS_ITEM_FIXOS = [
    "complexa_prazos",
    "complexa_revogacoes",
    "direta_objetivo",
    "direta_publico",
    "interpretativa_consequencias",
    "interpretativa_fundamentacao",
    "interpretativa_procedimentos",
    "interpretativa_responsabilidades",
]

def amostrar_golden_estratificado(
    golden_set: list[dict],
    n_por_tipo: int,
    seed: int = 42,
    tipos_item: list[str] = TIPOS_ITEM_FIXOS,
):
    """
    Retorna uma amostra estratificada do golden_set,
    com n_por_tipo perguntas para cada tipo_item.
    """
    random.seed(seed)

    grupos = defaultdict(list)
    for item in golden_set:
        tipo = item.get("tipo_item") or item.get("tipo")
        if tipo in tipos_item:
            grupos[tipo].append(item)

    amostra = []
    avisos = []

    for tipo in tipos_item:
        itens = grupos.get(tipo, [])
        if len(itens) >= n_por_tipo:
            selecionados = random.sample(itens, n_por_tipo)
        else:
            selecionados = itens
            avisos.append(f"⚠️ tipo_item='{tipo}' tem apenas {len(itens)} perguntas (esperado {n_por_tipo})")

        amostra.extend(selecionados)

    if avisos:
        print("\n".join(avisos))

    print(
        f"🎯 Golden set estratificado: {len(amostra)} perguntas "
        f"({n_por_tipo} por tipo_item × {len(tipos_item)} tipos)"
    )
    return amostra

# =========================================
# GRID (OBS: índice NÃO depende de k)
# =========================================
GRID_CHUNK_SIZE = [800, 1500, 3000]
GRID_CHUNK_OVERLAP = [160, 300, 600]
GRID_K = [3, 5, 10]

def _index_id(chunk_size: int, chunk_overlap: int) -> str:
    """Identificador do índice (só depende de chunk_size e overlap)."""
    return f"cs{chunk_size}_co{chunk_overlap}"

def _config_id(chunk_size: int, chunk_overlap: int, k: int) -> str:
    """Identificador da configuração avaliada (inclui k)."""
    return f"{_index_id(chunk_size, chunk_overlap)}_k{k}"

def _append_csv(df_new: pd.DataFrame, csv_path: str):
    if not os.path.exists(csv_path):
        df_new.to_csv(csv_path, index=False, encoding="utf-8")
        return
    df_new.to_csv(csv_path, index=False, mode="a", header=False, encoding="utf-8")

def _done_config_ids(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["config_id"])
        return set(df["config_id"].dropna().unique().tolist())
    except Exception:
        return set()

def _preparar_retriever(chunk_size: int, chunk_overlap: int, k_inicial: int = 5):
    """
    Prepara o índice UMA VEZ por (chunk_size, chunk_overlap).
    Depois você só muda retriever.search_kwargs['k'] no loop.
    """
    persist_dir = os.path.join(PERSIST_BASE, _index_id(chunk_size, chunk_overlap))
    os.makedirs(persist_dir, exist_ok=True)

    # k_inicial só define o retriever retornado; o índice gerado é o mesmo
    _, retriever = carregar_dados(
        diretorio_dados=DIRETORIO_DADOS,
        persist_dir=persist_dir,
        openai_api_key=OPENAI_API_KEY,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k_inicial,
        embedding_backend=EMBEDDING_BACKEND,
        hf_model_name=HF_MODEL_NAME,
        hf_device=HF_DEVICE,
        openai_embedding_model=OPENAI_EMBEDDING_MODEL,
        reuse_existing_index=False,  # primeira execução com OpenAI
    )
    return retriever, persist_dir

def executar_grid_search(tipo_golden: str = "normal"):
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY não definida. Tecnologias de geração podem falhar.")

    os.makedirs(PERSIST_BASE, exist_ok=True)

    golden_set, _ = carregar_golden_set(DIRETORIO_DADOS, tipo_golden)

    # Ajuste aqui:
    N_POR_TIPO = 2
    SEED = 42
    golden_set = amostrar_golden_estratificado(golden_set, n_por_tipo=N_POR_TIPO, seed=SEED)

    print(f"🎯 Golden set: {len(golden_set)} perguntas (estratificado)")

    done = _done_config_ids(OUTPUT_CSV)

    indices = [(cs, co) for cs in GRID_CHUNK_SIZE for co in GRID_CHUNK_OVERLAP]
    total_cfgs = len(indices) * len(GRID_K)

    print(f"🔬 Total de configurações: {total_cfgs} | já concluídas: {len(done)}")
    print(f"📄 CSV consolidado: {OUTPUT_CSV}")

    for i_idx, (chunk_size, chunk_overlap) in enumerate(indices, start=1):
        print("\n" + "=" * 60)
        print(f"🧱 Índice {i_idx}/{len(indices)} | {_index_id(chunk_size, chunk_overlap)}")
        print("=" * 60)

        # 1) prepara índice uma vez
        try:
            retriever, persist_dir = _preparar_retriever(chunk_size, chunk_overlap, k_inicial=max(GRID_K))
            print(f"✅ Retriever pronto | persist_dir: {persist_dir}")
        except Exception as e:
            print(f"❌ Falha ao preparar índice {_index_id(chunk_size, chunk_overlap)}: {type(e).__name__}: {e}")

            # registra falha para TODAS as configs k
            for k in GRID_K:
                cfg_id = _config_id(chunk_size, chunk_overlap, k)
                df_fail = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "config_id": cfg_id,
                    "tipo_golden": tipo_golden,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "k": k,
                    "id_golden": "__PREP_FAIL__",
                    "tipo_item": "__PREP_FAIL__",
                    "portaria": "__PREP_FAIL__",
                    "tecnologia": "__PREP_FAIL__",
                    "pergunta": "__PREP_FAIL__",
                    "ground_truth": "",
                    "resposta_gerada": f"__EXCEPTION__: {type(e).__name__}: {e}",
                    "contexts": [],
                    "rougeL": 0.0,
                    "erro": 1,
                    "tempo": None,
                    "tokens": None,
                    "custo": None,
                    "bertscore_p": None,
                    "bertscore_r": None,
                    "bertscore_f1": None,
                }])
                _append_csv(df_fail, OUTPUT_CSV)
            continue

        # 2) avalia para cada k sem reindexar
        for k in GRID_K:
            cfg_id = _config_id(chunk_size, chunk_overlap, k)
            if cfg_id in done:
                print(f"⏭️ Pulando {cfg_id} (já no CSV).")
                continue

            # muda apenas o k do retriever (não recria índice)
            try:
                retriever.search_kwargs = dict(getattr(retriever, "search_kwargs", {}) or {})
                retriever.search_kwargs["k"] = int(k)
            except Exception:
                retriever.search_kwargs = {"k": int(k)}

            print(f"\n▶️ Configuração: {cfg_id} (reuso do índice: {_index_id(chunk_size, chunk_overlap)})")

            try:
                df_res = avaliar_tecnologias(
                    retriever=retriever,
                    golden_set=golden_set,
                    openai_api_key=OPENAI_API_KEY,
                    config_id=cfg_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k,
                    tipo_golden=tipo_golden,
                    compute_bertscore=COMPUTE_BERTSCORE,
                )
                _append_csv(df_res, OUTPUT_CSV)
                done.add(cfg_id)
                print(f"✅ Salvo progresso ({len(df_res)} linhas) em {OUTPUT_CSV}")
            except Exception as e:
                print(f"❌ Falha ao avaliar {cfg_id}: {type(e).__name__}: {e}")
                df_fail = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "config_id": cfg_id,
                    "tipo_golden": tipo_golden,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "k": k,
                    "id_golden": "__BENCH_FAIL__",
                    "tipo_item": "__BENCH_FAIL__",
                    "portaria": "__BENCH_FAIL__",
                    "tecnologia": "__BENCH_FAIL__",
                    "pergunta": "__BENCH_FAIL__",
                    "ground_truth": "",
                    "resposta_gerada": f"__EXCEPTION__: {type(e).__name__}: {e}",
                    "contexts": [],
                    "rougeL": 0.0,
                    "erro": 1,
                    "tempo": None,
                    "tokens": None,
                    "custo": None,
                    "bertscore_p": None,
                    "bertscore_r": None,
                    "bertscore_f1": None,
                }])
                _append_csv(df_fail, OUTPUT_CSV)

            time.sleep(0.2)

    print("\n✅ Grid concluído.")
    print("📄 CSV final:", OUTPUT_CSV)

if __name__ == "__main__":
    executar_grid_search(tipo_golden="normal")
