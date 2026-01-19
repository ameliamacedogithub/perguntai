import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CAMINHO DO ARQUIVO

# 1. Pega o caminho atual onde você está
caminho_atual = os.getcwd()

# 2. Define as partes do caminho
pasta_intermediaria = "dados_documentos"
nome_arquivo = "benchmark_consolidado.csv"

CSV_PATH = os.path.join(caminho_atual, pasta_intermediaria, nome_arquivo)


# Tenta ler o CSV
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado em: {CSV_PATH}")
    # Cria um DF fake apenas para o código não quebrar se você rodar sem arquivo
    df = pd.DataFrame({
        "tecnologia": ["chroma", "faiss"] * 5,
        "tipo_item": ["lei", "manual"] * 5,
        "rougeL": np.random.rand(10),
        "bertscore_f1": np.random.rand(10),
        "erro": [0]*10,
        "config_id": range(10),
        "chunk_size": [1000]*10,
        "chunk_overlap": [100]*10,
        "k": [4]*10
    })

# -----------------------
# Limpeza / coerção
# -----------------------
def to_num(col, default=np.nan):
    return pd.to_numeric(df.get(col, default), errors="coerce")

df["rougeL"] = to_num("rougeL", 0).fillna(0.0)
df["erro"] = to_num("erro", 0).fillna(0).astype(int)

# métricas opcionais
df["bertscore_f1"] = to_num("bertscore_f1", np.nan)

# Remove linhas "internas" ou sujeira
df_valid = df[~df["tecnologia"].astype(str).str.startswith("__", na=False)].copy()

# Garante que existe 'tipo_item'
if "tipo_item" not in df_valid.columns:
    if "tipo" in df_valid.columns:
        df_valid["tipo_item"] = df_valid["tipo"]
    else:
        # Se não tiver, cria uma genérica para não travar
        df_valid["tipo_item"] = "geral"

# -----------------------
# Helpers de plot
# -----------------------
def has_non_nan(series):
    return series.notna().any()

def bar_plot(labels, values, title, ylabel, rotation=30, figsize=(8, 5)):
    plt.figure(figsize=figsize)
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def heatmap(table, title, cbar_label, xlab="", ylab="", figsize=None):
    if table.empty:
        return
    plt.figure(figsize=figsize)
    plt.imshow(table.values, aspect="auto", cmap='viridis')
    plt.yticks(range(len(table.index)), table.index)
    plt.xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Filtra apenas o que deu certo (erro == 0)
df_ok = df_valid[df_valid["erro"] == 0].copy()

# ==============================================================================
# AGREGAÇÕES (Correção: Adicionamos 'n_ok' aqui para uso interno, mas não exibiremos)
# ==============================================================================

# A) Por abordagem
agg_ok_tec = (
    df_ok.groupby("tecnologia")
    .agg(
        rougeL_mean=("rougeL", "mean"),
        bert_f1_mean=("bertscore_f1", "mean"),
        n_ok=("erro", "count") # Necessário para saber a confiabilidade
    )
    .reset_index()
)

# B) Por tipo_item × abordagem
agg_ok_tipo_tec = (
    df_ok.groupby(["tipo_item", "tecnologia"])
    .agg(
        rougeL_mean=("rougeL", "mean"),
        bert_f1_mean=("bertscore_f1", "mean"),
        n_ok=("erro", "count")
    )
    .reset_index()
)

# C) Por Configuração Completa (Fundamental para o top_configs)
agg_ok_cfg = (
    df_ok.groupby(["config_id", "chunk_size", "chunk_overlap", "k", "tecnologia"])
    .agg(
        rougeL_mean=("rougeL", "mean"),
        bert_f1_mean=("bertscore_f1", "mean"),
        n_ok=("erro", "count") # <--- AQUI ESTAVA O ERRO (Faltava essa contagem)
    )
    .reset_index()
)

#

# Abordagens esperadas na ordem desejada
TEC_ORDER = ["LLM", "RAG_Tradicional", "Advanced RAG"]

def pivot_wide_tipo_tecnologia(value_col):
    """
    Cria tabela wide:
      index = tipo_item
      columns = tecnologia (LLM, RAG_Tradicional, Advanced RAG)
      values = value_col (rougeL_mean ou bert_f1_mean)
    """
    # Filtra apenas tecnologias de interesse (se existir)
    sub = agg_ok_tipo_tec.copy()
    sub = sub[sub["tecnologia"].isin(TEC_ORDER)].copy()

    wide = sub.pivot_table(
        index="tipo_item",
        columns="tecnologia",
        values=value_col,
        aggfunc="mean"
    )

    # Garante colunas na ordem (mesmo que falte alguma)
    for t in TEC_ORDER:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[TEC_ORDER]

    # Ordena por média (opcional) para ficar mais legível
    wide = wide.loc[wide.mean(axis=1, skipna=True).sort_values(ascending=False).index]

    return wide

def style_bold_rowmax(df_wide, decimals=6):
    """
    Destaca em negrito o maior valor de cada linha (ignorando NaN).
    Para uso em notebook (display com Styler).
    """
    def _bold_row_max(row):
        if row.dropna().empty:
            return [""] * len(row)
        m = row.max(skipna=True)
        return ["font-weight: bold" if pd.notna(v) and v == m else "" for v in row]

    return (
        df_wide.style
        .apply(_bold_row_max, axis=1)
        .format(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "--")
    )
def montar_label_config(df):
    # Cria um label estável e legível (e ordenável)
    return (
        "cs" + df["chunk_size"].astype(int).astype(str)
        + "_co" + df["chunk_overlap"].astype(int).astype(str)
        + "_k" + df["k"].astype(int).astype(str)
    )

def pivot_config_wide(df_cfg, metric_col):
    """
    df_cfg deve ter: tecnologia, chunk_size, chunk_overlap, k, metric_col
    Retorna: index=config_label, cols=tecnologia, values=metric
    """
    sub = df_cfg.copy()

    

    sub["config_label"] = montar_label_config(sub)

    wide = sub.pivot_table(
        index=["chunk_size", "chunk_overlap", "k", "config_label"],
        columns="tecnologia",
        values=metric_col,
        aggfunc="mean"
    )

    # garante colunas na ordem e coloca -- quando não existir
    for t in TEC_ORDER:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[TEC_ORDER]

    # ordena pelas combinações (chunk_size, overlap, k)
    wide = wide.sort_index(level=[0, 1, 2], ascending=[True, True, True])

    return wide.reset_index().set_index("config_label")[TEC_ORDER]

def style_bold_rowmax(df_wide, decimals=6):
    def _bold_row_max(row):
        s = row.dropna()
        if s.empty:
            return [""] * len(row)
        m = s.max()
        return ["font-weight: bold" if pd.notna(v) and v == m else "" for v in row]

    return (
        df_wide.style
        .apply(_bold_row_max, axis=1)
        .format(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "--")
    )



# -----------------------
# 1) Tabelas principais
# -----------------------
print("\n=== A) Por abordagem (Ordenado por ROUGE-L) ===")
# Exibe sem mostrar a coluna n_ok
display(agg_ok_tec[["tecnologia", "rougeL_mean", "bert_f1_mean"]].sort_values("rougeL_mean", ascending=False))


# -----------------------
# 2) Gráficos por tecnologia
# -----------------------
if len(agg_ok_tec) > 0:
    sub = agg_ok_tec.sort_values("rougeL_mean", ascending=False)
    bar_plot(sub["tecnologia"], sub["rougeL_mean"].fillna(0),
             "ROUGE-L médio por tecnologia",
             "ROUGE-L (média)", rotation=30)

    if has_non_nan(df_ok["bertscore_f1"]):
        sub = agg_ok_tec.sort_values("bert_f1_mean", ascending=False)
        bar_plot(sub["tecnologia"], sub["bert_f1_mean"].fillna(0),
                 "BERTScore F1 médio por tecnologia",
                 "BERTScore F1 (média)", rotation=30)

# -----------------------
# 3) Melhor abordagem por tipo_item
# -----------------------
def melhor_por_tipo_ok(metric_col):
    """Retorna a melhor abordagem para cada tipo de documento."""
    # Remove NaNs da métrica
    sub = agg_ok_tipo_tec.dropna(subset=[metric_col]).copy()
    if sub.empty:
        return None

    # Pega o maior valor de cada grupo
    best = (
        sub.sort_values(["tipo_item", metric_col], ascending=[True, True])
        .groupby("tipo_item")
        .tail(1)
        .sort_values(metric_col, ascending=False)
    )
    # Seleciona APENAS as colunas desejadas (SEM n_ok)
    return best[["tipo_item", "tecnologia", metric_col]]

print("\n=== Tipo de pergunta x Abordagem — ROUGE-L ===")
wide_tipo_rouge = pivot_wide_tipo_tecnologia("rougeL_mean")
display(style_bold_rowmax(wide_tipo_rouge, decimals=6))

if has_non_nan(df_ok["bertscore_f1"]):
    print("\n=== Tipo de pergunta x Abordagem — BERTScore F1 ===")
    wide_tipo_bert = pivot_wide_tipo_tecnologia("bert_f1_mean")
    display(style_bold_rowmax(wide_tipo_bert, decimals=6))



# -----------------------
# 4) Heatmaps
# -----------------------
def heatmap_tipo_tecnologia_ok(value_col, title, cbar_label):
    if value_col not in agg_ok_tipo_tec.columns or not agg_ok_tipo_tec[value_col].notna().any():
        return

    table = agg_ok_tipo_tec.pivot_table(
        index="tipo_item",
        columns="tecnologia",
        values=value_col,
        aggfunc="mean"
    )

    # Tratamento caso a tabela venha vazia
    if table.empty:
        print(f"Heatmap vazio para {value_col}")
        return

    # Ordenação
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    table = table[table.mean(axis=0).sort_values(ascending=False).index]

    heatmap(
        table,
        title=title,
        cbar_label=cbar_label,
        xlab="Tecnologia",
        ylab="Tipo Item",
        figsize=(10, max(4, 0.5 * len(table.index)))
    )

heatmap_tipo_tecnologia_ok("rougeL_mean", "Heatmap: Tipo x Tec (ROUGE-L)", "ROUGE-L")
if has_non_nan(df_ok["bertscore_f1"]):
    heatmap_tipo_tecnologia_ok("bert_f1_mean", "Heatmap: Tipo x Tec (BERTScore)", "BERTScore")


# =========================
# 6) Melhores CONFIGS (Top N)
# =========================

def top_configs_por_metrica(metric_col, topn=10, min_n=5):
    """
    Lista as melhores configurações.
    Usa 'n_ok' internamente para filtrar configs com poucos testes,
    mas não exibe 'n_ok' no final.
    """
    sub = agg_ok_cfg.copy()

    if metric_col not in sub.columns:
        return None

    sub = sub.dropna(subset=[metric_col])

    # Filtra ruído (configs que rodaram poucas vezes)
    # Se n_ok não existir (caso raro), ignora filtro
    if "n_ok" in sub.columns:
        sub = sub[sub["n_ok"] >= min_n]

    if sub.empty:
        return None

    res = sub.sort_values(metric_col, ascending=False).head(topn)

    # Retorna APENAS colunas relevantes (sem n_ok)
    cols_to_show = ["config_id", "tecnologia", "chunk_size", "chunk_overlap", "k", metric_col]
    return res[cols_to_show]

# === TABELA wide por CONFIG (ROUGE-L)
cfg_rouge_wide = pivot_config_wide(agg_ok_cfg, "rougeL_mean")
print("\n=== Configurações x abordagem — ROUGE-L ===")
display(style_bold_rowmax(cfg_rouge_wide, decimals=6))

# === TABELA wide por CONFIG (BERTScore)
if has_non_nan(df_ok["bertscore_f1"]):
    cfg_bert_wide = pivot_config_wide(agg_ok_cfg, "bert_f1_mean")
    print("\n=== Configurações x Abordagem — BERTScore F1 ===")
    display(style_bold_rowmax(cfg_bert_wide, decimals=6))


# =========================
# 7) Efeito isolado de parâmetros
# =========================
def efeito_parametro_ok(param_col, metric_col, metric_label):
    # Verifica se colunas existem
    if metric_col not in agg_ok_cfg.columns or param_col not in agg_ok_cfg.columns:
        return

    # Aqui precisamos do n_ok para fazer uma média ponderada ou soma simples
    # Se n_ok existir no agg_ok_cfg, usamos ele.
    if "n_ok" in agg_ok_cfg.columns:
        sub = agg_ok_cfg.groupby(param_col).agg(
            m=(metric_col, "mean"),
            # n=("n_ok", "sum") # Mantemos o cálculo mas não precisamos exibir
        ).reset_index()
    else:
        # Fallback se n_ok não existir
        sub = agg_ok_cfg.groupby(param_col).agg(
            m=(metric_col, "mean")
        ).reset_index()

    sub = sub.dropna(subset=["m"]).sort_values(param_col)

    if sub.empty:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(sub[param_col].astype(str), sub["m"], marker="o")
    plt.xlabel(param_col)
    plt.ylabel(f"{metric_label} (média)")
    plt.title(f"Impacto de {param_col} no {metric_label}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Exibe tabela simples sem n_ok
    display(sub[[param_col, "m"]].rename(columns={"m": f"Média {metric_label}"}))

print("\n📈 Análise de Sensibilidade de Parâmetros")
efeito_parametro_ok("k", "rougeL_mean", "ROUGE-L")
efeito_parametro_ok("chunk_size", "rougeL_mean", "ROUGE-L")
efeito_parametro_ok("chunk_overlap", "rougeL_mean", "ROUGE-L")

efeito_parametro_ok("k", "bert_f1_mean", "BERTScore")
efeito_parametro_ok("chunk_size", "bert_f1_mean", "BERTScore")
efeito_parametro_ok("chunk_overlap", "bert_f1_mean", "BERTScore")
