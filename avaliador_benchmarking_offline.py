import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# Funções de resposta (devem existir no seu projeto)
from llm import responder_com_llm
from rag import responder_com_rag
from advanced_rag import responder_com_advancedrag

# Métricas
from evaluate import load as eval_load

# =========================
# Helpers
# =========================
def _portaria_to_filter(portaria_str: str):
    # aceita "2104/2022" ou "Portaria nº 2104/2022"
    if not portaria_str:
        return None
    m = re.search(r"(\d{1,5})\s*/\s*(\d{4})", portaria_str)
    if not m:
        return None
    return {"numero": int(m.group(1)), "ano": int(m.group(2))}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def _resolver_ground_truth(item: dict) -> str:
    return (
        item.get("ground_truth")
        or item.get("resposta")
        or item.get("resposta_esperada")
        or item.get("resposta_correta")
        or ""
    )

def _extract_contexts(fontes: Any) -> List[str]:
    """
    Tenta transformar 'fontes' em lista de strings (contexts).
    Aceita:
      - lista de Document (langchain) -> usa page_content
      - lista de dicts -> tenta 'page_content'/'content'/'texto'
      - string -> vira [string]
      - None -> []
    """
    if fontes is None:
        return []
    if isinstance(fontes, str):
        return [fontes]

    out: List[str] = []
    try:
        if isinstance(fontes, list):
            for f in fontes:
                if f is None:
                    continue
                # langchain Document
                if hasattr(f, "page_content"):
                    out.append(_safe_str(getattr(f, "page_content", "")))
                elif isinstance(f, dict):
                    out.append(_safe_str(f.get("page_content") or f.get("content") or f.get("texto") or ""))
                else:
                    out.append(_safe_str(f))
        else:
            out.append(_safe_str(fontes))
    except Exception:
        return []

    return [c for c in out if c.strip()]

def _ensure_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Garante que todas as colunas existam, mesmo quando métricas forem desativadas."""
    defaults = {
        "bertscore_p": None,
        "bertscore_r": None,
        "bertscore_f1": None,
    }
    for k, v in defaults.items():
        row.setdefault(k, v)
    return row

# =========================
# Função principal
# =========================

def avaliar_tecnologias(
    retriever,
    golden_set: list[dict],
    openai_api_key: str,
    config_id: str,
    chunk_size: int,
    chunk_overlap: int,
    k: int,
    tipo_golden: str,
    compute_bertscore: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Roda as tecnologias e devolve DataFrame (não salva).
    Métricas: ROUGE-L (por item) + BERTScore (batch opcional).
    """

    # Neutraliza proxies (evita bug proxies/httpx em alguns ambientes)
    for kk in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy","OPENAI_PROXY","openai_proxy"]:
        os.environ.pop(kk, None)

    # ROUGE-L
    rouge = eval_load("rouge")

    # BERTScore
    bert = None
    if compute_bertscore:
        try:
            bert = eval_load("bertscore")
        except Exception:
            bert = None

    tecnologias = {
        "LLM": responder_com_llm,
        "RAG_Tradicional": responder_com_rag,
        "Advanced_RAG": responder_com_advancedrag,
    }

    rows: List[Dict[str, Any]] = []

    # Guardar itens para cálculo em batch do BERTScore
    bert_inputs = []   # (row_idx, pred, ref)

    for item in tqdm(golden_set, desc=f"Avaliando ({tipo_golden})"):
        pergunta = _safe_str(item.get("pergunta"))
        gt = _safe_str(_resolver_ground_truth(item))

        meta_id = item.get("id")
        meta_tipo = item.get("tipo_item") or item.get("tipo")  # mantém compatibilidade
        meta_portaria = item.get("portaria")

        for nome, fn in tecnologias.items():
            t0 = datetime.now()
            erro = 0
            resposta = ""
            fontes = []
            tokens = None
            custo = None

            try:
                if nome == "LLM":
                    resposta, fontes, tempo, tokens, custo = fn(pergunta, openai_api_key)
                else:
                    out = fn(pergunta, retriever, openai_api_key)
                    resposta = out[0]
                    fontes = out[1] if len(out) > 1 else []
                    tempo = out[2] if len(out) > 2 else None
                    tokens = out[3] if len(out) > 3 else None
                    custo = out[4] if len(out) > 4 else None

            except Exception as e:
                erro = 1
                tempo = (datetime.now() - t0).total_seconds()
                resposta = f"__EXCEPTION__: {type(e).__name__}: {e}"
                fontes = []

            # ROUGE-L por item (quando erro=1 fica 0)
            rougeL = 0.0
            if (not erro) and gt.strip() and resposta.strip():
                try:
                    score = rouge.compute(predictions=[resposta], references=[gt])
                    rougeL = float(score.get("rougeL", 0.0))
                except Exception:
                    rougeL = 0.0

            row = _ensure_columns({
                "timestamp": _now_iso(),
                "config_id": config_id,
                "tipo_golden": tipo_golden,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "k": int(k),
                "id_golden": meta_id,
                "tipo_item": meta_tipo,
                "portaria": meta_portaria,
                "tecnologia": nome,
                "pergunta": pergunta,
                "ground_truth": gt,
                "resposta_gerada": resposta,
                "contexts": _extract_contexts(fontes),
                "rougeL": rougeL,
                "erro": int(erro),
                "tempo": tempo if tempo is not None else (datetime.now() - t0).total_seconds(),
                "tokens": tokens,
                "custo": custo,
            })
            rows.append(row)

            row_idx = len(rows) - 1
            if (not erro) and compute_bertscore and (bert is not None) and gt.strip() and resposta.strip():
                bert_inputs.append((row_idx, resposta, gt))

    # =========================
    # BERTScore (batch)
    # =========================
    if compute_bertscore and (bert is not None) and bert_inputs:
        try:
            preds = [x[1] for x in bert_inputs]
            refs  = [x[2] for x in bert_inputs]
            b = bert.compute(predictions=preds, references=refs, lang="pt")
            P = b.get("precision", [])
            R = b.get("recall", [])
            F = b.get("f1", [])
            for i, (row_idx, _, _) in enumerate(bert_inputs):
                rows[row_idx]["bertscore_p"]  = float(P[i]) if i < len(P) else None
                rows[row_idx]["bertscore_r"]  = float(R[i]) if i < len(R) else None
                rows[row_idx]["bertscore_f1"] = float(F[i]) if i < len(F) else None
        except Exception as e:
            print(f"⚠️ Falha ao calcular BERTScore: {type(e).__name__}: {e}")

    return pd.DataFrame(rows)


def carregar_golden_set(diretorio_dados: str, tipo_golden: str):
    fname = "golden_set_normal.json" if tipo_golden == "normal" else "golden_set_malicioso_full.json"
    path = os.path.join(diretorio_dados, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"Golden set vazio: {path}")
    return data, path
