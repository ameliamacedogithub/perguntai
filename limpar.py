import os
import re
import json
import pandas as pd
from tqdm import tqdm


# ===============================
# LIMPEZA
# ===============================
CABECALHO_PADRAO = [
    "MINISTÉRIO DA EDUCAÇÃO",
    "SECRETARIA DE EDUCAÇÃO PROFISSIONAL E TECNOLÓGICA",
    "INSTITUTO FEDERAL DE EDUCAÇÃO, CIÊNCIA E TECNOLOGIA DE GOIÁS",
    "REITORIA",
]

def limpar_texto(texto: str) -> str:
    if not texto:
        return ""

    linhas = texto.split("\n")
    linhas_limpas = []
    for l in linhas:
        s = l.strip()
        if not s:
            continue
        if s in CABECALHO_PADRAO:
            continue
        linhas_limpas.append(s)

    texto_limpo = " ".join(linhas_limpas)
    texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()
    return texto_limpo

# ===============================
# GERAÇÃO
# ===============================
def gerar_portarias_limpas(portarias_json_path: str, destino_dir: str):
    os.makedirs(destino_dir, exist_ok=True)

    with open(portarias_json_path, encoding="utf-8") as f:
        dados = json.load(f)

    if not dados:
        raise ValueError("portarias.json está vazio.")

    limpas = []

    for item in tqdm(dados, desc="Limpando textos"):
        texto_limpo = limpar_texto(item.get("conteudo", ""))

        limpas.append({
            # mantém metadados confiáveis do portarias.json
            "numero": item.get("numero"),
            "ano": item.get("ano"),
            "data": item.get("data"),
            "orgao": item.get("orgao"),
            "ementa": item.get("ementa"),
            "identificador": item.get("identificador"),
            # ✅ manter auditoria/linha do site
            "texto_item": item.get("texto_item"),
            # ✅ manter as duas urls (raw e download)
            "url_pdf_raw": item.get("url_pdf"),
            "url_pdf": item.get("url_pdf_download") or item.get("url_pdf"),
            "arquivo": item.get("arquivo"),
            # texto limpo
            "conteudo_limpo": texto_limpo,
        })

    # salvar JSON
    caminho_json = os.path.join(destino_dir, "portarias_limpas.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(limpas, f, ensure_ascii=False, indent=2)
    
    
    caminho_csv = os.path.join(destino_dir, "portarias_metadados_limpos.csv")

    df = pd.DataFrame([{
      "identificador": p.get("identificador"),
      "numero": p.get("numero"),
      "ano": p.get("ano"),
      "data": p.get("data"),
      "orgao": p.get("orgao"),

      # ✅ entra para auditoria humana
      "texto_item": p.get("texto_item"),

      "arquivo": p.get("arquivo"),
      "url_pdf": p.get("url_pdf"),
    } for p in limpas])



    df.to_csv(caminho_csv, index=False, encoding="utf-8")

    print("✅ JSON limpo:", caminho_json)
    print("✅ CSV metadados:", caminho_csv)
    print("✅ Total portarias:", len(limpas))
    return caminho_json


