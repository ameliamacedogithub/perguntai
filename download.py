# agente_download.py
# Versão com AJUSTES + DEBUGS (para você ver exatamente onde está falhando)
import os
import re
import json
import fitz  # PyMuPDF
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote, unquote
from unidecode import unidecode


# =========================
# CONFIG DE DEBUG
# =========================
DEBUG = False                 # liga/desliga prints de debug
DEBUG_SAVE_HTML = False       # salva debug_portarias.html no destino_dir
DEBUG_SAMPLE_ANCHORS = 10    # quantos <a> com "Portaria" mostrar no debug


def _log(msg: str):
    if DEBUG:
        print(msg)


def baixar_html(url: str, timeout: int = 60) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extrair_links_portarias(html: str, st_container=None):
    """
    Retorna lista de tuplas: (texto_item, url_pdf_original)
    - texto_item: texto visível do item (âncora + ementa se existir)
    - url_pdf_original: href absoluto (pode conter espaços ou %20 etc.)
    """
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=True)
    _log(f"DEBUG: Total <a> encontrados: {len(links)}")

    # Regex mais estrito para não capturar "Portarias (de conteúdo normativo)"
    # e para garantir que o texto do <a> é mesmo de uma portaria com PDF.
    padrao = re.compile(
        r"^Portaria\s+n[º°]\s*\d{1,4}(?:/\d{4})?.*\(PDF.*KB\)",
        re.IGNORECASE,
    )

    # Debug: amostra de anchors que contém "Portaria"
    if DEBUG:
        shown = 0
        for a in links:
            t = a.get_text(" ", strip=True)
            if "Portaria" in t:
                _log("---- DEBUG ANCHOR ----")
                _log(f"A_TEXT: {t}")
                _log(f"HREF: {a.get('href')}")
                # pai pode conter ementa completa
                _log(f"PARENT_TEXT: {a.parent.get_text(' ', strip=True)[:240]}")
                sib = a.next_sibling
                _log(f"NEXT_SIBLING: {str(sib).strip()[:240] if sib else ''}")
                shown += 1
                if shown >= DEBUG_SAMPLE_ANCHORS:
                    break

    portarias = []

    for link in links:
        texto_a = _normalize_text(link.get_text(" ", strip=True))
        if not texto_a:
            continue

        href = link["href"]

        # ✅ só PDFs (evita capturar links como "Portarias (de conteúdo normativo)")
        if ".pdf" not in href.lower():
            continue

        # ✅ precisa casar com o padrão de portaria no TEXTO do <a>
        if not padrao.search(texto_a):
            continue

        # href absoluto
        if not href.startswith("http"):
            href = "https://www.ifg.edu.br" + href

        # 2. APLICA A CORREÇÃO (Encode) - Transforma espaço em %20
        # Isso garante que o link funcione ao ser clicado ou baixado
        href = quote(href, safe=":/%?=&")
        
        # ✅ Captura ementa fora do <a> (next_sibling)
        ementa = ""
        sibling = link.next_sibling
        if sibling:
            ementa = _normalize_text(str(sibling))
            if ementa.startswith("-"):
                ementa = _normalize_text(ementa[1:])

        texto_item = texto_a
        if ementa and ementa not in texto_item:
            texto_item = f"{texto_item} - {ementa}"

        msg = f"📄 {texto_item} -> {href}"
        if st_container:
            st_container.markdown(msg)
        else:
            if DEBUG:
                print(msg)

        portarias.append((texto_item, href))

    _log(f"DEBUG: Total portarias extraídas: {len(portarias)}")
    if DEBUG and portarias:
        _log(f"DEBUG: Exemplo portaria[0]: {portarias[0][0]}")
        _log(f"DEBUG: Exemplo url[0]: {portarias[0][1]}")

    if st_container:
        st_container.success(f"✅ {len(portarias)} links de portarias extraídos.")
    else:
        print(f"✅ {len(portarias)} links de portarias extraídos.")
    return portarias


def extrair_texto_pdf(caminho_pdf: str) -> str:
    texto = ""
    with fitz.open(caminho_pdf) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto.strip()


def extrair_metadados(texto_item: str):
    """
    Extrai metadados a partir do texto do item (link + ementa).
    Ex:
      "Portaria nº 2147 - REITORIA/IFG, de 2 de dezembro de 2025 (PDF 49 KB) - Homologa ..."
    """
    texto_item = _normalize_text(texto_item)

    m_num = re.search(r"Portaria\s+n[º°]\s*(\d{1,4})(?:/(\d{4}))?", texto_item, re.IGNORECASE)
    if not m_num:
        raise ValueError(f"❌ Falha ao extrair número/ano de: {texto_item}")

    numero = m_num.group(1)
    ano_inline = m_num.group(2)
    ano = int(ano_inline) if ano_inline else None

    m_data = re.search(r"de\s+(\d{1,2}º?\s+de\s+[a-zç]+\s+de\s+\d{4})", texto_item, re.IGNORECASE)
    data = m_data.group(1) if m_data else None
    if data and not ano:
        m_ano = re.search(r"(\d{4})$", data)
        if m_ano:
            ano = int(m_ano.group(1))

    m_org = re.search(
        r"Portaria\s+n[º°]\s*(?:\d{1,4}(?:/\d{4})?)\s*-\s*([^,]+),",
        texto_item,
        re.IGNORECASE
    )
    orgao = m_org.group(1).strip() if m_org else None

    ementa = None
    m_em = re.search(r"\(PDF.*?\)\s*-\s*(.+)$", texto_item, re.IGNORECASE)
    if m_em:
        ementa = m_em.group(1).strip()
    else:
        if " - " in texto_item:
            ementa = texto_item.rsplit(" - ", 1)[-1].strip()

    identificador = f"Portaria nº {numero}/{ano}" if ano else f"Portaria nº {numero}"

    return {
        "numero": numero,
        "ano": ano,
        "data": data,
        "orgao": orgao,
        "ementa": ementa,
        "identificador": identificador,
        "texto_item": texto_item,  # útil para auditoria/debug
    }


def baixar_e_processar_portarias(
    destino_dir: str,
    url_html: str,
    st_container=None,
    timeout_html: int = 60,
    timeout_pdf: int = 120,
):
    destino_dir_abs = os.path.abspath(destino_dir)
    os.makedirs(destino_dir_abs, exist_ok=True)

    if st_container:
        st_container.info(f"📂 Portarias serão salvas em: `{destino_dir_abs}`")
    else:
        print(f"📂 Portarias serão salvas em: {destino_dir_abs}")

    # =========================
    # DEBUG: baixar e salvar HTML
    # =========================
    _log("DEBUG: chamando baixar_html...")
    html = baixar_html(url_html, timeout=timeout_html)
    _log(f"DEBUG: baixar_html OK | len(html)={len(html)}")
    _log(f"DEBUG: html[:200]={html[:200]!r}")

    if DEBUG_SAVE_HTML:
        debug_path = os.path.join(destino_dir_abs, "debug_portarias.html")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(html)
        _log(f"DEBUG: Salvei debug_portarias.html em: {debug_path}")
        _log(f"DEBUG: arquivo existe? {os.path.exists(debug_path)} | size={os.path.getsize(debug_path)}")

    # extrair links
    portarias = extrair_links_portarias(html, st_container)

    dados_extraidos = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for texto_item, url_pdf_original in tqdm(portarias, desc="⬇️ Baixando portarias"):
        # ✅ Para baixar: preservar % (evita re-encodar %20 para %2520) e manter possíveis params
        url_pdf_download = quote(url_pdf_original, safe=":/%?=&")

        # ✅ Nome do arquivo: DECODE primeiro (%20 -> espaço), depois troca espaço por "_"
        nome_arquivo_original = url_pdf_original.split("/")[-1]
        nome_arquivo_decodificado = unquote(nome_arquivo_original)
        nome_arquivo = unidecode(nome_arquivo_decodificado).replace(" ", "_")
        caminho_pdf = os.path.join(destino_dir_abs, nome_arquivo)

        try:
            if DEBUG:
                _log(f"DEBUG: baixando -> {url_pdf_download}")

            resp = requests.get(url_pdf_download, headers=headers, timeout=timeout_pdf)
            resp.raise_for_status()

            with open(caminho_pdf, "wb") as f:
                f.write(resp.content)

            texto_pdf = extrair_texto_pdf(caminho_pdf)

            metadados = extrair_metadados(texto_item)
            metadados["conteudo"] = texto_pdf
            metadados["url_pdf"] = url_pdf_original
            metadados["url_pdf_download"] = url_pdf_download
            metadados["arquivo"] = nome_arquivo

            dados_extraidos.append(metadados)

        except Exception as e:
            erro_msg = f"⚠️ Erro ao processar: {texto_item} | URL: {url_pdf_original} | Erro: {e}"
            if st_container:
                st_container.error(erro_msg)
            else:
                print(erro_msg)

    # Salva JSON completo
    caminho_json = os.path.join(destino_dir_abs, "portarias.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(dados_extraidos, f, ensure_ascii=False, indent=2)

    if st_container:
        st_container.success(f"📁 JSON salvo em: `{caminho_json}`")
    else:
        print(f"📁 JSON salvo em: {caminho_json}")

    # Salva CSV de metadados (sem conteúdo pesado)
    caminho_csv = os.path.join(destino_dir_abs, "portarias_metadados.csv")
    try:
        df = pd.DataFrame([{k: v for k, v in p.items() if k != "conteudo"} for p in dados_extraidos])
        df.to_csv(caminho_csv, index=False, encoding="utf-8")
        if st_container:
            st_container.success(f"📄 CSV salvo em: `{caminho_csv}`")
        else:
            print(f"📄 CSV salvo em: {caminho_csv}")
    except Exception as e:
        if st_container:
            st_container.error(f"⚠️ Erro ao salvar CSV: {e}")
        else:
            print(f"⚠️ Erro ao salvar CSV: {e}")

    return caminho_json



