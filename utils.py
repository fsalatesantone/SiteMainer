import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import re, math, requests, os, io, time
from urllib.parse import urljoin, urlparse
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Configurazione session di requests con retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_requests_session():
    """Crea una session con retry automatici e timeout configurati"""
    session = requests.Session()
    
    # Configurazione retry: max 3 tentativi con backoff
    retry_strategy = Retry(
        total=0,
        backoff_factor=0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Session globale da riutilizzare
_session = None

def get_session():
    global _session
    if _session is None:
        _session = get_requests_session()
    return _session


###################################################################################################
# FUNZIONI
def fetch_text(url: str, timeout=8) -> str:
    """
    Scarica il testo da un URL con gestione robusta degli errori.
    timeout può essere una tupla (connect_timeout, read_timeout)
    """
    try:
        session = get_session()
        
        # Timeout separato per connessione e lettura
        # (connect_timeout, read_timeout)
        timeout_tuple = (3, 8)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }

        r = session.get(
            url, 
            timeout=timeout_tuple,
            headers=headers,
            allow_redirects=True
        )

        # DEBUG: stampa status e lunghezza HTML
        print(f"📊 {url} → Status: {r.status_code}, HTML length: {len(r.text)}, Content-Type: {r.headers.get('content-type', 'N/A')}")
        
        if r.status_code >= 400:
            print(f"⚠️ Status {r.status_code} per {url}")
            return ""
            
        soup = BeautifulSoup(r.text, "html.parser")
        
        # DEBUG: stampa primi 500 caratteri dell'HTML
        if len(r.text) < 1000:
            print(f"⚠️ HTML molto corto per {url}: {r.text[:500]}")

        # Rimuove script, style, noscript
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
            
        # Estrae testo da tag significativi
        text = " ".join([
            x.get_text(" ", strip=True) 
            for x in soup.find_all(["title", "h1", "h2", "h3", "p", "li"])
        ])

        # DEBUG: lunghezza testo estratto
        print(f"✂️ Testo estratto da {url}: {len(text)} caratteri")
        
        return re.sub(r"\s+", " ", text)
        
    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout per {url}")
        return ""
    except requests.exceptions.ConnectionError:
        print(f"🔌 Errore di connessione per {url}")
        return ""
    except requests.exceptions.TooManyRedirects:
        print(f"🔄 Troppi redirect per {url}")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore generico per {url}: {type(e).__name__}")
        return ""
    except Exception as e:
        print(f"💥 Errore inaspettato per {url}: {type(e).__name__}")
        return ""


def extract_links(url: str, timeout=15) -> list:
    """Estrae tutti i link interni di un sito web"""
    try:
        session = get_session()
        timeout_tuple = (5, timeout)
        
        r = session.get(
            url, 
            timeout=timeout_tuple,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            allow_redirects=True
        )
        
        if r.status_code >= 400:
            return []
        
        soup = BeautifulSoup(r.text, "html.parser")
        base_domain = urlparse(url).netloc
        links = set()
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            
            # Solo link interni dello stesso dominio
            if parsed.netloc == base_domain and parsed.scheme in ["http", "https"]:
                # Rimuove fragment (#) e parametri query
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.add(clean_url)
        
        return list(links)
        
    except (requests.exceptions.Timeout, 
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException):
        print(f"⚠️ Errore estrazione link da {url}")
        return []
    except Exception:
        return []


def scan_website(url: str, max_pages=1, max_depth=1, sleep_time=1.0, timeout=15) -> tuple:
    """
    Scansiona un sito web fino a max_pages pagine e max_depth profondità.
    Ritorna (testo_combinato, numero_pagine_scansionate)
    """
    if max_pages == 1:
        # Solo homepage
        text = fetch_text(url, timeout)
        return text, 1 if text else 0
    
    # Scansione multipla con gestione profondità
    visited = set()
    current_level = [url]
    all_texts = []
    depth = 0
    
    while current_level and len(visited) < max_pages and depth < max_depth:
        next_level = []
        
        for current_url in current_level:
            if current_url in visited or len(visited) >= max_pages:
                continue
                
            # Fetch del testo
            text = fetch_text(current_url, timeout)
            if text:
                all_texts.append(text)
                visited.add(current_url)
                
                # Estrai link per il prossimo livello
                if depth < max_depth - 1 and len(visited) < max_pages:
                    new_links = extract_links(current_url, timeout)
                    for link in new_links:
                        if link not in visited and link not in next_level:
                            next_level.append(link)
            
            # Sleep tra le pagine
            if len(visited) < max_pages and (current_level[current_level.index(current_url)+1:] or next_level):
                time.sleep(sleep_time)
        
        current_level = next_level
        depth += 1
    
    # Combina tutti i testi
    combined_text = " ".join(all_texts)
    return combined_text, len(visited)


def keyword_score(text: str, keywords: set, peso_diversity, peso_frequenza) -> float:
    if not text:
        return 0.0
    
    t = text.lower()
    
    # Conta le occorrenze per ogni keyword
    counts = {}
    for k in keywords:
        k_lower = k.lower()
        pattern = r"\b" + re.escape(k_lower) + r"\b"
        counts[k_lower] = len(re.findall(pattern, t))
    
    # Componente diversità (keyword uniche trovate)
    unique_found = sum(1 for c in counts.values() if c > 0)
    diversity_score = unique_found / len(keywords) if keywords else 0
    
    # Componente frequenza (con cap per evitare spam)
    capped_hits = sum(min(c, 3) for c in counts.values())
    frequency_score = 1.0 - math.exp(-capped_hits/8.0)
    
    # Combina diversità + frequenza
    return peso_diversity * diversity_score + peso_frequenza * frequency_score


def keyword_matches(text: str, keywords) -> dict:
    """
    Ritorna un dict {keyword_norm: count_match} sul testo.
    Usa \b ... \b per il confine di parola (match esatto).
    """
    if not text:
        return {k.lower(): 0 for k in keywords}
    t = text.lower()
    counts = {}
    for k in (kw.lower() for kw in keywords):
        patt = r"\b" + re.escape(k) + r"\b"
        counts[k] = len(re.findall(patt, t))
    return counts


def semantic_score(text: str, topic_phrases, model) -> float:
    if not text.strip():
        return 0.0
    try:
        vecs = model.encode([text] + topic_phrases, normalize_embeddings=True, show_progress_bar=False)
        sims = vecs[1:] @ vecs[0]
        return float(np.max(sims))
    except Exception as e:
        print(f"⚠️ Errore nel calcolo semantic score: {type(e).__name__}")
        return 0.0


def classify_url(url: str, keywords, model, alpha_sem=0.7, alpha_kw=0.3, 
                alpha_div=0.6, alpha_fre=0.4, max_pages=1, max_depth=1, sleep_time=1.0):
    """
    Classifica un URL analizzando il contenuto testuale.
    Gestisce robustamente timeout e errori di connessione.
    """
    # Scansiona il sito web
    text, pages_scanned = scan_website(url.strip(), max_pages, max_depth, sleep_time)
    
    # Se non c'è testo, ritorna risultati vuoti
    if not text:
        per_kw_flags = {
            f"flag_{k.lower().replace(' ', '_')}": 0
            for k in (kw.lower() for kw in keywords)
        }
        return {
            "url": url,
            "text": "",
            "pages_scanned": 0,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "final_score": 0.0,
            "error": "Impossibile scaricare contenuto",
            **per_kw_flags
        }
    
    # Conteggi e score keyword
    kmatch = keyword_matches(text, keywords)
    kscore = keyword_score(text, keywords, peso_diversity=alpha_div, peso_frequenza=alpha_fre)

    # Embedding score
    sscore = semantic_score(text, list(keywords), model)

    # Fusione
    final = alpha_sem * sscore + alpha_kw * kscore

    # Colonne per keyword: flag_
    per_kw_flags = {
        f"flag_{k.lower().replace(' ', '_')}": 1 if kmatch[k.lower()] > 0 else 0
        for k in (kw.lower() for kw in keywords)
    }

    base = {
        "url": url,
        "text": text[:500] + "..." if len(text) > 500 else text,  # Limita lunghezza testo salvato
        "pages_scanned": pages_scanned,
        "semantic_score": round(sscore, 3),
        "keyword_score": round(kscore, 3),
        "final_score": round(final, 3),
    }
    
    # Aggiunge i flag per keyword
    base.update(per_kw_flags)

    return base


def prepare_excel_download(df_results):
    """Prepara il file Excel per il download"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="risultati")
    buf.seek(0)
    return buf