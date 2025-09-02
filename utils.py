import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"  # o "watchdog"
import re, math, requests, os, io, time
from urllib.parse import urljoin, urlparse
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



###################################################################################################
# FUNZIONI
def fetch_text(url: str, timeout=10) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code >= 400:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join([x.get_text(" ", strip=True) for x in soup.find_all(["title","h1","h2","h3","p","li"])])
        return re.sub(r"\s+", " ", text)
    except Exception:
        return ""

def extract_links(url: str, timeout=10) -> list:
    """Estrae tutti i link interni di un sito web"""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
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
                # Rimuove fragment (#) e parametri query se presenti
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.add(clean_url)
        
        return list(links)
    except Exception:
        return []

def scan_website(url: str, max_pages=1, max_depth=1, sleep_time=1.0, timeout=10) -> tuple:
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
    current_level = [url]  # URL del livello corrente
    all_texts = []
    depth = 0
    
    while current_level and len(visited) < max_pages and depth < max_depth:
        next_level = []  # URL del prossimo livello
        
        for current_url in current_level:
            if current_url in visited or len(visited) >= max_pages:
                continue
                
            # Fetch del testo
            text = fetch_text(current_url, timeout)
            if text:
                all_texts.append(text)
                visited.add(current_url)
                
                # Estrai link per il prossimo livello (solo se non siamo all'ultimo livello)
                if depth < max_depth - 1 and len(visited) < max_pages:
                    new_links = extract_links(current_url, timeout)
                    for link in new_links:
                        if link not in visited and link not in next_level:
                            next_level.append(link)
            
            # Sleep tra le pagine dello stesso sito
            if len(visited) < max_pages and (current_level[current_level.index(current_url)+1:] or next_level):
                time.sleep(sleep_time)
        
        # Passa al livello successivo
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
   
   # Score ibrido
   # Componente diversità (keyword uniche trovate)
   unique_found = sum(1 for c in counts.values() if c > 0)
   diversity_score = unique_found / len(keywords)
   
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
        patt = r"\b" + re.escape(k) + r"\b" # match esatto
        counts[k] = len(re.findall(patt, t))
    return counts    

def semantic_score(text: str, topic_phrases, model) -> float:
    if not text.strip():
        return 0.0
    vecs = model.encode([text] + topic_phrases, normalize_embeddings=True, show_progress_bar=False)
    sims = vecs[1:] @ vecs[0]
    return float(np.max(sims))

def classify_url(url: str, keywords, model, alpha_sem=0.7, alpha_kw=0.3, alpha_div=0.6, alpha_fre=0.4, max_pages=1, max_depth=1, sleep_time=1.0):
    # Scansiona il sito web
    text, pages_scanned = scan_website(url.strip(), max_pages, max_depth, sleep_time)
    
    # conteggi e score keyword
    kmatch = keyword_matches(text, keywords)
    kscore = keyword_score(text, keywords, peso_diversity=alpha_div, peso_frequenza=alpha_fre)

    # embedding score
    sscore = semantic_score(text, list(keywords), model)

    # fusione
    final = alpha_sem * sscore + alpha_kw * kscore

    # colonne per keyword: flag_
    per_kw_flags = {
        f"flag_{k.lower().replace(' ', '_')}": 1 if kmatch[k.lower()] > 0 else 0
        for k in (kw.lower() for kw in keywords)
    }

    base = {
        "url": url,
        "text": text,
        "pages_scanned": pages_scanned,
        "semantic_score": round(sscore, 3),
        "keyword_score": round(kscore, 3),
        "final_score": round(final, 3),
    }
    # aggiunge i flag per keyword
    base.update(per_kw_flags)

    return base

def prepare_excel_download(df_results):
    """Prepara il file Excel per il download"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="risultati")
    buf.seek(0)
    return buf