import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import re, math, requests, io
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

def keyword_score(text: str, keywords: set) -> float:
    if not text:
        return 0.0
    t = text.lower()
    hits = sum(len(re.findall(r"\b" + re.escape(k) + r"\b", t)) for k in keywords)
    return 1.0 - math.exp(-hits/5.0)


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

def keyword_score_from_counts(counts: dict) -> float:
    hits = sum(counts.values())
    return 1.0 - math.exp(-hits/5.0)
    

def semantic_score(text: str, topic_phrases, model) -> float:
    if not text.strip():
        return 0.0
    vecs = model.encode([text] + topic_phrases, normalize_embeddings=True, show_progress_bar=False)
    sims = vecs[1:] @ vecs[0]
    return float(np.max(sims))

def classify_url(url: str, keywords, model, alpha_sem=0.7, alpha_kw=0.3):
    text = fetch_text(url)
    # conteggi e score keyword
    kmatch = keyword_matches(text, keywords)
    kscore = keyword_score_from_counts(kmatch)

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


def main():
    st.set_page_config(page_title="SiteMainer", page_icon="img/favicon.png", layout="centered")
    st.image("img/logo.png")

    st.markdown("""
                    Scansione di un elenco di URL di siti internet e analisi dei contenuti per ricerca di <strong>parole chiave</strong> e valutazione della <strong>similarità semantica</strong>
                    rispetto a una lista di keyword fornite.<br>
                    
                L'utente deve caricare un <i>file excel</i> (.xlsx) con due fogli:<br>
                    - <strong>websites</strong>: elenco di URL da analizzare<br>
                    - <strong>keywords</strong>: elenco di parole chiave da ricercare<br>
                Per il corretto funzionamento del tool, è necessario rispettare la struttura e il formato del file excel di temaplate (scaricabile in basso).<br>

                Viene effettuato il download del contenuto testuale di ogni URL e calcolati due score:<br>
                    - <strong>keyword_score</strong>: misura la presenza delle parole chiave nel testo in un intervallo tra 0 e 1<br>
                    - <strong>semantic_score</strong>: misura la similarità semantica tra il testo e le parole chiave in un intervallo tra 0 e 1<br>
                I due score sono combinati in un <strong>final_score</strong> che rappresenta il risultato finale.<br>
                    L'utente può regolare il peso dei due score tramite slider riportati nel pannello <i>"Opzioni"</i>.<br>
                
                I risultati sono esportabili in formato Excel (xlsx).<br>
                        
"""
                    , unsafe_allow_html=True)

    # Inizializza session state
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None

    with st.expander("📂 Carica il file", expanded=True):
        uploaded_file = st.file_uploader("Carica il file con elenco dei siti web e lista di keyword da analizzare", type=["xlsx"])
        st.caption("Il file deve seguire la struttura del file 'Template.xlsx' (scaricabile in basso)")
        with open(os.path.join("data", "Template.xlsx"), "rb") as f:
            template_bytes = f.read()
        st.download_button(
            label="📥 Scarica il template di esempio",
            data=template_bytes,
            file_name="Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if uploaded_file:
        df_url = pd.read_excel(uploaded_file, sheet_name='websites')
        df_keywords = pd.read_excel(uploaded_file, sheet_name='keywords')
        keywords = {k.lower().strip() for k in df_keywords['keywords'].to_list()}
        st.success(f"✅ File '{uploaded_file.name}' caricato con successo!")

        with st.expander("⚙️ Opzioni", expanded=False):
            alpha_sem = st.slider("Peso similarità semantica", 0.0, 1.0, 0.7, 0.05)
            alpha_kw = st.slider("Peso presenza keyword", 0.0, 1.0, 1.0 - alpha_sem, 0.05, disabled=True)

        if st.button("▶️ Avvia l'analisi"):
            st.info("⏳ Analisi in corso...")
            st.info("🤖 Configurazione modello embeddings...")

            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

            n = len(df_url)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            # Pre-crea alcune colonne utili
            for col in ["text", "keyword_score", "semantic_score", "final_score"]:
                if col not in df_url.columns:
                    df_url[col] = np.nan

            for pos, (idx, row) in enumerate(df_url.iterrows(), start=1):
                url = row["url"].strip()
                res = classify_url(url, keywords, model, alpha_sem=alpha_sem, alpha_kw=alpha_kw)

                # aggiorna df
                for k, v in res.items():
                    if k not in df_url.columns:
                        df_url[k] = np.nan
                    df_url.at[idx, k] = v

                # progress
                progress_bar.progress(pos / n)
                status_text.text(f"Analizzato {pos}/{n} URL → {url}")

            # Salva i risultati nel session state
            st.session_state.df_results = df_url.copy()
            st.session_state.analysis_completed = True
            
            st.success("Analisi completata ✅")

    # Mostra i risultati se l'analisi è stata completata
    if st.session_state.analysis_completed and st.session_state.df_results is not None:
        st.subheader("📊 Risultati dell'analisi")
        st.dataframe(st.session_state.df_results)

        # Prepara il file Excel per il download
        excel_buffer = prepare_excel_download(st.session_state.df_results)

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.download_button(
                label="📥 Download dei risultati (XLSX)",
                data=excel_buffer,
                file_name="risultati.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_results"
            )
        
        with col2:
            if st.button("🔄 Nuova analisi", key="reset_analysis"):
                st.session_state.analysis_completed = False
                st.session_state.df_results = None
                st.rerun()

if __name__ == "__main__":
    main()