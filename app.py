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

from utils import fetch_text, extract_links, scan_website, keyword_score, keyword_matches, semantic_score, classify_url, prepare_excel_download


def main():
    st.set_page_config(page_title="SiteMainer", page_icon="img/favicon.png", layout="centered")
    st.image("img/logo.png")

    st.markdown("""
        Il tool consente di scansionare un elenco di <strong>URL di siti internet</strong> ed effettuare un'analisi dei contenuti per ricerca di <strong>parole chiave</strong>, valutando la <strong>similarità semantica</strong>
        rispetto ad una lista di keyword fornite.<br>""", unsafe_allow_html=True)

    with st.expander("ℹ️ Come funziona", expanded=False):
        st.markdown("""               
            Per il corretto funzionamento del tool, l'utente deve caricare un <i>file excel</i> (.xlsx) che rispetti la struttura del template (scaricabile in basso).
            In particolare il file in input deve essere composto da due fogli:
            <ul>
                <li><strong>websites</strong>: elenco di URL da analizzare</li>
                <li><strong>keywords</strong>: elenco di parole chiave da ricercare</li>
            </ul>
            Il tool esegue il download del contenuto testuale di ogni URL e calcola due score:
            <ul>
                <li><strong>keyword score</strong>: combina diversità e frequenza delle keyword trovate. Premia siti che contengono molte keyword diverse piuttosto che ripetizioni della stessa parola. Range: 0-1</li>
                <li><strong>semantic score</strong>: utilizza AI embeddings per misurare quanto il contenuto del sito sia semanticamente correlato alle keyword, anche senza match esatti. Range: 0-1</li>
            </ul>
            I due score sono combinati in un <strong>final score</strong> pesato che rappresenta il risultato finale.<br>
            L'utente può regolare il peso dei due score e i parametri di scansione (solo homepage o più pagine del sito web) tramite il pannello <i>"⚙️ Opzioni"</i>.<br><br>

            I risultati sono esportabili in formato Excel (xlsx).<br>
                    
    """
                , unsafe_allow_html=True)

        st.warning(
        "**Nota operativa**\n\n"
        "- Il sistema è stato testato con file contenenti fino a **5.000 siti**, limitando la scansione alla sola homepage di ciascun dominio.\n"
        "- Per liste di grandi dimensioni è consigliabile suddividere gli URL in più file, così da ridurre i tempi di elaborazione e minimizzare il rischio di dover ripetere l’analisi in caso di interruzioni.\n"
        "- Le prestazioni dipendono da fattori esterni (rete, CPU, risposta dei server). Alcuni provider possono applicare **rate-limit** o blocchi temporanei, causando rallentamenti o stop dell’elaborazione.\n"
                    , icon="🚨"
    )
    
    # Inizializza session state
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None

    with st.sidebar:
        st.header("⚙️ Opzioni")

        with st.expander("🌐 Parametri di scansione", expanded=False):
            st.caption("Regola il comportamento della scansione dei siti web (numero di pagine, profondità, delay tra richieste)")
            deep_scan = st.checkbox("Scansiona l'intero sito web (non solo homepage)", value=False)
            max_pages = st.slider("Numero massimo di pagine per sito", 1, 50, 5, 1, disabled=not deep_scan)
            max_depth = st.slider("Profondità massima nell'alberatura", 1, 5, 2, 1, disabled=not deep_scan)
            sleep_time = st.slider("Delay tra pagine (secondi)", 0.0, 5.0, 1.0, 0.25, disabled=not deep_scan)

        with st.expander("🧮 Pesi del keyword score", expanded=False):
            st.caption("Regola i pesi di diversità e frequenza per la definizione dello score delle keyword")
            alpha_div = st.slider("Peso diversità", 0.0, 1.0, 0.6, 0.05)
            alpha_fre = st.slider("Peso frequenza", 0.0, 1.0, 1.0 - alpha_div, 0.05, disabled=True)
        
        with st.expander("⚖️ Pesi del final score", expanded=False):
            st.caption("Regola il peso relativo tra similarità semantica e presenza keyword per la definizione dello score finale")
            alpha_sem = st.slider("Peso similarità semantica", 0.0, 1.0, 0.5, 0.05)
            alpha_kw = st.slider("Peso presenza keyword", 0.0, 1.0, 1.0 - alpha_sem, 0.05, disabled=True)

        if not deep_scan:
            max_pages = 1
            max_depth = 1

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
        keywords = {k.lower() for k in df_keywords['keywords'].to_list()}
        st.success(f"✅ File '{uploaded_file.name}' caricato con successo!")


        if st.button("▶️ Avvia l'analisi"):
            st.info("⏳ Analisi in corso...")
            st.info("🤖 Configurazione modello embeddings...")

            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

            n = len(df_url)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            # Pre-crea alcune colonne utili
            for col in ["text", "pages_scanned", "keyword_score", "semantic_score", "final_score"]:
                if col not in df_url.columns:
                    df_url[col] = np.nan

            for pos, (idx, row) in enumerate(df_url.iterrows(), start=1):
                url = row["url"]
                res = classify_url(url, keywords, model, alpha_sem=alpha_sem, alpha_kw=alpha_kw,
                                   alpha_div=alpha_div, alpha_fre=alpha_fre,
                                   max_pages=max_pages, max_depth=max_depth, sleep_time=sleep_time)

                # aggiorna df
                for k, v in res.items():
                    if k not in df_url.columns:
                        df_url[k] = np.nan
                    df_url.at[idx, k] = v

                # progress
                progress_bar.progress(pos / n)
                pages_info = f"({res.get('pages_scanned', 1)} pagine)" if max_pages > 1 else ""
                status_text.text(f"Analizzato {pos}/{n} URL → {url} {pages_info}")

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