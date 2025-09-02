# SiteMainer

Un potente strumento di analisi web che consente di scansionare una lista di URL e valutare la rilevanza dei contenuti attraverso ricerca di parole chiave e analisi semantica basata su AI.

## 🚀 Caratteristiche

- **Analisi multi-URL**: Scansiona automaticamente una lista di siti web
- **Doppio scoring system**:
  - **Keyword Score**: Combina diversità e frequenza delle parole chiave trovate
  - **Semantic Score**: Utilizza embeddings AI per valutare la correlazione semantica
- **Scansione flessibile**: Opzione per analizzare solo la homepage o esplorare il sito in profondità
- **Export Excel**: Esporta i risultati in formato .xlsx per ulteriori analisi
- **Interface intuitiva**: Web app costruita con Streamlit

## 📋 Prerequisiti

- Python 3.8+
- Dipendenze elencate in `requirements.txt`

## 🛠️ Installazione

1. Clona il repository:
```bash
git clone https://github.com/[username]/sitemainer.git
cd sitemainer
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

3. Avvia l'applicazione:
```bash
streamlit run main.py
```

## 📁 Struttura del progetto

```
sitemainer/
│
├── main.py              # Script principale dell'applicazione
├── utils.py             # Funzioni di utilità per web scraping e analisi
├── requirements.txt     # Dipendenze Python
├── data/
│   └── Template.xlsx    # Template di esempio per l'input
└── img/
    ├── logo.png         # Logo dell'applicazione
    └── favicon.png      # Favicon
```

## 🎯 Come utilizzare

### 1. Preparazione del file di input

Crea un file Excel (.xlsx) con due fogli:

- **websites**: Contiene la colonna `url` con la lista dei siti da analizzare
- **keywords**: Contiene la colonna `keywords` con le parole chiave da ricercare

Puoi scaricare il template di esempio direttamente dall'applicazione.

### 2. Configurazione dei parametri

L'applicazione offre diverse opzioni di configurazione:

#### Parametri di scansione:
- **Deep scan**: Abilita la scansione di più pagine per sito
- **Numero massimo pagine**: Limita le pagine da scansionare per sito (1-50)
- **Profondità massima**: Profondità nell'alberatura del sito (1-5)
- **Delay**: Pausa tra le richieste per evitare rate limiting (0-5 secondi)

#### Pesi del keyword score:
- **Diversità**: Peso per la varietà delle keyword trovate
- **Frequenza**: Peso per la frequenza delle keyword

#### Pesi del final score:
- **Similarità semantica**: Peso dell'analisi AI semantica
- **Presenza keyword**: Peso della ricerca esatta delle keyword

### 3. Esecuzione dell'analisi

1. Carica il file Excel preparato
2. Configura i parametri desiderati
3. Clicca su "▶️ Avvia l'analisi"
4. Attendi il completamento (visualizzazione del progresso in tempo reale)
5. Scarica i risultati in formato Excel

## 📊 Output e metriche

L'analisi produce diverse metriche per ogni URL:

- **keyword_score** (0-1): Combina diversità e frequenza delle keyword
- **semantic_score** (0-1): Misura la correlazione semantica tramite AI
- **final_score** (0-1): Punteggio finale pesato
- **text**: Contenuto testuale estratto
- **pages_scanned**: Numero di pagine analizzate
- **keyword_matches**: Lista delle keyword trovate

## ⚡ Prestazioni e limitazioni

- **Testato con**: Fino a 5.000 URL (solo homepage)
- **Raccomandazioni**: 
  - Per liste grandi, suddividere in più file
  - Utilizzare delay appropriati per evitare rate limiting
  - Monitorare la connessione di rete durante l'elaborazione

## 🧠 Tecnologie utilizzate

- **Streamlit**: Framework per l'interfaccia web
- **Sentence Transformers**: Modello AI per embeddings semantici
- **BeautifulSoup**: Web scraping e parsing HTML
- **Pandas**: Manipolazione dati

## 📝 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

## 📞 Supporto

Per domande, bug report o richieste di funzionalità, apri un issue su GitHub.

---

**Note**: Questo strumento è progettato per analisi di contenuti pubblicamente accessibili. Assicurati di rispettare i termini di servizio dei siti web analizzati e le policy sui robots.txt.