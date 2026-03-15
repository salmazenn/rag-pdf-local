# RAG PDF Local

Posez des questions sur vos PDFs — 100% local avec Ollama, LangChain et ChromaDB.

## Stack
- Ollama (llama3)
- LangChain
- ChromaDB
- Streamlit

## 🚀 Lancer l'application

### 1. Prérequis
- Python 3.11+
- [Ollama](https://ollama.com) installé

### 2. Installation
\```bash
git clone https://github.com/salmazenn/rag-pdf-local.git
cd rag-pdf-local
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3
\```

### 3. Ingérer un PDF
\```bash
python src/rag.py ingest ~/Downloads/mon-fichier.pdf
\```

### 4. Lancer l'interface web
\```bash
ollama serve &
streamlit run src/app.py
\```
Ouvre **http://localhost:8501** dans ton navigateur.

### 5. Ou utiliser le terminal
\```bash
python src/rag.py chat
\```
