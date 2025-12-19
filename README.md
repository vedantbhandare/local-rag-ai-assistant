
# Local RAG AI Assistant (Zero Cost, Fully Offline)

A **fully local AI assistant** that reads your documents (PDF, TXT, CSV) and answers questions using **Retrieval-Augmented Generation (RAG)**.

No paid APIs  
No cloud services  
Runs entirely on a local laptop  

---

## Features

- Local LLM using **Ollama (Mistral 7B – quantized)**
- Vector database using **FAISS**
- Retrieval-Augmented Generation (RAG)
- Hallucination prevention (answers only from documents)
- Interactive **Streamlit UI**
- Shows retrieved document context for transparency

---

## Tech Stack

- **Python**
- **Ollama**
- **Mistral 7B (quantized)**
- **LangChain (modern modular version)**
- **FAISS**
- **Streamlit**

---

## Project Structure

```

local-rag-ai/
│
├── app.py              # Streamlit UI
├── ingest.py           # Document ingestion & embedding
├── rag_chain.py        # Terminal-based RAG (optional)
├── documents/          # Input documents
├── vector_store/       # FAISS index (generated)
├── requirements.txt
└── .gitignore

```

---

## Setup Instructions

### 1️ Install Ollama
 https://ollama.com  
Then pull the model:

```bash
ollama pull mistral:7b-instruct-q4_K_M
````

---

### 2️ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️ Add Documents

Place your files inside:

```
documents/
```

(Supported: PDF, TXT, CSV)

---

### 5️ Ingest Documents

```bash
python ingest.py
```

---

### 6 Run the App

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## Example Questions

* What is this document about?
* Summarize the main points
* What topics are discussed?
* Ask an out-of-scope question → returns **"I don't know"**

---

## Why This Project Matters

This project demonstrates:

* Real-world **RAG architecture**
* Local-first AI deployment
* Vector search & embeddings
* Prompt grounding & hallucination control
* Production-style debugging of fast-moving AI libraries

---

## Future Improvements

* File upload via UI
* Chat history
* Source citations
* Model switcher
* Multi-document comparison

---
