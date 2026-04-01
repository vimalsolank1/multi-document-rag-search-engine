# 🤖 Multi-Document RAG Search Engine

A hybrid RAG (Retrieval-Augmented Generation) chatbot that answers questions
from your documents and live web search - built with LangChain, FAISS, and Streamlit.

---

## 👤 Author

- **Name:** Vimal solanki
- **Email:** vimal162002@email.com


---

## 🎥 Demo & Explanation


- 🖥️ **Live Demo:** [multi-document-rag-ai-chatbot.streamlit.app](https://multi-document-rag-ai-chatbot.streamlit.app/)

- 📹 **Video Explanation:** [Watch here](https://drive.google.com/file/d/1E_5fXD9SDdZIvhdY_1XVXMDZfePXj4j9/view?usp=sharing)

<img width="1915" height="1038" alt="Image" src="https://github.com/user-attachments/assets/10ffa4e0-3f83-4cd6-9067-844243b135a8" />

---


## 📌 Project Overview

Organizations usually store knowledge across multiple unstructured documents like PDFs, reports, and notes.  
However, these documents are static and do not contain real-time information.

This creates a gap where users need both internal knowledge and up-to-date data.

To solve this problem, I built a hybrid RAG chatbot that combines document retrieval with live web search.

This system can:
- 📄 Search across multiple documents at the same time  
- 🌐 Fetch real-time information from the web  
- 🔀 Combine both sources intelligently  
- 📌 Clearly distinguish between document-based and real-time answers  

This project solves that by combining:
- 📄 Multi-document semantic search (your private files)
- 🌐 Live web search via Tavily (real-time facts)
- 🔀 Hybrid mode (both combined)
---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python |
| LLM | LLaMA 3.3 70B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (FREE, local) |
| Vector DB | FAISS |
| Web Search | Tavily |
| Orchestration | LangChain |
| UI | Streamlit |

---

## 🏗️ Project Structure
```
multi-document-rag-search-engine/
│
├── config/
│   └── settings.py          # loads all .env variables
│
├── core/
│   ├── ingestion.py         # loads and chunks PDF/TXT files
│   ├── embedding.py         # HuggingFace embedding model
│   ├── vector_store.py      # FAISS index management
│   └── chain.py             # RAG pipeline (retrieve → context → answer)
│
├── tools/
│   └── tavily_search.py     # Tavily web search integration
│
├── ui/
│   ├── chat.py              # main controller connecting UI and backend
│   └── components.py        # all Streamlit UI components
│
├── data/documents/          # sample documents for testing
├── main.py                  # app entry point
├── .env                     # API keys and config (not committed)
└── requirements.txt
```

---

## 🔄 How It Works — Architecture


<img width="1410" height="3276" alt="Image" src="https://github.com/user-attachments/assets/94c34a2b-897a-41ec-b001-84eb6dae128a" />

### 🧠 Component Roles
```
main.py  ──────────────────▶  Streamlit UI (entry point)
    │
    ▼
chat.py  ──────────────────▶  Controller (connects UI ↔ backend)
    │
    ├──▶  ingestion.py    ──▶  Load & chunk documents
    │
    ├──▶  embedding.py    ──▶  Convert text to vectors
    │
    ├──▶  vector_store.py ──▶  Store & search in FAISS
    │
    ├──▶  chain.py        ──▶  RAG pipeline (retrieve → LLM → answer)
    │
    └──▶  tavily_search.py──▶  Live web search
```

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/multi-document-rag-search-engine.git
cd multi-document-rag-search-engine
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
# Add your API keys in .env
```
```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K_RESULTS=6

GPT_MODEL_NAME=llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_api_key
TEMPERATURE=0

TAVILY_API_KEY=your_tavily_api_key
TOP_K_WEB_RESULTS=3
```

**4. Run the app**
```bash
streamlit run main.py
```

---

## 💡 How to Use

1. Upload one or more PDF or TXT files
2. Click **Process Documents**
3. Select retrieval mode — Documents, Web, or Hybrid
4. Ask your question and get answers with citations

---

## 📊 Example Queries

| Query | Best Mode |
|-------|-----------|
| "Explain attention mechanism" | 📄 Documents |
| "Latest news about GPT-5" | 🌐 Web |
| "How does RAG compare to current LLM tools?" | 🔀 Hybrid |

---

## 🔑 API Keys Required

| Service | Free Tier | Link |
|---------|-----------|------|
| Groq | ✅ Yes | [console.groq.com](https://console.groq.com) |
| Tavily | ✅ Yes | [tavily.com](https://tavily.com) |
| HuggingFace Embeddings | ✅ 100% Free | Runs locally |

---

## 🎯 Key Learnings

- Built a hybrid RAG system with document + web retrieval
- Used FAISS for fast semantic vector search
- Integrated Tavily for real-time web results
- Implemented citation-aware answer generation
- Deployed a full AI app with Streamlit

---

## 📄 License

This project is for educational and portfolio purposes.
