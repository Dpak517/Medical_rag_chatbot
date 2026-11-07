# 🩺 Medical RAG Chatbot

> A Retrieval-Augmented Generation (RAG) medical assistant built with LangChain, Pinecone, Sentence-Transformers and Groq, answering health questions grounded in *Where There Is No Doctor*.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Groq](https://img.shields.io/badge/LLM-Groq-purple)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

This chatbot answers health questions using only the content of a medical encyclopedia, instead of relying on an LLM's memory. The 61.4 MB corpus is split into **68,342 chunks**, embedded into **384-dimensional vectors** with `all-MiniLM-L6-v2`, and stored in Pinecone for fast semantic search. The most relevant chunks are passed to an LLM through Groq, which is instructed to answer strictly from that context and to decline when the answer isn't there.

## ✨ Core Capabilities

- **Grounded answers**: responses are generated only from retrieved book passages, which reduces hallucination.
- **Semantic search**: cosine-similarity retrieval over 68k chunks handles everyday wording such as "loose motions" and "yellow eyes".
- **Safe refusals**: off-topic questions (coding, finance, sports) are declined instead of answered.
- **Source transparency**: the Streamlit UI shows the retrieved passages behind each answer.
- **Medical safety**: includes a disclaimer and emergency guidance for danger signs.

## 🏗️ Architecture

```
User question (Streamlit)
        │
        ▼
 Embedding model  (all-MiniLM-L6-v2, 384-dim)
        │
        ▼
 Pinecone vector search  (top-K chunks, cosine similarity)
        │
        ▼
 Prompt  (strict "answer only from context" rules)
        │
        ▼
 Groq LLM  (openai/gpt-oss-20b)
        │
        ▼
 Answer + sources
```

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangChain |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` |
| Vector database | Pinecone |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Evaluation | Custom test set + RAGAS |
| Frontend | Streamlit |

## 📊 Evaluation

The system was tested on a self-written **30-question test set** with reference answers taken from the book:

| Category | Questions | Purpose |
|---|---|---|
| Normal questions | 20 | Direct factual questions |
| Casual / symptom-style | 5 | Everyday wording instead of book terms |
| Out-of-scope | 5 | The bot should decline |

**Results**

| Metric | Baseline | After prompt tuning |
|---|---|---|
| RAGAS Faithfulness (5-question sample) | 0.53 | **0.66** |
| Out-of-scope questions correctly declined | 5 / 5 | 5 / 5 |

Faithfulness measures whether every claim in the answer is supported by the retrieved context. It improved after the prompt was rewritten to forbid facts outside the context and to include all relevant points from it.

**Limitations:** the test set is small and was written and graded by the author, and the RAGAS run covers only 5 questions, so results are indicative rather than conclusive. Retrieval is weakest on casual wording.

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd medical_chatbot

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys to a .env file
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key

# 5. Build the vector index (first time only)
python vectors.py

# 6. Run the app
streamlit run main.py
```

## 📁 Project Structure

```
medical_chatbot/
├── main.py          # Streamlit app and RAG chain
├── vectors.py       # Chunking, embedding and Pinecone indexing
├── eval.py          # Runs the test set through the pipeline
├── ragas_eval.py    # RAGAS faithfulness evaluation
├── requirements.txt
└── .env             # API keys (not committed)
```

## 🔮 Future Work

- Hybrid search (BM25 + embeddings) for better recall on casual wording
- Cross-encoder reranking of retrieved chunks
- Larger test set and a stronger embedding model

## ⚠️ Disclaimer

This project is for educational purposes. It provides general health information from a reference book and is **not a substitute for professional medical advice, diagnosis or treatment**. In an emergency, contact a doctor or local emergency services immediately.

## 📄 License

MIT
