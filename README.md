# ML-RAG-101: News RAG Pipeline

A minimal **Retrieval-Augmented Generation (RAG)** pipeline that ingests news (RSS), indexes it in a vector store, and answers questions using an LLM grounded in retrieved articles.

---

## 1. RAG Pipeline Theory

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is a pattern that lets large language models (LLMs) use **external knowledge** at answer time instead of relying only on what they learned during training. The idea: *retrieve* relevant documents, *augment* the prompt with that context, then *generate* the answer. Hence: **R**etrieval **A**ugmented **G**eneration.

### Why It Exists

LLMs have a fixed “knowledge cutoff” and can’t see the latest data, your private docs, or domain-specific sources. Retraining or fine-tuning is expensive and slow. RAG avoids that by treating the model as a *reasoner* and feeding it the right *evidence* for each question.

### Stages of a RAG Pipeline

A typical RAG pipeline has these stages:

| Stage | What happens |
|-------|-------------------------------|
| **1. Ingestion** | Raw data (documents, web pages, feeds) is loaded, cleaned, and split into chunks. |
| **2. Embedding** | Each chunk is turned into a vector (embedding) using a model (e.g. sentence-transformers). |
| **3. Indexing** | Vectors are stored in a vector database (e.g. ChromaDB) so you can search by *semantic similarity*, not just keywords. |
| **4. Query processing** | The user question is (optionally) rewritten or expanded to improve retrieval (e.g. by an LLM). |
| **5. Retrieval** | The query is embedded and the vector DB returns the most similar chunks (and their metadata). |
| **6. Augmentation** | Retrieved chunks are packed into a “context” block (sometimes reranked/summarized). |
| **7. Generation** | The LLM gets a prompt that includes this context and the user question, and generates an answer. |

So the pipeline is: **Ingest → Embed → Index → (Optional: Refine query) → Retrieve → Augment prompt → Generate.**

### Core Idea

The model doesn’t “remember” the news or docs; the system **fetches** the right pieces and **puts them in the prompt**. The LLM then *deduces* the answer from that context, which keeps responses grounded and up to date.

---

## 2. How RAG Helps LLMs Deduce Information

### The Problem RAG Solves

LLMs are good at pattern matching and language, but they can **hallucinate**: invent facts, dates, or sources that sound plausible but aren’t true. They also can’t see information that appeared after their training or that lives only in your data. RAG addresses this by **grounding** the model in real, retrieved text.

### How Deduction Works in RAG

Think of it in three steps:

1. **Evidence**  
   The retriever finds passages that are *semantically* related to the question (e.g. “EU AI regulations”) even if the exact words differ. That gives the LLM **evidence** instead of asking it to rely on memory.

2. **Context in the prompt**  
   That evidence is pasted into the prompt (often with clear labels like “Context: …”). The model is instructed to answer **using** this context. So the “premises” for the model’s reasoning are explicit and controllable.

3. **Generation from context**  
   The LLM’s job becomes: *given this context and this question, produce an answer*. The model **deduces** the answer from the provided text—summarizing, comparing, or extracting—rather than from its internal weights alone.

So RAG doesn’t change how the LLM *thinks*; it changes **what information it is allowed to use** for each answer. That’s why we say the model “deduces” from the retrieved material: the conclusion is supposed to follow from the context you gave it.

### Benefits in Practice

- **Fewer hallucinations** — Answers are tied to concrete passages; the model has less room to invent.
- **Up-to-date answers** — New documents (e.g. today’s news) are indexed and retrieved; no retraining.
- **Traceability** — You can show which documents were retrieved and, if needed, cite them.
- **Domain and private data** — Internal docs, tickets, or knowledge bases can be the only “knowledge” the model sees for that query.

### Limits to Keep in Mind

RAG improves grounding but doesn’t remove all errors. The model can still misread the context, over-rely on one passage, or be sensitive to the order of retrieved chunks. Good retrieval (and optionally reranking) is essential so the “premises” you feed the LLM are actually the right ones.

---

## 3. How to Run This Project

### Prerequisites

- **Python 3.10+**
- An **LLM API key** (for cloud models) or a local **Ollama** setup (for local models)

### 3.1 Install Dependencies

Install everything in one go from the project root:

```bash
cd /path/to/ML-Rag-101
python3 -m venv .venv
source .venv/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you already use a virtual environment, activate it and run:

```bash
pip install -r requirements.txt
```

**One-liner (no venv):**

```bash
pip install -r requirements.txt
```

### 3.2 Configure the LLM (API key or Ollama)

The pipeline uses **LiteLLM**, which supports OpenAI, Ollama, and others.

**Option A — OpenAI (or other cloud API):**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B — Local model with Ollama:**

Install [Ollama](https://ollama.ai), pull a model (e.g. `ollama pull llama3`), then either set:

```bash
export OLLAMA_HOST="http://localhost:11434"
```

or pass the model explicitly in code (e.g. `model="ollama/llama3"`). LiteLLM will use Ollama by default when you use an `ollama/...` model name.

### 3.3 Run the Server (default)

The app runs as a **long-lived server** until you stop it (e.g. Ctrl+C):

```bash
python main.py
```

Server listens on **http://0.0.0.0:8000** (port 8000). Stop with **Ctrl+C**. API docs: http://localhost:8000/docs — Health: http://localhost:8000/health

**Endpoints:** POST `/query` — body `{"query": "Your question", "model": "gpt-3.5-turbo"}` (model optional). POST `/ingest` — optional body `{"url": "https://..."}` to ingest that RSS feed (default: BBC).

**Example: RAG query**
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "Latest on AI regulations in EU"}'
```

**Example: Ingest**
```bash
curl -X POST http://localhost:8000/ingest
```



**Step by step from Python (without using the server)**

```python
from ingester import ingest_news_feed
from rag_llm import rag_query

# 1. Ingest (do once, or when you want to refresh the index)
ingest_news_feed()   # optional: ingest_news_feed("https://other-feed.xml")

# 2. Ask anything; answer is grounded in retrieved news
result = rag_query("Latest on AI regulations in EU")
print(result)
```

### 3.4 Invoking RAG from Your Code (LLM in the loop)

To use the RAG pipeline from your own script or app:

```python
from rag_llm import rag_query

# Uses default model (e.g. gpt-3.5-turbo); set OPENAI_API_KEY or use Ollama
answer = rag_query("What is the latest on climate summit 2025?")
print(answer)
```

**Using a specific model:**

```python
# OpenAI
answer = rag_query("Summarize today's tech news", model="gpt-4o-mini")

# Ollama (local)
answer = rag_query("Summarize today's tech news", model="ollama/llama3")
```

**Flow inside `rag_query`:**

1. **Refine** — The LLM turns your question into a sharper search query (e.g. for “current affairs”).
2. **Retrieve** — `searcher.search_topic(refined_query)` does semantic search in ChromaDB and returns the top chunks + metadata.
3. **Augment & generate** — Retrieved text is added to the prompt; the LLM answers using that context.

So “invoking from an LLM” here means: your user question → (optional LLM query refinement) → retrieval → LLM generation with context. You can reuse `rag_query` in a chatbot, API, or CLI.

### 3.5 Project Layout (for reference)

| File / folder | Role |
|---------------|------|
| `ingester.py` | Fetches RSS, scrapes articles, embeds and indexes into ChromaDB. |
| `searcher.py` | Loads ChromaDB collection and runs semantic search. |
| `rag_llm.py` | Query refinement + retrieval + prompt building + LLM call (`rag_query`). |
| `main.py` | Starts the RAG server (run until Ctrl+C). |
| `server.py` | FastAPI app: `/query`, `/ingest`, `/health`, `/docs`. |
| `requirements.txt` | Python dependencies (see above). |
| `./rag_db/` | ChromaDB data (created on first ingest). |

---

## Pushing to GitHub

To put this project in your GitHub account as **ML-RAG-101**:

1. **Create the repository on GitHub**  
   - Go to [github.com/new](https://github.com/new).  
   - Repo name: `ML-RAG-101`.  
   - Choose Public (or Private).  
   - Do **not** add a README, .gitignore, or license (this project already has them).  
   - Click **Create repository**.

2. **Commit locally** (if you haven’t already):
   ```bash
   cd /path/to/ML-Rag-101
   git add .
   git commit -m "Initial commit: ML-RAG-101 news RAG pipeline"
   ```

3. **Add GitHub as remote and push**:
   ```bash
   git remote add origin https://github.com/anandharidas/ML-RAG-101.git
   git branch -M main
   git push -u origin main
   ```
   If you use SSH: `git remote add origin git@github.com:anandharidas/ML-RAG-101.git`

4. **If the repo already existed** and you created it with a README, pull first then push:
   ```bash
   git remote add origin https://github.com/anandharidas/ML-RAG-101.git
   git pull origin main --allow-unrelated-histories
   git push -u origin main
   ```

Your repo will be at: **https://github.com/anandharidas/ML-RAG-101**

---

## Quick reference: dependency list

From `requirements.txt`:

- **requests**, **beautifulsoup4** — Fetching and parsing RSS/HTML  
- **Pillow** — Image handling (ingester)  
- **sentence-transformers**, **chromadb** — Embeddings and vector store  
- **litellm** — LLM calls (OpenAI, Ollama, etc.)

Install all: `pip install -r requirements.txt`.
