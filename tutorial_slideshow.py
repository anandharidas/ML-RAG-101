#!/usr/bin/env python3
"""
RAG Tutorial — Rich terminal slideshow.
Covers RAG theory, how to construct RAG, and how LLM, Agent, and RAG interact.
Run: python tutorial_slideshow.py
"""
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

SLIDES = [
    {
        "title": "RAG Tutorial: From Theory to This Project",
        "body": """
# Retrieval-Augmented Generation (RAG)

This slideshow covers:
- **RAG theory** — what it is and why it matters
- **How to construct RAG** — essential building blocks
- **LLM, Agent & RAG** — how they talk to each other
- **This project** — ML-RAG-101 architecture and flow

[Press Enter to continue]
""",
    },
    {
        "title": "What is RAG?",
        "body": """
**RAG = Retrieval-Augmented Generation**

- Combine **retrieval** (fetch relevant documents from a corpus) with **generation** (LLM produces an answer).
- The LLM is *augmented* by external knowledge at query time instead of relying only on its training data.

**Core idea:** Ground the model’s answer in real documents (your docs, feeds, DB) so answers are up-to-date and traceable.
""",
    },
    {
        "title": "Why RAG?",
        "body": """
**Limits of “LLM-only” systems:**
- Knowledge cut-off → no recent events or private data.
- Hallucinations → no way to check facts.
- No citations → users can’t verify.

**What RAG adds:**
- **Fresh knowledge** from your indexed sources (e.g. news, docs).
- **Evidence** — retrieved chunks can back the answer.
- **Control** — you choose what goes into the index and what gets retrieved.
""",
    },
    {
        "title": "RAG Pipeline — Two Phases",
        "body": """
**1. Indexing (offline / on ingest)**
- Ingest sources (RSS, PDFs, web) → chunk text → compute embeddings → store in a vector DB (e.g. Chroma).

**2. Query (online)**
- User question → (optional) query refinement → retrieve top-k similar chunks → build a prompt with “context” + question → LLM generates answer.

This project does both: **ingester** = indexing, **searcher + rag_llm** = query.
""",
    },
    {
        "title": "How to Construct RAG — Essentials",
        "body": """
| Step | What you need | In this project |
|------|----------------|-----------------|
| **Ingest** | Load and parse sources | RSS feeds → BeautifulSoup, feedsources.md |
| **Chunk** | Split into searchable units | Per-article docs (title + desc + excerpt) |
| **Embed** | Vector representation | SentenceTransformer + Chroma |
| **Store** | Vector DB | ChromaDB `news` collection |
| **Retrieve** | Similarity search | `searcher.search_topic()` |
| **Augment** | Add context to prompt | Concatenate retrieved docs |
| **Generate** | LLM answer | `rag_llm.rag_query()` via LiteLLM |
""",
    },
    {
        "title": "LLM vs Agent vs RAG",
        "body": """
**LLM** — A model that takes text in, produces text out. No built-in tools or memory; can be used for refinement, summarization, or final answer.

**Agent** — A system that uses an LLM to *decide* what to do next (call a tool, search, run code) and iterates until the task is done. Has a loop: think → act → observe.

**RAG** — A fixed pipeline: retrieve relevant docs, then pass them + user question to an LLM to generate one answer. No decision loop; retrieval is a defined step, not a “tool” the model chooses.
""",
    },
    {
        "title": "How LLM, Agent, and RAG Talk to Each Other",
        "body": """
**RAG uses the LLM in two places (in this project):**
1. **Query refinement** — User question → LLM → better search query (e.g. “latest on GPT” → “GPT-4 API updates 2024”).
2. **Answer generation** — Retrieved context + user question → LLM → final answer.

**Agent could sit on top:** An agent could *choose* to call RAG as one of its tools (e.g. “search our news index”) and then use the RAG result along with other tools (calculator, web search) to complete a task.

**Flow:** User → [Agent?] → RAG (refine → retrieve → augment → LLM) → Answer.
""",
    },
    {
        "title": "This Project: ML-RAG-101 Architecture",
        "body": """
```
feedsources.md → ingester (ingest_all_sources / ingest_news_feed)
                        ↓
                 ChromaDB "news" (embeddings + metadata)
                        ↑
user query → server (POST /query) → rag_llm.rag_query()
                        ↓
              generate_query_llm() → search_topic() → LLM with context
                        ↓
                   answer → client
```
- **ingester.py** — Parses feeds, fetches articles, embeds, writes to Chroma.
- **searcher.py** — Semantic search over Chroma.
- **rag_llm.py** — Refine query (LLM), retrieve (searcher), then generate (LLM).
- **server.py** — FastAPI: /query (RAG), /ingest (add feed).
""",
    },
    {
        "title": "Flow in Code (Query Path)",
        "body": """
**main.py** → On start: `ingest_all_sources()` then uvicorn.

**POST /query** (server.py):
1. `rag_query(body.query)` (rag_llm.py)
2. `generate_query_llm(user_query)` → refined query (LLM)
3. `search_topic(refined_query)` (searcher.py) → docs, metas from Chroma
4. Build prompt: context (retrieved snippets) + user question
5. `litellm.completion(..., rag_prompt)` → answer

**Ingest path:** POST /ingest or startup → `ingest_news_feed(url)` → fetch RSS → parse → embed → `collection.add(...)`.
""",
    },
    {
        "title": "Summary",
        "body": """
- **RAG** = Retrieve relevant docs, then generate an answer with an LLM so responses are grounded and up-to-date.
- **Constructing RAG:** Ingest → Chunk → Embed → Store → Retrieve → Augment prompt → Generate.
- **LLM** is used inside RAG for refinement and for the final answer; an **Agent** could use RAG as one of its tools in a broader loop.

**This project:** Static ingester (feedsources.md) + Chroma + searcher + rag_llm + FastAPI = end-to-end RAG over tech news.

[End of tutorial — press Enter to exit]
""",
    },
]


def show_slide(index: int, total: int) -> None:
    slide = SLIDES[index]
    title = slide["title"]
    body = slide["body"]
    footer = Text(f" Slide {index + 1} / {total} ", style="dim")
    content = Markdown(body)
    panel = Panel(
        content,
        title=title,
        title_align="left",
        subtitle=footer,
        subtitle_align="right",
        box=box.ROUNDED,
        border_style="blue",
        padding=(1, 2),
    )
    console.clear()
    console.print(panel)
    console.print()
    console.print("[dim]Press Enter for next slide (or Ctrl+C to quit)[/dim]")


def main() -> None:
    total = len(SLIDES)
    for i in range(total):
        show_slide(i, total)
        try:
            input()
        except KeyboardInterrupt:
            console.print("\n[yellow]Slideshow stopped.[/yellow]")
            break
    else:
        console.print("[green]Thanks for watching![/green]")


if __name__ == "__main__":
    main()
