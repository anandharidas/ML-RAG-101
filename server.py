"""
RAG server: serves RAG queries and ingest until process is killed.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingester import ingest_news_feed
from rag_llm import rag_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG server starting; use Ctrl+C to stop.")
    yield
    logger.info("RAG server shutting down.")


app = FastAPI(
    title="ML-RAG-101",
    description="RAG pipeline server: query news index and trigger ingest.",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question for the RAG pipeline")
    model: str = Field(default="gpt-3.5-turbo", description="LLM model (e.g. gpt-3.5-turbo, ollama/llama3)")


class QueryResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    url: str = Field(
        default="http://feeds.bbci.co.uk/news/rss.xml",
        description="RSS feed URL to ingest",
    )


class IngestResponse(BaseModel):
    status: str = "ok"
    message: str


@app.get("/")
def root():
    return {
        "service": "ML-RAG-101",
        "docs": "/docs",
        "query": "POST /query",
        "ingest": "POST /ingest",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def post_query(body: QueryRequest):
    """Run RAG: refine query, retrieve from index, generate answer."""
    try:
        answer = rag_query(user_query=body.query, model=body.model)
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def post_ingest(body: IngestRequest | None = Body(None)):
    """Ingest RSS feed into the vector index. Optional body: { \"url\": \"...\" }."""
    url = body.url if body else "http://feeds.bbci.co.uk/news/rss.xml"
    try:
        ingest_news_feed(url=url)
        return IngestResponse(message=f"Ingested feed: {url}")
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
