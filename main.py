"""
Start the RAG server. Runs until manually killed (Ctrl+C).
On startup, runs the static ingester to ingest from all URLs in feedsources.md.
"""
from dotenv import load_dotenv
import uvicorn

from ingester import ingest_all_sources

load_dotenv()

if __name__ == "__main__":
    ingest_all_sources()
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
