"""
Start the RAG server. Runs until manually killed (Ctrl+C).
"""
from dotenv import load_dotenv
import uvicorn

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
