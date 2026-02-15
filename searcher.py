import logging
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./rag_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="news")

def search_topic(query: str, n_results: int = 3):
    """Semantic search for topic/excerpt."""
    logger.info(f"Searching for: '{query}'")
    results = collection.query(query_texts=[query], n_results=n_results)
    logger.info(f"Retrieved {len(results['documents'][0])} results")
    
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        logger.info(f"Hit {i+1}: {meta['title']} (images: {meta.get('images', [])})")
    
    return results['documents'][0], results['metadatas'][0]
