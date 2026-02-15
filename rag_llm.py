import logging
import litellm  # pip install litellm (supports OpenAI/Ollama)
from ingester import ingest_news_feed  # Re-ingest if needed
from searcher import search_topic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your API key: os.environ["OPENAI_API_KEY"] or for Ollama: litellm.model="ollama/llama3"

def generate_query_llm(user_query: str, model: str = "gpt-3.5-turbo"):
    """LLM refines user query for better search."""
    prompt = f"Refine this into a precise search query for current affairs: {user_query}"
    resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
    refined = resp.choices[0].message.content.strip()
    logger.info(f"Refined query: '{refined}'")
    return refined

def rag_query(user_query: str, model: str = "gpt-3.5-turbo"):
    """Full RAG: Refine → Search → Augment LLM."""
    logger.info(f"RAG query: '{user_query}'")
    
    # Step 1: Refine query
    refined_query = generate_query_llm(user_query, model)
    
    # Step 2: Retrieve
    docs, metas = search_topic(refined_query)
    context = "\n\n".join([f"{meta['title']}: {doc[:500]}..." for doc, meta in zip(docs, metas)])
    
    # Step 3: LLM with RAG context
    rag_prompt = f"""Using this current affairs context:
{context}

Answer: {user_query}"""
    
    resp = litellm.completion(
        model=model,
        messages=[{"role": "system", "content": "You are a news analyst."}, {"role": "user", "content": rag_prompt}]
    )
    answer = resp.choices[0].message.content.strip()
    logger.info("RAG response generated")
    return answer
