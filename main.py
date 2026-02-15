from ingester import ingest_news_feed
from rag_llm import rag_query

# Ingest once
ingest_news_feed()  # Logs indexing

# Query
result = rag_query("Latest on AI regulations in EU")
print(result)  # LLM-enhanced answer with logs throughout
