import logging
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./rag_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="news", embedding_function=ef)

def ingest_news_feed(url: str = "http://feeds.bbci.co.uk/news/rss.xml"):
    """Ingest RSS feed, extract content, index all."""
    logger.info(f"Fetching news from {url}")
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'xml')
    items = soup.find_all('item')
    
    docs, metas, ids, images_desc = [], [], [], []
    for i, item in enumerate(items[:10]):  # Limit 10
        title = item.title.text
        desc = item.description.text
        link = item.link.text
        pubdate = item.pubDate.text if item.pubDate else ""
        
        # Fetch full article for text/images
        art_resp = requests.get(link)
        art_soup = BeautifulSoup(art_resp.content, 'html.parser')
        full_text = art_soup.get_text(separator=' ', strip=True)[:2000]  # Truncate
        
        # Extract image desc (alt text/metadata)
        imgs = art_soup.find_all('img')
        img_descs = [img.get('alt', '') for img in imgs[:3]]
        
        doc = f"{title}\n{desc}\n{full_text}"
        meta = {"title": title, "url": link, "date": pubdate, "images": img_descs}
        
        docs.append(doc)
        metas.append(meta)
        ids.append(f"doc_{i}")
        images_desc.append("; ".join(img_descs))
        
        logger.info(f"Processed article {i+1}: {title}")
    
    # Index
    collection.add(documents=docs, metadatas=metas, ids=ids)
    logger.info(f"Indexed {len(docs)} articles with images/metadata")
