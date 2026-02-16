import logging
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default path to feed sources (relative to project root)
FEED_SOURCES_PATH = Path(__file__).resolve().parent / "feedsources.md"


def _urls_from_feedsources(path: Path | None = None) -> list[str]:
    """Parse feedsources.md table and return list of RSS URLs."""
    path = path or FEED_SOURCES_PATH
    if not path.exists():
        logger.warning(f"Feed sources file not found: {path}")
        return []
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("| :---"):
                continue
            if line.startswith("| "):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    # RSS URL is second column (index 1)
                    raw = parts[1]
                    # Handle markdown links [text](url)
                    match = re.search(r"https?://[^\s)\]\"']+", raw)
                    candidate = match.group(0) if match else raw
                    if candidate.startswith("http"):
                        urls.append(candidate)
    return urls


def ingest_all_sources(sources_path: Path | None = None) -> None:
    """Ingest from all RSS URLs documented in feedsources.md."""
    urls = _urls_from_feedsources(sources_path)
    if not urls:
        logger.info("No feed URLs to ingest.")
        return
    logger.info(f"Ingesting from {len(urls)} feed sources.")
    for url in urls:
        try:
            ingest_news_feed(url)
        except Exception as e:
            logger.exception(f"Failed to ingest {url}: {e}")

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
    for i, item in enumerate(items[:1000]):  # Limit 10
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
