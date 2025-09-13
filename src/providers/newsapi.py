# src/providers/newsapi.py
import requests
from datetime import datetime
from typing import List, Optional
from .base import ArticleProvider
from ..data_models import Article, Source, ContentType
from datetime import timezone

# TODO, match it with the API calls!

class NewsAPIProvider(ArticleProvider):
    """Simple NewsAPI provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
        # Simple source list with credibility info
        self.sources = {
            "bbc-news": Source("bbc-news", "BBC News", "https://bbc.com", "general", "gb", "en", 0.9),
            "cnn": Source("cnn", "CNN", "https://cnn.com", "general", "us", "en", 0.8),
            "fox-news": Source("fox-news", "Fox News", "https://foxnews.com", "general", "us", "en", 0.7),
            "reuters": Source("reuters", "Reuters", "https://reuters.com", "general", "us", "en", 0.95),
        }
    
    async def fetch_articles(self, limit: int = 100) -> List[Article]:
        """Fetch articles from NewsAPI"""
        url = f"{self.base_url}/everything"
        
        params = {
            "apiKey": self.api_key,
            "sources": "bbc-news,cnn,fox-news,reuters",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100)
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for item in data.get("articles", []):
                # Skip articles without content
                if not item.get("content") or "[Removed]" in item.get("content", ""):
                    continue
                    
                # Get source info
                source_id = item.get("source", {}).get("id", "unknown")
                source = self.sources.get(source_id, self._create_unknown_source(item))
                
                # Parse date
                try:
                    published_at = datetime.fromisoformat(item["publishedAt"].replace("Z","+00:00"))
                    published_at = published_at.astimezone(timezone.utc)
                except:
                    published_at = datetime.now()
                
                # Create article
                article = Article(
                    id=str(hash(item["url"])),
                    title=item["title"],
                    content=item["content"],
                    description=item.get("description", ""),
                    url=item["url"],
                    source=source,
                    published_at=published_at,
                    author=item.get("author"),
                    url_to_image=item.get("urlToImage")
                )
                
                articles.append(article)
            
            return articles
            
        except Exception as e:
            print(f"Error fetching articles: {e}")
            return []
    
    def get_sources(self) -> List[Source]:
        """Get available sources"""
        return list(self.sources.values())
    
    def _create_unknown_source(self, item):
        """Create source for unknown publishers"""
        source_name = item.get("source", {}).get("name", "Unknown")
        return Source(
            id="unknown",
            name=source_name,
            url="",
            category="general",
            credibility_score=0.5
        )