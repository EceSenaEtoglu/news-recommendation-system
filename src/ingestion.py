# src/ingestion.py
import asyncio
import os
from datetime import datetime
from typing import List
from .providers import create_provider
from .storage import ArticleDB
from .models import Article

# TODO just an architectural template
# match with the actual api
class NewsIngestionPipeline:
    """Pipeline for fetching and storing news articles"""
    
    def __init__(self, api_key: str, db_path: str = "db/articles.db"):
        self.provider = create_provider(api_key)
        self.db = ArticleDB(db_path)
        
    async def ingest_articles(self, limit: int = 100) -> dict:
        """Main ingestion pipeline"""
        print(f" Starting article ingestion (limit: {limit})...")
        
        # Step 1: Fetch articles from NewsAPI
        articles = await self.provider.fetch_articles(limit=limit)
        print(f" Fetched {len(articles)} articles from NewsAPI")
        
        if not articles:
            return {"success": False, "message": "No articles fetched"}
        
        # Step 2: Process articles (bias detection happens automatically in __post_init__)
        processed_articles = self._process_articles(articles)
        print(f" Processed {len(processed_articles)} articles")
        
        # Step 3: Save to database
        saved_count = self.db.save_articles(processed_articles)
        print(f" Saved {saved_count} articles to database")
        
        # Step 4: Show stats
        stats = self.db.get_stats()
        self._print_stats(processed_articles, stats)
        
        return {
            "success": True,
            "fetched": len(articles),
            "processed": len(processed_articles),
            "saved": saved_count,
            "stats": stats
        }
    
    def _process_articles(self, articles: List[Article]) -> List[Article]:
        """Additional processing if needed"""
        processed = []
        
        for article in articles:
            # Filter out very short articles
            if len(article.content) < 200:
                continue
                
            # Filter out articles without meaningful titles
            if len(article.title) < 20:
                continue
            
            # Auto-bias detection already happened in Article.__post_init__
            processed.append(article)
        
        return processed
    
    def _print_stats(self, articles: List[Article], db_stats: dict):
        """Print interesting statistics"""
        print("\n Ingestion Results:")
        print(f"   Total articles in DB: {db_stats['total_articles']}")
        
        # Content type breakdown
        content_types = {}
        for article in articles:
            ct = article.content_type.value
            content_types[ct] = content_types.get(ct, 0) + 1
        
        print(f"   Content types in this batch:")
        for ct, count in content_types.items():
            print(f"     {ct}: {count}")
        
        # Target audience breakdown  
        audiences = {}
        for article in articles:
            aud = article.target_audience.value
            audiences[aud] = audiences.get(aud, 0) + 1
        
        print(f"   Target audiences in this batch:")
        for aud, count in audiences.items():
            print(f"     {aud}: {count}")
        
        # Urgency breakdown
        high_urgency = len([a for a in articles if a.urgency_score > 0.7])
        print(f"   High urgency articles: {high_urgency}")
        
        print(f"   Top sources: {db_stats.get('top_sources', {})}")

# Convenience function for scripts
async def ingest_news(api_key: str, limit: int = 50):
    """Simple function to ingest news"""
    pipeline = NewsIngestionPipeline(api_key)
    return await pipeline.ingest_articles(limit)

# Script mode
if __name__ == "__main__":
    import sys
    
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print(" Set NEWSAPI_KEY environment variable")
        sys.exit(1)
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    # Run ingestion
    result = asyncio.run(ingest_news(api_key, limit))
    
    if result["success"]:
        print(f"\n Ingestion complete! Saved {result['saved']} articles.")
    else:
        print(f" Ingestion failed: {result['message']}")