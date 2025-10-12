"""
Unified Recommendation System using MultiRAGRetriever
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone

from src.data_models import Article, SearchQuery, SearchResult
from src.config import RAGConfig
from src.storage import ArticleDB
from src.embeddings import EmbeddingSystem
from src.retrieval import MultiRAGRetriever


class RecommendationSystem:
    """
    Recommendation system interface that uses MultiRAGRetriever for hybrid search
    and adds recommendation-specific features like topic overlap.
    """
    
    def __init__(self, db: ArticleDB, embeddings: EmbeddingSystem, config: RAGConfig = None):
        self.db = db
        self.embeddings = embeddings
        self.config = config or RAGConfig()
        
        # Initialize the hybrid search system
        self.retriever = MultiRAGRetriever(db, embeddings, self.config)
    
    def recommend_for_article(self, current: Article, k: Optional[int] = None) -> List[Tuple[Article, float]]:
        """
        Get recommendations for an article using hybrid search + recommendation features.
        
        Pipeline:
        1. Build query from article content
        2. Use MultiRAGRetriever for hybrid search (BM25 + semantic + RRF + Graph RAG + MMR)
        3. Apply topic overlap bonus
        4. Deduplicate results
        """
        k = k or self.config.top_k
        
        # Step 1: Build query text from article
        query_text = self._article_to_query_text(current)
        
        # Step 2: Use MultiRAGRetriever for hybrid search (includes MMR diversification)
        search_query = SearchQuery(text=query_text, limit=k * 4)  # Get more candidates
        import asyncio
        search_results = asyncio.run(self.retriever.search(search_query))
        
        if not search_results:
            return []
        
        # Step 3: Convert to (article, score) format and exclude current article
        candidates = []
        current_topics = set(current.topics or [])
        
        for result in search_results:
            if result.article.id == current.id:
                continue
                
            # Apply topic overlap bonus
            score = result.final_score
            if current_topics and result.article.topics:
                overlap = current_topics.intersection(set(result.article.topics))
                bonus = min(self.config.max_topic_bonus, 
                           self.config.topic_overlap_boost * len(overlap))
                score += bonus
            
            candidates.append((result.article, score))
        
        if not candidates:
            return []
        
        # Step 4: Sort by score and take top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:k]
        
        # Step 5: Deduplicate results
        candidates = self._deduplicate_results(candidates, current)
        
        return candidates[:k]
    
    def _article_to_query_text(self, article: Article) -> str:
        """Build query text from article content."""
        description = article.description or ""
        content_head = (article.content or "")[:500]
        return f"{article.title} {description} {content_head}".strip()
    
    
    def _deduplicate_results(self, candidates: List[Tuple[Article, float]], current: Article) -> List[Tuple[Article, float]]:
        """Remove duplicate articles based on URL and title similarity."""
        
        def _normalize_url(url: Optional[str]) -> str:
            if not url:
                return ""
            return url.strip().lower().rstrip("/")
        
        def _normalize_title(title: Optional[str]) -> str:
            if not title:
                return ""
            return title.strip().lower()
        
        # Normalize current article
        current_url = _normalize_url(getattr(current, "url", None))
        current_title = _normalize_title(getattr(current, "title", None))
        
        seen_urls = {current_url} if current_url else set()
        seen_titles = {current_title} if current_title else set()
        
        deduplicated = []
        for article, score in candidates:
            url = _normalize_url(getattr(article, "url", None))
            title = _normalize_title(getattr(article, "title", None))
            
            # Skip if URL or title already seen
            if (url and url in seen_urls) or (title and title in seen_titles):
                continue
            
            # Add to seen sets
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
            
            deduplicated.append((article, score))
        
        return deduplicated
    
    def explain_recommendation(self, seed: Article, candidate: Article) -> str:
        """Generate explanation for why an article was recommended."""
        reasons = []
        
        # Topic overlap
        seed_topics = set(seed.topics or [])
        cand_topics = set(candidate.topics or [])
        overlap = seed_topics.intersection(cand_topics)
        if overlap:
            reasons.append(f"overlap topics: {', '.join(list(overlap)[:2])}")
        
        # Recency
        try:
            hours_old = (datetime.now(timezone.utc) - 
                        candidate.published_at.replace(tzinfo=timezone.utc).astimezone(timezone.utc)
                        ).total_seconds() / 3600.0
        except Exception:
            hours_old = 9999
        
        if hours_old < 6:
            reasons.append("very recent")
        elif hours_old < 24:
            reasons.append("recent")
        
        # Source credibility
        if getattr(candidate.source, "credibility_score", 0.5) > 0.8:
            reasons.append("trusted source")
        
        # Similarity (from hybrid search)
        reasons.append("similar content")
        
        return ", ".join(reasons[:3]) if reasons else "relevant"


# Convenience function for quick demos
def get_recommendations_for_article_id(db: ArticleDB, embeddings: EmbeddingSystem, 
                                     article_id: str, top_k: int = 10, 
                                     config: RAGConfig = None) -> List[Tuple[Article, float]]:
    """Get recommendations for an article by ID."""
    article = db.get_article_by_id(article_id)
    if not article:
        return []
    
    if config is None:
        config = RAGConfig()
    config.top_k = top_k
    recommender = RecommendationSystem(db, embeddings, config)
    
    return recommender.recommend_for_article(article)
