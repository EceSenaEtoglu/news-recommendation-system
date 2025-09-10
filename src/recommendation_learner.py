from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from .data_models import Article
from .storage import ArticleDB
from .embeddings import EmbeddingSystem


@dataclass
class RecommendationConfig:
    """Lightweight config for the AI recommender."""
    top_k: int = 10
    min_score: float = 0.0  # cosine similarity threshold in FAISS (0..1 after normalize)
    topic_overlap_boost: float = 0.1  # additional boost per topic overlap (small)
    max_topic_bonus: float = 0.5      # cap total bonus from topics


class AIRecommender:
    """Recommend similar articles to a given article using semantic similarity.

    Approach:
    - Build a text query from the current article (title + description + first 500 chars of content)
    - Use the EmbeddingSystem to perform a semantic search against the local FAISS index
    - Exclude the current article from results
    - Apply a small topic-overlap boost (no topic extraction; uses existing article.topics)
    - Return articles sorted by final score
    """

    def __init__(self, db: ArticleDB, embeddings: EmbeddingSystem, config: Optional[RecommendationConfig] = None):
        self.db = db
        self.embeddings = embeddings
        self.config = config or RecommendationConfig()

    def _article_to_query_text(self, article: Article) -> str:
        description = article.description or ""
        content_head = (article.content or "")[:500]
        return f"{article.title} {description} {content_head}".strip()

    def recommend_for_article(self, current: Article, k: Optional[int] = None) -> List[Tuple[Article, float]]:
        """Return a ranked list of (Article, score) similar to the current article."""
        k = k or self.config.top_k

        # Build query text from the article itself (we have full text locally)
        query_text = self._article_to_query_text(current)

        # Semantic search over FAISS using the query text
        base_results = self.embeddings.semantic_search(query_text, k=k + 10, score_threshold=self.config.min_score)
        if not base_results:
            return []

        # Map article_id -> base_score (exclude the same article id)
        base_scores = {aid: score for (aid, score) in base_results if aid != current.id}
        if not base_scores:
            return []

        # Fetch candidates in one batch
        candidates = self.db.get_articles_by_ids(list(base_scores.keys()))
        current_topics = set((current.topics or []))

        ranked: List[Tuple[Article, float]] = []
        for cand in candidates:
            score = float(base_scores.get(cand.id, 0.0))

            # Topic-overlap bonus (simple and bounded)
            if current_topics and cand.topics:
                overlap = current_topics.intersection(set(cand.topics))
                bonus = min(self.config.max_topic_bonus, self.config.topic_overlap_boost * len(overlap))
                score += bonus

            ranked.append((cand, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:k]


# Convenience helper for quick demos/portfolio snippets

def get_recommendations_for_article_id(db: ArticleDB, embeddings: EmbeddingSystem, article_id: str, top_k: int = 10) -> List[Tuple[Article, float]]:
    """Fetch an article by id and return similar articles with scores."""
    article = db.get_article_by_id(article_id)
    if not article:
        return []
    recommender = AIRecommender(db, embeddings, RecommendationConfig(top_k=top_k))
    return recommender.recommend_for_article(article)