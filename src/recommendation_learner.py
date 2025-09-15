from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from .data_models import Article
from .storage import ArticleDB
from .embeddings import EmbeddingSystem
from .reranker import RerankFeatureExtractor, TrainableLogisticReranker, NeuralReranker, AdvancedFeatureExtractor, NeuralRerankerManager
from .config import RecommendationConfig, RerankFeatureConfig, NeuralRerankerConfig
from datetime import datetime, timezone
import numpy as np


class AIRecommender:
    """Recommend similar articles to a given article using semantic similarity.

    Approach:
    - Build a text query from the current article (title + description + first 500 chars of content)
    - Use the EmbeddingSystem to perform a semantic search against the local FAISS index
    - Exclude the current article from results
    - Apply a small topic-overlap boost (no topic extraction; uses existing article.topics)
    - Optionally apply MMR to diversify the final top-k list
    - Return articles sorted by final score
    """

    def __init__(self, db: ArticleDB, embeddings: EmbeddingSystem, config: Optional[RecommendationConfig] = None):
        self.db = db
        self.embeddings = embeddings
        self.config = config or RecommendationConfig()
        self._feature_extractor = RerankFeatureExtractor(db, embeddings, RerankFeatureConfig())
        self._logistic_reranker: Optional[TrainableLogisticReranker] = None
        
        # Neural reranker components (for future use - requires user interaction data)
        self._neural_reranker_manager: Optional[NeuralRerankerManager] = None
        
        if self.config.use_neural_reranker:
            self._init_neural_reranker()

    def _init_neural_reranker(self):
        """Initialize neural reranker with appropriate input dimension (for future use)"""
        # Input dimension based on advanced feature extractor
        input_dim = 10  # Number of features extracted by AdvancedFeatureExtractor
        neural_reranker = NeuralReranker(input_dim, self.config.neural_config)
        advanced_feature_extractor = AdvancedFeatureExtractor(self.db, self.embeddings)
        self._neural_reranker_manager = NeuralRerankerManager(
            neural_reranker, advanced_feature_extractor, self.config.neural_config, self.db, self.embeddings
        )
        print(f"Initialized neural reranker with {input_dim} input features")
    
    def train_neural_reranker(self, user_id: str = "default", days: int = 14):
        """Train the neural reranker on user interaction data (for future use)"""
        if not self._neural_reranker_manager:
            print("Neural reranker not initialized")
            return
        
        self._neural_reranker_manager.train_neural_reranker(user_id, days)
    

    def _article_to_query_text(self, article: Article) -> str:
        description = article.description or ""
        content_head = (article.content or "")[:500]
        return f"{article.title} {description} {content_head}".strip()

    def recommend_for_article(self, current: Article, k: Optional[int] = None) -> List[Tuple[Article, float]]:
        """Return a ranked list of (Article, score) similar to the current article by applying several methods:
        Methods: 1. Semantic search, 2. Topic overlap bonus, 3. MMR Diversification, 4. Neural reranking or Trainable logistic reranker."""
        k = k or self.config.top_k

        # Build query text from the article itself (we have full text locally)
        query_text = self._article_to_query_text(current)

        # Semantic search over FAISS using the query text
        pool_n = max(k + 10, self.config.mmr_pool if self.config.use_mmr else k + 10)
        base_results = self.embeddings.semantic_search(query_text, k=pool_n, score_threshold=self.config.min_score)
        if not base_results:
            return []

        # Map article_id -> base_score (exclude the same article id)
        base_scores = {aid: score for (aid, score) in base_results if aid != current.id}
        if not base_scores:
            return []

        # Fetch candidates in one batch
        candidates = self.db.get_articles_by_ids(list(base_scores.keys()))
        id_to_cand = {a.id: a for a in candidates}
        current_topics = set((current.topics or []))

        # Build initial relevance scores with small topic overlap bonus
        scored: List[Tuple[str, float]] = []
        for aid, score in base_scores.items():
            cand = id_to_cand.get(aid)
            if not cand:
                continue

            # apply topic overlap bonus if there are topics in candidates and current article
            if current_topics and cand.topics:
                overlap = current_topics.intersection(set(cand.topics))
                score = float(score) + min(self.config.max_topic_bonus, self.config.topic_overlap_boost * len(overlap))
            scored.append((aid, float(score)))

        if not scored:
            return []

        # If MMR is disabled, return top-k by score
        if not self.config.use_mmr:
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = [(id_to_cand[aid], s) for aid, s in scored if aid in id_to_cand]
        else:
            # apply MMR Diversification
            pool = sorted(scored, key=lambda x: x[1], reverse=True)[: self.config.mmr_pool]
            pool_ids = [aid for aid, _ in pool]
            pool_articles = [id_to_cand[aid] for aid in pool_ids if aid in id_to_cand]
            pool_texts = [self._article_to_query_text(a) for a in pool_articles]
            if not pool_texts:
                return []
            vecs = self.embeddings.encode_texts(pool_texts).astype("float32", copy=False)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
            id_to_score = dict(pool)
            lambda_ = float(self.config.mmr_lambda)
            selected: List[int] = []
            remaining = list(range(len(pool_ids)))
            sim = vecs @ vecs.T
            np.fill_diagonal(sim, 1.0)
            while remaining and len(selected) < k:
                best_idx = None
                best_val = -1e9
                for idx in remaining:
                    aid = pool_ids[idx]
                    rel = id_to_score.get(aid, 0.0)
                    if not selected:
                        mmr_val = rel
                    else:
                        max_sim = max(sim[idx, j] for j in selected)
                        # MMR Score = λ × Relevance - (1-λ) × Max_Similarity
                        mmr_val = lambda_ * rel - (1.0 - lambda_) * max_sim
                    if mmr_val > best_val:
                        best_val = mmr_val
                        best_idx = idx
                selected.append(best_idx)
                remaining.remove(best_idx)
            ranked = [(id_to_cand[pool_ids[i]], id_to_score[pool_ids[i]]) for i in selected if pool_ids[i] in id_to_cand]
        
        # Optional neural reranking (for future use - requires user interaction data)
        if self._neural_reranker_manager and ranked:
            ranked = self._neural_reranker_manager.apply_neural_reranking(current, ranked, base_scores)
        
        # Optional trainable logistic rerank on the ranked list (fallback)
        elif self._logistic_reranker and ranked:
            arts = [a for a, _ in ranked]
            base_scores = {a.id: s for a, s in ranked}
            X, ids = self._feature_extractor.build_features(current, arts, base_scores)
            if X.size:
                preds = self._logistic_reranker.predict_scores(X)
                id_to_pred = {aid: float(p) for aid, p in zip(ids, preds)}
                ranked = sorted(ranked, key=lambda t: id_to_pred.get(t[0].id, t[1]), reverse=True)

        # --- Post-filter: ensure no duplicates and exclude anything identical to seed ---
        def _norm_url(u: Optional[str]) -> str:
            try:
                u = (u or "").strip().lower()
                return u.rstrip("/")
            except Exception:
                return ""

        def _norm_title(t: Optional[str]) -> str:
            return (t or "").strip().lower()

        seed_url = _norm_url(getattr(current, "url", None))
        seed_title = _norm_title(getattr(current, "title", None))

        seen_urls = set([seed_url]) if seed_url else set()
        seen_titles = set([seed_title]) if seed_title else set()

        deduped: List[Tuple[Article, float]] = []
        for art, score in ranked:
            u = _norm_url(getattr(art, "url", None))
            t = _norm_title(getattr(art, "title", None))
            if (u and u in seen_urls) or (t and t in seen_titles):
                continue
            if u:
                seen_urls.add(u)
            if t:
                seen_titles.add(t)
            deduped.append((art, score))

        # Ensure the final list is presented in descending score order 
        deduped.sort(key=lambda x: x[1], reverse=True)
        return deduped[:k]

    def set_logistic_reranker(self, model: TrainableLogisticReranker):
        self._logistic_reranker = model

    def explain_recommendation(self, seed: Article, candidate: Article) -> str:
        reasons = []
        # Topic overlap
        seed_topics = set(seed.topics or [])
        cand_topics = set(candidate.topics or [])
        inter = [t for t in cand_topics if t in seed_topics]
        if inter:
            reasons.append(f"overlap topics: {', '.join(inter[:2])}")
        # Recency
        try:
            hours_old = (datetime.utcnow() - candidate.published_at.replace(tzinfo=datetime.timezone.utc).astimezone(datetime.timezone.utc)).total_seconds() / 3600.0
        except Exception:
            # fallback naive
            hours_old = 9999
        if hours_old < 6:
            reasons.append("very recent")
        elif hours_old < 24:
            reasons.append("recent")
        # Source credibility
        if getattr(candidate.source, "credibility_score", 0.5) > 0.8:
            reasons.append("trusted source")
        # Similarity placeholder (we used NN retrieval)
        reasons.append("similar content")
        return ", ".join(reasons[:3]) if reasons else "relevant"


# Convenience helper for quick demos/portfolio snippets

def get_recommendations_for_article_id(db: ArticleDB, embeddings: EmbeddingSystem, article_id: str, top_k: int = 10) -> List[Tuple[Article, float]]:
    """Fetch an article by id and return similar articles with scores."""
    article = db.get_article_by_id(article_id)
    if not article:
        return []
    recommender = AIRecommender(db, embeddings, RecommendationConfig(top_k=top_k))
    return recommender.recommend_for_article(article)