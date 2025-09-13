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
from .reranker import RerankFeatureExtractor, TrainableLogisticReranker, RerankFeatureConfig
from datetime import datetime, timezone
import numpy as np


@dataclass
class NeuralRerankerConfig:
    """Configuration for neural reranker"""
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class RecommendationConfig:
    """Lightweight config for the AI recommender."""
    top_k: int = 10
    min_score: float = 0.0  # cosine similarity threshold in FAISS (0..1 after normalize)
    topic_overlap_boost: float = 0.1  # additional boost per topic overlap (small)
    max_topic_bonus: float = 0.5      # cap total bonus from topics

    # MMR diversification
    use_mmr: bool = False
    mmr_lambda: float = 0.7  # trade-off: 1.0 = all relevance, 0.0 = all diversity
    mmr_pool: int = 50       # candidate pool size before MMR selection
    
    # Neural reranker settings
    use_neural_reranker: bool = False
    neural_config: NeuralRerankerConfig = None
    
    def __post_init__(self):
        if self.neural_config is None:
            self.neural_config = NeuralRerankerConfig()


class NeuralReranker(nn.Module):
    """
    Neural network reranker for news recommendations.
    
    Architecture:
    - Input: Feature vector (semantic similarity, topic overlap, freshness, etc.)
    - Hidden layers: Multi-layer perceptron with dropout
    - Output: Relevance score (0-1)
    """
    
    def __init__(self, input_dim: int, config: NeuralRerankerConfig):
        super().__init__()
        self.config = config
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = config.hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """Predict relevance scores for given features"""
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features)
            scores = self.forward(X).squeeze().numpy()
        return scores


class AdvancedFeatureExtractor:
    """
    Advanced feature extraction for neural reranker.
    Extracts richer features than the basic logistic reranker.
    """
    
    def __init__(self, db: ArticleDB, embeddings: EmbeddingSystem):
        self.db = db
        self.embeddings = embeddings
        
    def extract_features(self, 
                        seed_article: Article, 
                        candidate_articles: List[Article],
                        base_scores: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive features for neural reranking.
        
        Features:
        1. Semantic similarity (cosine)
        2. Topic overlap ratio
        3. Entity overlap ratio
        4. Freshness score
        5. Source credibility
        6. Content type match
        7. Title similarity
        8. Content length ratio
        9. Author match
        10. Publication time difference
        """
        
        features = []
        article_ids = []
        
        for candidate in candidate_articles:
            feature_vector = []
            
            # 1. Base semantic similarity
            base_score = base_scores.get(candidate.id, 0.0)
            feature_vector.append(base_score)
            
            # 2. Topic overlap ratio
            seed_topics = set(seed_article.topics or [])
            cand_topics = set(candidate.topics or [])
            topic_overlap = len(seed_topics.intersection(cand_topics)) / max(len(seed_topics), 1)
            feature_vector.append(topic_overlap)
            
            # 3. Entity overlap ratio
            seed_entities = set(seed_article.entities or [])
            cand_entities = set(candidate.entities or [])
            entity_overlap = len(seed_entities.intersection(cand_entities)) / max(len(seed_entities), 1)
            feature_vector.append(entity_overlap)
            
            # 4. Freshness score (0-1, newer = higher)
            now = datetime.now(timezone.utc)
            time_diff = (now - candidate.published_at).total_seconds()
            freshness = max(0, 1 - (time_diff / (7 * 24 * 3600)))  # 7 days decay
            feature_vector.append(freshness)
            
            # 5. Source credibility
            feature_vector.append(candidate.source.credibility_score)
            
            # 6. Content type match (binary)
            content_type_match = 1 if seed_article.content_type == candidate.content_type else 0
            feature_vector.append(content_type_match)
            
            # 7. Title similarity (simple word overlap)
            seed_title_words = set(seed_article.title.lower().split())
            cand_title_words = set(candidate.title.lower().split())
            title_overlap = len(seed_title_words.intersection(cand_title_words)) / max(len(seed_title_words), 1)
            feature_vector.append(title_overlap)
            
            # 8. Content length ratio
            length_ratio = min(len(candidate.content), len(seed_article.content)) / max(len(candidate.content), len(seed_article.content), 1)
            feature_vector.append(length_ratio)
            
            # 9. Author match (binary)
            author_match = 1 if (seed_article.author and candidate.author and 
                               seed_article.author.lower() == candidate.author.lower()) else 0
            feature_vector.append(author_match)
            
            # 10. Publication time difference (normalized)
            time_diff_hours = abs((seed_article.published_at - candidate.published_at).total_seconds() / 3600)
            time_diff_norm = min(time_diff_hours / (24 * 7), 1.0)  # Normalize to 7 days
            feature_vector.append(time_diff_norm)
            
            features.append(feature_vector)
            article_ids.append(candidate.id)
        
        return np.array(features), article_ids


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
        
        # Neural reranker components
        self._neural_reranker: Optional[NeuralReranker] = None
        self._advanced_feature_extractor: Optional[AdvancedFeatureExtractor] = None
        
        if self.config.use_neural_reranker:
            self._init_neural_reranker()

    def _init_neural_reranker(self):
        """Initialize neural reranker with appropriate input dimension"""
        # Input dimension based on advanced feature extractor
        input_dim = 10  # Number of features extracted by AdvancedFeatureExtractor
        self._neural_reranker = NeuralReranker(input_dim, self.config.neural_config)
        self._advanced_feature_extractor = AdvancedFeatureExtractor(self.db, self.embeddings)
        print(f"Initialized neural reranker with {input_dim} input features")
    
    def train_neural_reranker(self, user_id: str = "default", days: int = 14):
        """Train the neural reranker on user interaction data"""
        if not self._neural_reranker:
            print("Neural reranker not initialized")
            return
        
        print("Building training data for neural reranker...")
        
        # Build training data (synthetic for now)
        X, y = self._build_neural_training_data(user_id, days)
        
        if X.size == 0:
            print("No training data available")
            return
        
        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self._train_neural_model(X_train, y_train, X_val, y_val)
        
        print("Neural reranker training completed")
    
    def _build_neural_training_data(self, user_id: str, days: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build training data for neural reranker from articles"""
        # Get recent articles
        articles = self.db.get_recent_articles(limit=100, hours_back=days * 24)
        
        if len(articles) < 10:
            return np.array([]), np.array([])
        
        features = []
        labels = []
        
        # Create synthetic training data
        for i, seed_article in enumerate(articles[:20]):  # Use first 20 as seeds
            # Get similar articles
            seed_text = self._article_to_query_text(seed_article)
            similar_articles = self.embeddings.semantic_search(seed_text, k=10, score_threshold=0.3)
            
            if not similar_articles:
                continue
                
            # Get candidate articles
            candidate_ids = [aid for aid, _ in similar_articles if aid != seed_article.id]
            candidates = self.db.get_articles_by_ids(candidate_ids)
            
            if not candidates:
                continue
            
            # Extract features
            base_scores = {aid: score for aid, score in similar_articles if aid != seed_article.id}
            X, _ = self._advanced_feature_extractor.extract_features(seed_article, candidates, base_scores)
            
            if X.size == 0:
                continue
            
            # Create synthetic labels based on similarity and freshness
            y = []
            for j, candidate in enumerate(candidates):
                # Higher score for more similar and fresher articles
                similarity_score = base_scores.get(candidate.id, 0.0)
                freshness = max(0, 1 - (datetime.now(timezone.utc) - candidate.published_at).total_seconds() / (7 * 24 * 3600))
                
                # Synthetic label: 1 if high similarity and freshness, 0 otherwise
                label = 1 if (similarity_score > 0.7 and freshness > 0.5) else 0
                y.append(label)
            
            features.append(X)
            labels.extend(y)
        
        if not features:
            return np.array([]), np.array([])
        
        X = np.vstack(features)
        y = np.array(labels)
        
        return X, y
    
    def _train_neural_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the neural reranker model"""
        optimizer = torch.optim.Adam(self._neural_reranker.parameters(), lr=self.config.neural_config.learning_rate)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.neural_config.num_epochs):
            # Training
            self._neural_reranker.train()
            train_loss = 0.0
            train_correct = 0
            
            # Mini-batch training
            for i in range(0, len(X_tensor), self.config.neural_config.batch_size):
                batch_X = X_tensor[i:i + self.config.neural_config.batch_size]
                batch_y = y_tensor[i:i + self.config.neural_config.batch_size]
                
                optimizer.zero_grad()
                outputs = self._neural_reranker(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += ((outputs > 0.5) == batch_y).sum().item()
            
            avg_train_loss = train_loss / (len(X_tensor) // self.config.neural_config.batch_size + 1)
            train_acc = train_correct / len(X_tensor)
            
            # Validation
            if X_val is not None and y_val is not None:
                self._neural_reranker.eval()
                with torch.no_grad():
                    val_outputs = self._neural_reranker(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_correct = ((val_outputs > 0.5) == y_val_tensor).sum().item()
                    val_acc = val_correct / len(X_val_tensor)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.neural_config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
                if X_val is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def _apply_neural_reranking(self, 
                               current: Article, 
                               ranked: List[Tuple[Article, float]], 
                               base_scores: Dict[str, float]) -> List[Tuple[Article, float]]:
        """Apply neural reranking to results"""
        if not self._neural_reranker or not self._advanced_feature_extractor:
            return ranked
        
        articles = [a for a, _ in ranked]
        
        # Extract features
        X, ids = self._advanced_feature_extractor.extract_features(current, articles, base_scores)
        
        if X.size == 0:
            return ranked
        
        # Predict scores
        neural_scores = self._neural_reranker.predict_scores(X)
        id_to_neural_score = {aid: float(score) for aid, score in zip(ids, neural_scores)}
        
        # Combine base scores with neural scores
        combined_ranked = []
        for article, base_score in ranked:
            neural_score = id_to_neural_score.get(article.id, base_score)
            # Weighted combination: 70% neural, 30% base
            combined_score = 0.7 * neural_score + 0.3 * base_score
            combined_ranked.append((article, combined_score))
        
        # Sort by combined score
        combined_ranked.sort(key=lambda x: x[1], reverse=True)
        
        return combined_ranked

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
            # --- MMR Diversification ---
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
                        mmr_val = lambda_ * rel - (1.0 - lambda_) * max_sim
                    if mmr_val > best_val:
                        best_val = mmr_val
                        best_idx = idx
                selected.append(best_idx)
                remaining.remove(best_idx)
            ranked = [(id_to_cand[pool_ids[i]], id_to_score[pool_ids[i]]) for i in selected if pool_ids[i] in id_to_cand]
        # Optional neural reranking
        if self._neural_reranker and ranked:
            ranked = self._apply_neural_reranking(current, ranked, base_scores)
        
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

        # Ensure the final list is presented in descending score order for UX
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