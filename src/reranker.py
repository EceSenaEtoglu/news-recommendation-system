from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.data_models import Article
from src.storage import ArticleDB
from src.embeddings import EmbeddingSystem
from src.config import RerankFeatureConfig, NeuralRerankerConfig, RAGConfig


class RerankFeatureExtractor:
    """Compute simple, interpretable features for reranking."""

    def __init__(self, db: ArticleDB, embeddings: EmbeddingSystem, cfg: Optional[RerankFeatureConfig] = None):
        self.db = db
        self.embeddings = embeddings
        self.cfg = cfg or RerankFeatureConfig()

    def build_features(
        self,
        seed: Article,
        candidates: List[Article],
        base_scores: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """Return (X, ids) where X is [n, d] feature matrix and ids correspond to candidates order.
        Features:
          - base_sim: base similarity score from FAISS
          - topic_overlap: count of overlapping topics
          - recency: exp(-hours_old / half_life)
          - source_cred: source credibility (0..1)
          - content_len_norm: min(1, len(content)/content_length_norm)
        """
        seed_topics = set(self._flatten_topics(seed.topics or []))
        now = datetime.now(timezone.utc)
        X = []
        ids: List[str] = []
        for a in candidates:
            base = float(base_scores.get(a.id, 0.0))
            overlap = 0
            if seed_topics and a.topics:
                overlap = len(seed_topics.intersection(set(self._flatten_topics(a.topics or []))))

            hours_old = max(0.0, (now - a.published_at).total_seconds() / 3600.0)
            recency = math.exp(-hours_old / max(1e-6, self.cfg.recency_half_life_hours))

            src_cred = float(getattr(getattr(a, "source", None), "credibility_score", 0.5) or 0.5)
            clen = len(a.content or "")
            clen_norm = min(1.0, clen / float(self.cfg.content_length_norm))

            X.append([base, overlap, recency, src_cred, clen_norm])
            ids.append(a.id)

        return np.asarray(X, dtype=float), ids


@dataclass
class LogisticParams:
    lr: float = 0.1
    epochs: int = 300
    l2: float = 0.0
    fit_intercept: bool = True


class TrainableLogisticReranker:
    """Simple logistic regression reranker (numpy), no external deps.

     Predicts p(click|features). Use as a rerank score.
    """

    def __init__(self, params: Optional[LogisticParams] = None):
        self.params = params or LogisticParams()
        self.w: Optional[np.ndarray] = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        if self.params.fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=float)
            return np.hstack([ones, X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xb = self._prep(X)
        y = y.reshape(-1, 1).astype(float)
        n, d = Xb.shape
        self.w = np.zeros((d, 1), dtype=float)
        lr = self.params.lr
        l2 = self.params.l2
        for _ in range(self.params.epochs):
            z = Xb @ self.w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xb.T @ (p - y) / n
            if l2 > 0:
                grad += l2 * self.w
            self.w -= lr * grad

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        Xb = self._prep(X)
        z = Xb @ self.w
        p = 1.0 / (1.0 + np.exp(-z))
        return p.reshape(-1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)


class NeuralReranker(nn.Module):
    """
    Neural network reranker for news recommendations (for future use).
    
    Architecture:
    - Input: Feature vector (semantic similarity, topic overlap, freshness, etc.)
    - Hidden layers: Multi-layer perceptron with dropout
    - Output: Relevance score (0-1)
    
    Note: Currently not used as it requires user interaction data for training.
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
    Advanced feature extraction for neural reranker (for future use).
    Extracts richer features than the basic logistic reranker.
    
    Note: Currently not used as neural reranker requires user interaction data.
    """
    
    def __init__(self, db: ArticleDB, embeddings: EmbeddingSystem):
        self.db = db
        self.embeddings = embeddings
        
    def extract_features(self, 
                        seed_article: Article, 
                        candidate_articles: List[Article],
                        base_scores: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract advanced features for neural reranker.
        
        Features (10 total):
        1. Semantic similarity (FAISS score)
        2. Topic overlap count
        3. Recency score (exponential decay)
        4. Source credibility
        5. Content length (normalized)
        6. Entity overlap count
        7. Content quality score
        8. Temporal relevance
        9. User engagement score
        10. Cross-modal alignment
        """
        features = []
        article_ids = []
        
        for article in candidate_articles:
            # Feature 1: Base semantic similarity
            semantic_sim = base_scores.get(article.id, 0.0)
            
            # Feature 2: Topic overlap
            topic_overlap = len(set(self._flatten_topics(seed_article.topics or [])).intersection(set(self._flatten_topics(article.topics or []))))
            
            # Feature 3: Recency (exponential decay)
            hours_old = max(0.0, (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600.0)
            recency = math.exp(-hours_old / 48.0)  # 48-hour half-life
            
            # Feature 4: Source credibility
            source_cred = getattr(getattr(article, "source", None), "credibility_score", 0.5) or 0.5
            
            # Feature 5: Content length (normalized)
            content_len = min(1.0, len(article.content or "") / 2000.0)
            
            # Feature 6: Entity overlap
            seed_entities = set(seed_article.entities or [])
            article_entities = set(article.entities or [])
            entity_overlap = len(seed_entities.intersection(article_entities))
            
            # Feature 7: Content quality (simple heuristic)
            content_quality = min(1.0, len(article.description or "") / 200.0)
            
            # Feature 8: Temporal relevance (time-based)
            temporal_relevance = 1.0 if article.content_type == "breaking" else 0.5
            
            # Feature 9: User engagement (placeholder - would use real data)
            user_engagement = 0.5  # Default neutral score
            
            # Feature 10: Cross-modal alignment (title-content similarity)
            cross_modal = 0.5  # Placeholder - would compute title-content similarity
            
            feature_vector = [
                semantic_sim, topic_overlap, recency, source_cred, content_len,
                entity_overlap, content_quality, temporal_relevance, user_engagement, cross_modal
            ]
            
            features.append(feature_vector)
            article_ids.append(article.id)
        
        return np.array(features), article_ids


class NeuralRerankerManager:
    """
    Manager class for neural reranker operations (for future use).
    Handles training, data preparation, and reranking logic.
    """
    
    def __init__(self, neural_reranker: NeuralReranker, advanced_feature_extractor: AdvancedFeatureExtractor, 
                 config: NeuralRerankerConfig, db: ArticleDB, embeddings: EmbeddingSystem):
        self.neural_reranker = neural_reranker
        self.advanced_feature_extractor = advanced_feature_extractor
        self.config = config
        self.db = db
        self.embeddings = embeddings
    
    def train_neural_reranker(self, user_id: str = "default", days: int = 14):
        """Train the neural reranker on user interaction data (for future use)"""
        if not self.neural_reranker:
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
            X, _ = self.advanced_feature_extractor.extract_features(seed_article, candidates, base_scores)
            
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
        optimizer = torch.optim.Adam(self.neural_reranker.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.neural_reranker.train()
            train_loss = 0.0
            train_correct = 0
            
            # Mini-batch training
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch_X = X_tensor[i:i + self.config.batch_size]
                batch_y = y_tensor[i:i + self.config.batch_size]
                
                optimizer.zero_grad()
                outputs = self.neural_reranker(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += ((outputs > 0.5) == batch_y).sum().item()
            
            avg_train_loss = train_loss / (len(X_tensor) // self.config.batch_size + 1)
            train_acc = train_correct / len(X_tensor)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.neural_reranker.eval()
                with torch.no_grad():
                    val_outputs = self.neural_reranker(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_correct = ((val_outputs > 0.5) == y_val_tensor).sum().item()
                    val_acc = val_correct / len(X_val_tensor)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
                if X_val is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def apply_neural_reranking(self, 
                               current: Article, 
                               ranked: List[Tuple[Article, float]], 
                               base_scores: Dict[str, float]) -> List[Tuple[Article, float]]:
        """Apply neural reranking to results"""
        if not self.neural_reranker or not self.advanced_feature_extractor:
            return ranked
        
        articles = [a for a, _ in ranked]
        
        # Extract features
        X, ids = self.advanced_feature_extractor.extract_features(current, articles, base_scores)
        
        if X.size == 0:
            return ranked
        
        # Predict scores
        neural_scores = self.neural_reranker.predict_scores(X)
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
        """Convert article to query text for semantic search"""
        parts = []
        if article.title:
            parts.append(article.title)
        if article.description:
            parts.append(article.description)
        if article.content:
            # Take first 500 characters of content
            content_preview = article.content[:500]
            parts.append(content_preview)
        
        return " ".join(parts)


class RerankingEngine:
    """Handles reranking operations for news recommendations"""
    
    def __init__(self, config: RAGConfig, embeddings: EmbeddingSystem):
        self.config = config
        self.embeddings = embeddings
        self.cross_encoder = None
        if config.enable_cross_encoder:
            self._init_cross_encoder()
    
    def _init_cross_encoder(self):
        """Initialize cross-encoder for reranking (lazy loading)"""
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
            print(" Loaded cross-encoder for reranking")
        except Exception as e:
            print(f" Failed to load cross-encoder: {e}")
            self.config.enable_cross_encoder = False
    
    def apply_mmr_diversification(self, results: List[Tuple[str, float]], 
                                top_k: int, articles_dict: Dict[str, Article]) -> List[Tuple[str, float]]:
        """Apply MMR diversification on (article_id, score) list"""
        if not results or not getattr(self.config, "use_mmr_in_search", False):
            return results

        pool_n = max(top_k, int(getattr(self.config, "mmr_pool", 50)))
        lambda_ = float(getattr(self.config, "mmr_lambda", 0.7))

        # Truncate pool and fetch article texts
        pool = sorted(results, key=lambda x: x[1], reverse=True)[:pool_n]
        pool_ids = [aid for aid, _ in pool]

        texts = []
        kept = []
        for i, aid in enumerate(pool_ids):
            a = articles_dict.get(aid)
            if not a:
                continue
            desc = a.description or ""
            head = (a.content or "")[:500]
            texts.append(f"{a.title} {desc} {head}")
            kept.append(i)

        if len(texts) < 2:
            return pool

        # Encode and normalize for cosine
        X = self.embeddings.encode_texts(texts).astype('float32', copy=False)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = X @ X.T
        np.fill_diagonal(S, 1.0)

        id_to_score = {aid: s for aid, s in pool}
        selected_local: List[int] = []
        remaining_local: List[int] = list(range(len(kept)))

        while remaining_local and len(selected_local) < top_k:
            best_j = None
            best_val = -1e9
            for j in remaining_local:
                aid = pool_ids[kept[j]]
                rel = float(id_to_score.get(aid, 0.0))
                if not selected_local:
                    val = rel
                else:
                    max_sim = max(S[j, t] for t in selected_local)
                    val = lambda_ * rel - (1.0 - lambda_) * max_sim
                if val > best_val:
                    best_val = val
                    best_j = j
            selected_local.append(best_j)
            remaining_local.remove(best_j)

        # Build final sequence
        final_pairs: List[Tuple[str, float]] = []
        for j in selected_local:
            aid = pool_ids[kept[j]]
            final_pairs.append((aid, id_to_score.get(aid, 0.0)))

        return final_pairs
    
    async def apply_cross_encoder_reranking(self,query: str,candidate_ids: List[str],articles_dict: Dict[str, "Article"],limit: int,) -> List[Tuple[str, float]]:
        """
        - candidate_ids: IDs to score (already capped upstream to CE_K).
        - articles_dict: id -> Article (pre-fetched).
        - limit: safety cap (usually == CE_K).
        Returns:
        - [(article_id, ce_logit)] sorted desc by ce_logit.
        """
        if not self.cross_encoder or not candidate_ids:
            return [(cid, 0.0) for cid in candidate_ids]

        # Slice to limit (top-CE_K) BEFORE building pairs 
        candidate_ids = candidate_ids[:limit]

        pairs, kept_ids = [], []
        for article_id in candidate_ids:
            article = articles_dict.get(article_id)
            if not article:
                continue
            
            #Build text for CE 
            title = article.title or ""
            description = article.description or ""
            body = article.content or ""

            # Combine
            article_text = f"{title}. {description} {body}".strip()
            
            if not article_text:
                continue
            pairs.append([query, article_text])
            kept_ids.append(article_id)

        if not pairs:
            return []

        # Get cross-encoder logits
        ce_logits = self.cross_encoder.predict(pairs)

        # Map ids -> ce score
        ce_articles = [(kept_ids[i], float(ce_logits[i])) for i in range(len(ce_logits))]

        # Sort by CE score (descending) — 
        ce_articles.sort(key=lambda x: x[1], reverse=True)

        # Debug/log
        print(f"Cross-encoder reranked {len(ce_articles)} results (limit={limit})")

        return ce_articles
    
    def _flatten_topics(self, topics) -> List[str]:
        """Safely flatten topics list, handling nested lists."""
        if not topics:
            return []
        
        flattened = []
        for topic in topics:
            if isinstance(topic, str):
                flattened.append(topic)
            elif isinstance(topic, list):
                # Recursively flatten nested lists
                flattened.extend(self._flatten_topics(topic))
            else:
                # Convert other types to string
                flattened.append(str(topic))
        
        return flattened

            

def build_training_data_from_events(
    db: ArticleDB,
    embeddings: EmbeddingSystem,
    feature_extractor: RerankFeatureExtractor,
    user_id: str,
    days: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) for reranker training using recent user events.
    Labels: save/read=1, skip=0. Uses article pairs implicitly per event against seed (same article as seed context).
    For simplicity, treat each event's article as candidate to its own seed; in practice you would pair with top-N neighbors.
    """
    events = db.get_recent_events(user_id=user_id, days=days)
    if not events:
        return np.zeros((0, 5)), np.zeros((0,))

    # Build articles map
    article_ids = list({e["article_id"] for e in events})
    arts = db.get_articles_by_ids(article_ids)
    by_id = {a.id: a for a in arts}

    Xs: List[np.ndarray] = []
    ys: List[float] = []

    for ev in events:
        aid = ev["article_id"]
        a = by_id.get(aid)
        if not a:
            continue
        # Use the same article as seed context (approximation for weak supervision)
        seed = a
        candidates = [a]
        base_scores = {a.id: 1.0}
        X, _ = feature_extractor.build_features(seed, candidates, base_scores)
        if X.size == 0:
            continue
        label = 1.0 if ev["event_type"] in ("save", "read") else 0.0
        Xs.append(X[0:1])
        ys.append(label)

    if not Xs:
        return np.zeros((0, 5)), np.zeros((0,))

    X = np.vstack(Xs)
    y = np.asarray(ys, dtype=float)
    return X, y


