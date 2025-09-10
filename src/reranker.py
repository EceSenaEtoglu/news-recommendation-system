from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone
import math
import numpy as np

from .data_models import Article
from .storage import ArticleDB
from .embeddings import EmbeddingSystem


@dataclass
class RerankFeatureConfig:
    recency_half_life_hours: float = 48.0
    content_length_norm: int = 2000


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
        seed_topics = set(seed.topics or [])
        now = datetime.now(timezone.utc)
        X = []
        ids: List[str] = []
        for a in candidates:
            base = float(base_scores.get(a.id, 0.0))
            overlap = 0
            if seed_topics and a.topics:
                overlap = len(seed_topics.intersection(set(a.topics)))

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

    y in {0,1}. Predicts p(click|features). Use as a rerank score.
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


