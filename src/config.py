"""
Central configuration for all modules to avoid circular imports and scattered configs.
"""

from dataclasses import dataclass
from typing import Optional

# Learning rates for prefs updates
ENTITY_LR: float = 1.0
TOPIC_LR: float = 0.5

# Freshness decay constant (days) for recency weighting
FRESHNESS_DECAY_EXP_CONSTANT_DAYS: float = 4.0

# Limits used when truncating extracted entities/topics per article
MAX_ENTITIES_PER_ARTICLE: int = 10
MAX_TOPICS_PER_ARTICLE: int = 10


@dataclass
class RAGConfig:
    """Configuration for different RAG strategies"""

    # Database retrieval constants
    entity_lr: float = ENTITY_LR
    topic_lr: float = TOPIC_LR
    freshness_decay_exp_constant: float = FRESHNESS_DECAY_EXP_CONSTANT_DAYS

    # Hybrid weights
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6

    # Freshness parameters
    freshness_decay_hours: float = 24.0
    freshness_weight: float = 0.3

    # Personalization weights and penalties
    user_profile_weight: float = 0.2
    personalization_weight: float = 0.10
    topic_string_weight: float = 0.05
    content_type_bonus: float = 0.10
    # audience removed
    per_source_repeat_penalty: float = 0.02
    # Personalization clamp
    personalization_cap: tuple[float, float] = (-0.5, 0.8)

    article_entity_score_clamp: tuple[float, float] = (-2.0, 2.0)

    # Diversity/Balance
    min_sources: int = 2
    opposing_view_boost: float = 0.1

    # Self-RAG (Cross-encoder)
    enable_cross_encoder: bool = True
    cross_encoder_weight: float = 0.3

    # Graph RAG
    enable_graph_rag: bool = True
    entity_boost_weight: float = 0.15

    # CONFIGS FOR NEWS CLASSIFICATION
    breaking_news_urgency_coeff: float = 0.7
    breaking_news_freshness_coeff: float = 0.3

    background_content_length_threshold: int = 2000
    background_analysis_boost: float = 1.2
    background_feature_boost: float = 1.0

    # Content quality weights
    content_length_weight: float = 0.4
    source_credibility_weight: float = 0.3
    content_complexity_weight: float = 0.3

    max_entities_per_article: int = MAX_ENTITIES_PER_ARTICLE
    max_topics_per_article: int = MAX_TOPICS_PER_ARTICLE

    # MMR diversification for search results
    use_mmr_in_search: bool = False
    mmr_lambda: float = 0.7
    mmr_pool: int = 50


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    name: str
    model_path: str
    dimension: int
    description: str
    is_news_specific: bool = False


@dataclass
class RerankFeatureConfig:
    """Configuration for reranking feature extraction"""
    recency_half_life_hours: float = 48.0
    content_length_norm: int = 2000


@dataclass
class NeuralRerankerConfig:
    """Configuration for neural reranker (for future use)"""
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
    
    # Neural reranker settings (for future use - requires user interaction data)
    use_neural_reranker: bool = False
    neural_config: Optional[NeuralRerankerConfig] = None
    
    def __post_init__(self):
        if self.neural_config is None:
            self.neural_config = NeuralRerankerConfig()


