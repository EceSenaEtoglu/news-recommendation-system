"""
Central configuration for retrieval and storage to avoid circular imports.

These values are the single source of truth used by both retrieval and storage.
"""

from dataclasses import dataclass

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


