"""
Scoring and personalization utilities for news recommendations.
"""

import math
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional
from src.data_models import Article, UserProfile
from src.config import RAGConfig


class ScoringEngine:
    """Handles various scoring mechanisms for news recommendations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def apply_user_preferences(self, results: List[Tuple[str, float]], 
                             user_profile: UserProfile,
                             articles_dict: Dict[str, Article],
                             user_prefs: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Apply bounded personalization on top of base relevance/freshness scores. 
        Personalization is clamped so input score dominates.

        INPUTS
        -results: List of (article_id, base_score) tuples.
        -user_profile: User profile with preferences
        -articles_dict: Dictionary mapping article_id to Article objects
        -user_prefs: Learned preferences from database (ent:*, topic:*)

        LOGIC
        1) sum learned entity weights if learned, clamp to [-2, 2],
        then scale by personalization_weight.
        2) sum topic weights
        - If learned: add learned topic weights.
        - Else: add +1.0 per default user preferred topic match.
        3) add fixed boost if content_type matches.
        4) Diversity: small penalty if the same source repeats.
        5) Combine all deltas, clamp total personalization
        (default cap = [-0.5, 0.8]).
        Final score = base_score + clamped_delta - diversity_penalty.

        returns sorted list of (article_id, score).
        """
        if not results or not user_profile:
            return results

        # Parse learned preferences
        user_ent_weights = {k.split(":", 1)[1]: v for k, v in user_prefs.items() if k.startswith("ent:")}
        user_topic_weights = {k.split(":", 1)[1]: v for k, v in user_prefs.items() if k.startswith("topic:")}

        # Weights (safe defaults if missing from config)
        entity_weight_constant = getattr(self.config, "personalization_weight", 0.10)
        topic_weight_constant = getattr(self.config, "topic_string_weight", 0.05)
        bonus_content_type = getattr(self.config, "content_type_bonus", 0.10)
        repeat_src_penalty = getattr(self.config, "per_source_repeat_penalty", 0.02)

        scored: List[Tuple[str, float]] = []
        seen_sources: set[str] = set()

        for article_id, base_score in results:
            a = articles_dict.get(article_id)
            if not a:
                continue

            # 1) Learned entities weight (primary signal)
            ents = [e.lower() for e in (getattr(a, "entities", None) or [])][:self.config.max_entities_per_article]
            ent_delta = sum(user_ent_weights.get(e, 0.0) for e in ents)
            
            article_entity_score_clamp = getattr(self.config, "article_entity_score_clamp")
            if ent_delta > article_entity_score_clamp[1]:
                ent_delta = article_entity_score_clamp[1]
            elif ent_delta < article_entity_score_clamp[0]:
                ent_delta = article_entity_score_clamp[0]

            # 2) Topic weight (fallback to default preferred topics if not learned any)
            topic_delta = 0.0
            search_text = f"{a.title}".lower()
            if a.description:
                search_text += a.description.lower()

            if user_topic_weights:
                # boost if learned topic:* weights
                for t, w in user_topic_weights.items():
                    if t and t in search_text:
                        topic_delta += w
            else:
                # cold-start: use user's default preferred topics
                for t in (user_profile.preferred_topics):
                    t = (t or "").lower().strip()
                    if t and t in search_text:
                        topic_delta += 1.0

            # 3) Simple categorical bonuses
            ctype_b = bonus_content_type if a.content_type in (user_profile.preferred_content_types or []) else 0.0

            # 4) Source diversity nudge (tiny penalty for repeats in the list)
            src_name = getattr(getattr(a, "source", None), "name", None)
            diversity_pen = repeat_src_penalty if src_name and src_name in seen_sources else 0.0

            # 5) Final personalized score
            raw_delta = (
                entity_weight_constant * ent_delta      # learned per-entity prefs (already ±2 clamped)
                + topic_weight_constant * topic_delta # topic string matches
                + ctype_b                 # categorical bonus
            )

            # clamp the total personalization bump so that base_score is the dominant score
            low, high = getattr(self.config, "personalization_cap", (-0.5, 0.8))
            bump = float(np.clip(raw_delta, low, high))

            # apply tiny diversity penalty after clamp
            score = base_score + bump - diversity_pen

            scored.append((article_id, score))
            if src_name:
                seen_sources.add(src_name)

        return sorted(scored, key=lambda x: x[1], reverse=True)
    
