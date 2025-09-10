import math
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

from .data_models import Article, UserProfile, SearchQuery, SearchResult, ContentType
from .storage import ArticleDB
from .embeddings import EmbeddingSystem
import spacy
_NER = spacy.load("en_core_web_sm")

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np


@dataclass
class RAGConfig:
    """Configuration for different RAG strategies"""
    
    # Database retrieval constants
    entity_lr = 1.0
    topic_lr  = 0.5
    freshness_decay_exp_constant = 4

    # Hybrid weights
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    
    # Freshness parameters
    freshness_decay_hours: float = 24.0
    freshness_weight: float = 0.3
    
    # Personalization weights and penalties
    user_profile_weight: float = 0.2
    personalization_weight: float = 0.10
    topic_string_weight : float = 0.05
    content_type_bonus : float = 0.10
    # audience removed
    per_source_repeat_penalty = 0.02
    # Personlization clamp
    personalization_cap = (-0.5, 0.8)
    
    article_entity_score_clamp = (-2.0,2.0)

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
    breaking_news_urgency_coeff= 0.7
    breaking_news_freshness_coeff = 0.3
    
    background_content_length_threshold: int = 2000
    background_analysis_boost: float = 1.2
    background_feature_boost: float = 1.0

    # Content quality weights
    content_length_weight: float = 0.4
    source_credibility_weight: float = 0.3
    content_complexity_weight: float = 0.3
    
    max_entities_per_article: int = 10
    max_topics_per_article:   int = 10

    # MMR diversification for search results
    use_mmr_in_search: bool = False
    mmr_lambda: float = 0.7
    mmr_pool: int = 50

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
nltk.download("punkt")

_STEM = PorterStemmer()

def _tokenize(text: str) -> List[str]:
    """
    BM25 tokenizer using:
    - NLTK word_tokenize
    - lowercasing
    - gensim STOPWORDS removal
    - Porter stemming
    """
    if not text:
        return []

    # Lowercase and tokenize
    toks = word_tokenize(text.lower())

    # Remove stopwords using gensim's list
    toks = [t for t in toks if t not in GENSIM_STOPWORDS]

    # Stem each token
    toks = [_STEM.stem(t) for t in toks if t.isalnum()]

    return toks

    
class MultiRAGRetriever:
    """Enhanced RAG system with caching, cross-encoder, and graph features"""
    
    def __init__(self, db: ArticleDB, embeddings: EmbeddingSystem, config: RAGConfig = None):
        self.db = db
        self.embeddings = embeddings
        self.config = config or RAGConfig()
        
        # BM25 index with caching
        self.bm25_index = None
        self.bm25_articles = []
        self.bm25_cache_path = "db/bm25_cache.pkl"
        self._build_bm25_index()
        
        # Cross-encoder for reranking (lazy loading)
        self.cross_encoder = None
        if self.config.enable_cross_encoder:
            self._init_cross_encoder()
        
        # Entity cache for Graph RAG
        self.entity_cache = {}
    
    def _build_bm25_index(self):
        """Build BM25 index with caching support"""
        cache_valid = self._is_bm25_cache_valid()
        
        if cache_valid:
            print(" Loading BM25 index from cache...")
            self._load_bm25_cache()
        else:
            print(" Rebuilding BM25 index...")
            self._rebuild_bm25_index()
            self._save_bm25_cache()
    
    def _is_bm25_cache_valid(self) -> bool:
        """Check if BM25 cache is fresh (less than 6 hours old)"""
        if not os.path.exists(self.bm25_cache_path):
            return False
        
        cache_age = time.time() - os.path.getmtime(self.bm25_cache_path)
        return cache_age < 6 * 3600  # 6 hours
    
    def _load_bm25_cache(self):
        """Load BM25 index from cache"""
        try:
            with open(self.bm25_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.bm25_index = cache_data['index']
                self.bm25_articles = cache_data['articles']
            print(f" Loaded BM25 cache with {len(self.bm25_articles)} articles")
        except Exception as e:
            print(f" Failed to load BM25 cache: {e}")
            self._rebuild_bm25_index()
            
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from database"""
        articles = self.db.get_recent_articles(limit=1000, hours_back=24*7)
        
        if articles:
            corpus = []
            self.bm25_articles = []
            
            for article in articles:
                text = f"{article.title} {article.description or ''} {article.content[:500]}"
                doc_tokens = _tokenize(text)
                corpus.append(doc_tokens)
                self.bm25_articles.append(article)
            
            self.bm25_index = BM25Okapi(corpus)
            print(f"Built BM25 index with {len(articles)} articles")
        else:
            print("No articles found for BM25 index")
    
    def _save_bm25_cache(self):
        """Save BM25 index to cache"""
        try:
            os.makedirs(os.path.dirname(self.bm25_cache_path), exist_ok=True)
            cache_data = {
                'index': self.bm25_index,
                'articles': self.bm25_articles,
                'created_at': datetime.now(datetime.timezone.utc).isoformat()
            }
            with open(self.bm25_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print("Saved BM25 cache")
        except Exception as e:
            print(f" Failed to save BM25 cache: {e}")
    
    def _init_cross_encoder(self):
        """Initialize cross-encoder for reranking (lazy loading)"""
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
            print(" Loaded cross-encoder for reranking")

        except Exception as e:
            print(f" Failed to load cross-encoder: {e}")
            self.config.enable_cross_encoder = False

    async def search(self, query: SearchQuery, user_profile: Optional[UserProfile] = None) -> List[SearchResult]:
        """Main search function with performance monitoring. Uses query routing mechanism"""
        start_time = time.time()
        
        try:
            # Route to appropriate RAG strategy
            
            # Query type refers to different angles or aspects of coverage for the same underlying news event
            # "Breaking news" = immediate, surface-level updates
            #"Background pieces" = explanatory journalism that provides context
            if query.query_type == "breaking":
                results = await self._breaking_news_rag(query, user_profile)
            elif query.query_type == "background": 
                results = await self._background_analysis_rag(query, user_profile)
            else:
                results = await self._hybrid_rag(query, user_profile)
            
            # Performance logging
            search_time = time.time() - start_time
            print(f" Search completed in {search_time:.2f}s ({len(results)} results)")
            
            return results
            
        except Exception as e:
            print(f" Search failed: {e}")
            return []

    async def _hybrid_rag(self, query: SearchQuery, user_profile: Optional[UserProfile]) -> List[SearchResult]:
        """Enhanced Hybrid RAG with cross-encoder and graph features"""
        
        # Step 1: BM25 Search (lexical)
        bm25_results = self._bm25_search(query.text, limit=50)
        
        # Step 2: Semantic Search (dense embeddings)
        # 
        semantic_results = self._semantic_search(query.text, limit=50)
        
        # Step 3: Combine scores based on weight configs in RagConfig (Hybrid RAG)
        # returns (article_id, final_score) sorted based on final_score. final_score is in range [0,1]
        combined_results = self._combine_hybrid_scores(bm25_results, semantic_results)
        
        # Step 4: Graph RAG - Entity-based expansion (optional)
        if self.config.enable_graph_rag:
            expanded_results = await self._apply_graph_expansion(query.text, combined_results)
        else:
            expanded_results = combined_results
        
        # Step 5: Cross-encoder reranking (Self-RAG)
        if self.config.enable_cross_encoder:
            reranked_results = await self._cross_encoder_rerank(query.text, expanded_results[:20])
        else:
            reranked_results = expanded_results
        
        # Step 6: Apply freshness weighting (Freshness-aware RAG)
        fresh_results = await self._apply_freshness_boost(reranked_results)
        
        # Step 7: Apply user personalization (Memory RAG)  
        if user_profile:
            personalized_results = await self._apply_user_preferences(fresh_results, user_profile)
        else:
            personalized_results = fresh_results
        
        # Step 8: Optional MMR diversification, then ensure source diversity
        diversified_results = self._apply_mmr_if_enabled(personalized_results, top_k=100)
        balanced_results = await self._ensure_balance_and_diversity(diversified_results)
        
        # Step 9: Convert to SearchResult objects
        final_results = self._create_search_results(balanced_results, query, user_profile)
        
        return final_results[:query.limit]

    def _apply_mmr_if_enabled(self, results: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
        """Optionally apply MMR diversification on (article_id, score) list.
        Returns (article_id, score) reranked and truncated to top_k if enabled.
        """
        if not results or not getattr(self.config, "use_mmr_in_search", False):
            return results

        pool_n = max(top_k, int(getattr(self.config, "mmr_pool", 50)))
        lambda_ = float(getattr(self.config, "mmr_lambda", 0.7))

        # Truncate pool and fetch article texts
        pool = sorted(results, key=lambda x: x[1], reverse=True)[:pool_n]
        pool_ids = [aid for aid, _ in pool]
        articles = self.db.get_articles_by_ids(pool_ids)
        id_to_article = {a.id: a for a in articles}

        texts = []
        kept = []
        for i, aid in enumerate(pool_ids):
            a = id_to_article.get(aid)
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

    async def _cross_encoder_rerank(self, query: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Self-RAG: Rerank results using cross-encoder"""
        if not self.cross_encoder or not results:
            return results
        
        try:
            # Get article objects for cross-encoder
            article_ids = [aid for aid, _ in results]
            articles = self.db.get_articles_by_ids(article_ids)
            articles_dict = {a.id: a for a in articles}
            
            # Prepare query-article pairs for cross-encoder
            pairs = []
            valid_results = []
            
            for article_id, original_score in results:
                article = articles_dict.get(article_id)
                if article:
                    # Combine title and description for reranking
                    article_text = f"{article.title} {article.description or ''}"
                    pairs.append([query, article_text])
                    valid_results.append((article_id, original_score))
            
            if not pairs:
                return results
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Combine original scores with cross-encoder scores
            reranked_results = []
            for i, (article_id, original_score) in enumerate(valid_results):
                cross_score = cross_scores[i]
                
                # Weighted combination
                combined_score = (
                    original_score * (1 - self.config.cross_encoder_weight) +
                    cross_score * self.config.cross_encoder_weight
                )
                reranked_results.append((article_id, combined_score))
            
            # Sort by combined score
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            print(f" Cross-encoder reranked {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            print(f" Cross-encoder reranking failed: {e}")
            return results

    async def _apply_graph_expansion(self, query: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not results:
            return results

        # 1) seed entities from query (preferred)
        seed = set(self._ner_entities_from_text(query))

        # 2) if none, bootstrap from entities in top docs
        if not seed:
            top_ids = [aid for aid, _ in results[:5]]
            arts = self.db.get_articles_by_ids(top_ids)
            for a in arts:
                for e in (a.entities or []):
                    seed.add(e)
            seed = set(list(seed)[:5])
        if not seed:
            return results

        # 3) neighbors by co-mention counts (uses new storage helpers)
        neighbors = self.db.get_comention_counts(list(seed), limit=50)
        neighbor_bonus = {name: min(0.3, 0.02*c) for name, c in neighbors.items()}

        # 4) fetch more docs mentioning seeds or strong neighbors
        expand_terms = list(seed) + [n for n, _ in sorted(neighbors.items(), key=lambda x: -x[1])[:10]]
        graph_docs = self.db.get_articles_by_entities(expand_terms, limit=80, hours_back=24*7)

        # 5) score boost: overlap + neighbor strength
        base = {aid: s for aid, s in results}
        from collections import Counter
        scored = Counter(base)

        for a in graph_docs:
            eid_names = set(a.entities or [])
            overlap = len(seed & eid_names)
            if overlap:
                scored[a.id] += self.config.entity_boost_weight * min(0.3, 0.1*overlap)
            nb = sum(neighbor_bonus.get(e, 0.0) for e in eid_names)
            if nb:
                scored[a.id] += self.config.entity_boost_weight * nb
            if a.id not in base and (overlap or nb):
                scored[a.id] += 0.25  # small base for graph-discovered docs

        return sorted(scored.items(), key=lambda x: x[1], reverse=True)


    async def _extract_entities_from_articles(self, article_ids: List[str]) -> List[str]:
        """Extract named entities from articles (simple implementation)"""
        try:
            articles = self.db.get_articles_by_ids(article_ids)
            entities = set()
            
            for article in articles:
                # Simple entity extraction using title words and known patterns
                # In production, you'd use spaCy or similar NER
                text = f"{article.title} {article.description or ''}"
                words = text.split()
                
                for word in words:
                    word = word.strip('.,!?;:"()[]{}')
                    # Simple heuristics for entities
                    if (len(word) > 3 and 
                        (word.istitle() or word.isupper()) and
                        word.isalpha()):
                        entities.add(word.lower())
            
            # Filter common words (basic stop words)
            # TODO the language can be parametrized
            stop_words = set(stopwords.words('english'))
            
            filtered_entities = [e for e in entities if e not in stop_words]
            return filtered_entities[:self.config.max_entities_per_article]  # Limit to top 10 entities
            
        except Exception as e:
            print(f" Entity extraction failed: {e}")
            return []

    async def _find_articles_by_entities(self, entities: List[str]) -> Dict[str, int]:
        """Find articles that mention the given entities"""
        try:
            related_articles = {}
            
            # Search for articles containing these entities
            for entity in entities:
                articles = self.db.search_articles(
                    query=entity,
                    limit=20,
                    hours_back=24*7  # Last week
                )
                
                for article in articles:
                    # Count entity mentions in this article
                    text = f"{article.title} {article.description or ''}"
                    entity_count = text.lower().count(entity.lower())
                    
                    if entity_count > 0:
                        if article.id in related_articles:
                            related_articles[article.id] += entity_count
                        else:
                            related_articles[article.id] = entity_count
            
            return related_articles
            
        except Exception as e:
            print(f" Entity-based article search failed: {e}")
            return {}

    # Fix the remaining database query loops
    async def _apply_freshness_boost(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Apply time decay boost with batch database query"""
        if not results:
            return results
        
        boosted_results = []
        article_ids = [aid for aid, _ in results]
        articles = self.db.get_articles_by_ids(article_ids)
        articles_dict = {a.id: a for a in articles}
        
        for article_id, score in results:
            article = articles_dict.get(article_id)
            if not article:
                continue
            
            hours_old = (datetime.now(datetime.timezone.utc) - article.published_at).total_seconds() / 3600
            freshness_factor = math.exp(-hours_old / self.config.freshness_decay_hours)
            
            boosted_score = score + (freshness_factor * self.config.freshness_weight)
            boosted_results.append((article_id, boosted_score))
        
        return sorted(boosted_results, key=lambda x: x[1], reverse=True)

    async def _apply_user_preferences(
        self, results: List[Tuple[str, float]], user_profile: UserProfile ) -> List[Tuple[str, float]]:
        """
        Apply bounded personalization on top of base relevance/freshness scores. Personalization is clamped
        so input score dominates.

        INPUTS
        -results: List of (article_id, base_score) tuples.
        -user_profile:
            - user_id: to fetch stored prefs from DB (ent:*, topic:*).
            - preferred_topics: keywords used if no learned topic prefs are found.
            - preferred_content_types: adds a fixed bonus when article content_type matches.

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

        # ---- Batch fetch article rows once
        article_ids = [aid for aid, _ in results]
        articles = self.db.get_articles_by_ids(article_ids)
        articles_by_id = {a.id: a for a in articles}

        # ---- Pull learned prefs (entity-centric) for this user
        user_id = getattr(user_profile, "user_id", None) or "default"
        prefs = self.db.get_user_prefs(user_id)  # e.g., {'ent:nvidia': 1.3, 'ent:ecb': -0.6, ...}
        
        # {"nvidia":1.3}
        user_ent_weights = {k.split(":", 1)[1]: v for k, v in prefs.items() if k.startswith("ent:")}
        user_topic_weights = {k.split(":", 1)[1]: v for k, v in prefs.items() if k.startswith("topic:")}

        # ---- Weights (safe defaults if missing from config)
        entity_weight_constant = getattr(self.config, "personalization_weight", 0.10)   # scales learned prefs
        topic_weight_constant = getattr(self.config, "topic_string_weight", 0.05)     # scales title/desc topic hits
        bonus_content_type = getattr(self.config, "content_type_bonus", 0.10)
        # audience removed
        repeat_src_penalty = getattr(self.config, "per_source_repeat_penalty", 0.02)

        scored: List[Tuple[str, float]] = []
        seen_sources: set[str] = set()

        for article_id, base_score in results:
            a = articles_by_id.get(article_id)
            if not a:
                continue

            # 1) Learned entities weight  (primary signal)
            
            # get 10 entities of the article
            ents = [e.lower() for e in (getattr(a, "entities", None) or [])][:self.config.max_entities_per_article]
            
            # if they are in user prefs, get their weight sum
            ent_delta = sum(user_ent_weights.get(e, 0.0) for e in ents)
            
            article_entity_score_clamp = getattr(self.config,"article_entity_score_clamp")

            # per-article clamp
            if ent_delta > article_entity_score_clamp[1]:
                ent_delta = article_entity_score_clamp[1]
            elif ent_delta < article_entity_score_clamp[0]:
                ent_delta = article_entity_score_clamp[0]


            # TODO should we clamp?
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


    async def _ensure_balance_and_diversity(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Ensure source diversity with batch database query"""
        if len(results) < 5:
            return results
        
        balanced_results = []
        seen_sources = set()
        
        # Batch query for all articles
        article_ids = [aid for aid, _ in results]
        articles = self.db.get_articles_by_ids(article_ids)
        articles_dict = {a.id: a for a in articles}
        
        for article_id, score in results:
            article = articles_dict.get(article_id)
            if not article:
                continue
            
            if (article.source.id not in seen_sources or 
                len(balanced_results) < self.config.min_sources):
                balanced_results.append((article_id, score))
                seen_sources.add(article.source.id)
            elif len(balanced_results) >= self.config.min_sources:
                balanced_results.append((article_id, score))
        
        return balanced_results

    # Keep existing methods for BM25 search, semantic search, etc.
    # bm25 has no upper bound >=0, the higher the better
    def _bm25_search(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """BM25 keyword search"""
        if not self.bm25_index:
            return []
        
        q_tokens = _tokenize(query)
        scores = self.bm25_index.get_scores(q_tokens)
        
        scored_indices = [(i, scores[i]) for i in range(len(scores))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_indices[:limit]:
            if score > 0:
                article = self.bm25_articles[idx]
                results.append((article.id, score))
        
        return results
    
    def _semantic_search(self, query: str, limit: int = 50,score_threshold = 0) -> List[Tuple[str, float]]:
        """Semantic search using embeddings, raises expection if score_threshold is less than 0"""
        try:
            if score_threshold < 0:
                raise Exception("score threshold cant be less than 0")
            return self.embeddings.semantic_search(query, k=limit,score_threshold=score_threshold)
        except Exception as e:
            print(f" Semantic search failed: {e}")
            return []

    def _combine_hybrid_scores(self, bm25_results: List[Tuple[str, float]], 
                              semantic_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine BM25 and semantic scores based on configured weighting combinations in self.config"""
        
        # normalize to 0-1 range
        # assumes input scores are positive
        def normalize_scores(results):
            if not results:
                return {}
            max_score = max(score for _, score in results)
            if max_score == 0:
                return {}
            return {article_id: score/max_score for article_id, score in results}
        
        bm25_norm = normalize_scores(bm25_results)
        semantic_norm = normalize_scores(semantic_results)
        
        all_ids = set(bm25_norm.keys()) | set(semantic_norm.keys())
        
        combined = []
        for article_id in all_ids:
            bm25_score = bm25_norm.get(article_id, 0) * self.config.bm25_weight
            semantic_score = semantic_norm.get(article_id, 0) * self.config.semantic_weight
            
            final_score = bm25_score + semantic_score
            combined.append((article_id, final_score))
        
        return sorted(combined, key=lambda x: x[1], reverse=True)

    # breaking news rag
    async def _breaking_news_rag(self, query: SearchQuery, user_profile: Optional[UserProfile]) -> List[SearchResult]:
        """Breaking News RAG: Heavy freshness weighting + urgency bias"""
        recent_articles = self.db.search_articles(
            query=query.text,
            hours_back=6,
            limit=100
        )
        
        scored_results = []
        for article in recent_articles:
            urgency_score = article.urgency_score
            hours_old = (datetime.now(datetime.timezone.utc)- article.published_at).total_seconds() / 3600
            freshness_score = max(0, 1 - (hours_old / 6))
            
            final_score = (urgency_score * RAGConfig.breaking_news_urgency_coeff)+ (freshness_score * RAGConfig.breaking_news_freshness_coeff)
            scored_results.append((article.id, final_score, article))
        
        # sort based on reverse final_score to get the most recent news
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return self._create_search_results(scored_results, query, user_profile)[:query.limit]
    

    # background analysis rag
    # anaylzes based on content quality
    # assumes long articles = better quality
    # score content quality based on 3 factors: Length (40% weight) Source credibility (30% weight) Content complexity (30% weight)
    async def _background_analysis_rag(self, query: SearchQuery, user_profile: Optional[UserProfile]) -> List[SearchResult]:
        """Background Analysis RAG: Multi-factor content quality scoring"""
        
        analysis_articles = self.db.search_articles(
            query=query.text,
            content_types=[ContentType.ANALYSIS, ContentType.FEATURE],
            hours_back=24*7,  # Last week
            limit=50
        )
        
        # entity aware expansion, only 30 days, 30 new articles
        seed = set(self._ner_entities_from_text(query.text))
        if seed:
            extra = self.db.get_articles_by_entities(list(seed), limit=30, hours_back=24*30)
            existing_ids = {a.id for a in analysis_articles}
            analysis_articles += [a for a in extra if a.id not in existing_ids]

        
        scored_results = []
        for article in analysis_articles:
            # Multi-factor content quality scoring
            content_quality_score = self._calculate_content_quality(article)
            
            # Content type boost based on configuration
            if article.content_type == ContentType.ANALYSIS:
                content_boost = self.config.background_analysis_boost
            else:  # ContentType.FEATURE
                content_boost = self.config.background_feature_boost
            
            final_score = content_quality_score * content_boost
            scored_results.append((article.id, final_score, article))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return self._create_search_results(scored_results, query, user_profile)[:query.limit]

    def _calculate_content_quality(self, article: Article) -> float:
       """Calculate multi-factor content quality score"""
       
       # 1. Length-based quality (longer = more comprehensive)
       length_score = min(1.0, len(article.content) / self.config.background_content_length_threshold)
       
       # 2. Source credibility (Reuters > Fox News > Unknown)
       credibility_score = article.source.credibility_score  # Already 0-1 range
       
       # 3. Content complexity (simple heuristic for now)
       complexity_score = self._estimate_content_complexity(article)
       
       # Weighted combination
       quality_score = (
           length_score * self.config.content_length_weight +
           credibility_score * self.config.source_credibility_weight +
           complexity_score * self.config.content_complexity_weight
       )
       
       return min(1.0, quality_score)  # Ensure max score is 1.0
    
    def _estimate_content_complexity(self, article: Article) -> float:
       """Simple content complexity estimation"""
       try:
           content = article.content
           if not content or len(content) < 100:
               return 0.0
           
           # Simple complexity metrics
           avg_sentence_length = self._calculate_avg_sentence_length(content)
           unique_word_ratio = self._calculate_unique_word_ratio(content)
           
           # Normalize to 0-1 range
           # Optimal sentence length: 15-25 words (inverted U-shape)
           sentence_complexity = 1 - abs(avg_sentence_length - 20) / 20
           sentence_complexity = max(0, min(1, sentence_complexity))
           
           # Higher unique word ratio = more complex vocabulary
           vocab_complexity = min(1.0, unique_word_ratio * 2)  # Cap at 50% unique words
           
           # Combined complexity score
           complexity_score = (sentence_complexity * 0.6 + vocab_complexity * 0.4)
           return max(0.0, min(1.0, complexity_score))
           
       except Exception:
           # Fallback to medium complexity if calculation fails
           return 0.5
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
       """Calculate average sentence length in words"""
       # Simple sentence splitting (could use NLTK for better accuracy)
       sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
       
       if not sentences:
           return 10.0  # Default fallback
       
       total_words = sum(len(sentence.split()) for sentence in sentences)
       return total_words / len(sentences)
    
    def _calculate_unique_word_ratio(self, text: str) -> float:
       """Calculate ratio of unique words to total words"""
       words = text.lower().split()
       if not words:
           return 0.0
       
       unique_words = set(words)
       return len(unique_words) / len(words)
    
    def _create_search_results(self, scored_results: List[Tuple], 
                             query: SearchQuery, 
                             user_profile: Optional[UserProfile]) -> List[SearchResult]:
       """Convert scored results to SearchResult objects"""
       search_results = []
       
       for item in scored_results:
           if len(item) == 3:
               article_id, final_score, article = item
           else:
               article_id, final_score = item
               article = self.db.get_article_by_id(article_id)
               if not article:
                   continue
           
           explanation = self._generate_explanation(article, query, user_profile)
           
           search_result = SearchResult(
               article=article,
               relevance_score=final_score,
               final_score=final_score,
               explanation=explanation
           )
           
           search_results.append(search_result)
       
       return search_results
   
    def _generate_explanation(self, article: Article, query: SearchQuery, 
                             user_profile: Optional[UserProfile]) -> str:
        """Generate explanation for recommendations"""
        reasons = []
        
        if query.text.lower() in article.title.lower():
            reasons.append("matches your search terms")
        
        hours_old = (datetime.now(datetime.timezone.utc) - article.published_at).total_seconds() / 3600
        if hours_old < 6:
            reasons.append("breaking news")
        elif hours_old < 24:
            reasons.append("recent coverage")
        
        if article.content_type == ContentType.ANALYSIS:
            reasons.append("in-depth analysis")
        elif article.content_type == ContentType.BREAKING_NEWS:
            reasons.append("breaking news alert")
        
        if article.source.credibility_score > 0.8:
            reasons.append("from trusted source")
        
        # audience removed
            
        # Entity explanation: show up to two seed entities present in this article
        seed = set(self._ner_entities_from_text(query.text))
        if seed and article.entities:
            inter = [e for e in article.entities if e in seed][:2]
            if inter:
                reasons.append(f"mentions {', '.join(inter)}")

        return f"Recommended because: {', '.join(reasons[:3])}" if reasons else "Relevant content"
    
    def _ner_entities_from_text(self, text: str) -> list[str]:
        if not text: return []
        doc = _NER(text)
        keep = {"PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART"}
        return list({e.text.strip() for e in doc.ents if e.label_ in keep})
    