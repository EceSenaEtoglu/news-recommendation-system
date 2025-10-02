import math
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

from .data_models import Article, UserProfile, SearchQuery, SearchResult, ContentType
from .config import RAGConfig
from .storage import ArticleDB
from .embeddings import EmbeddingSystem
from .scoring import ScoringEngine
from .content_analysis import ContentQualityAnalyzer
from .reranker import RerankingEngine
import spacy
_NER = spacy.load("en_core_web_sm")

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS

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
        
        # Initialize helper engines
        self.scoring_engine = ScoringEngine(self.config)
        self.content_analyzer = ContentQualityAnalyzer(self.config)
        self.reranking_engine = RerankingEngine(self.config, embeddings)
        
        # Entity cache for Graph RAG
        self.entity_cache = {}
    
    # ============================================================================
    # INITIALIZATION AND CACHING
    # ============================================================================
    
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
    

    # ============================================================================
    # MAIN SEARCH INTERFACE
    # ============================================================================

    # Route to appropriate RAG strategy based on query type
    # TODO routing mechanism should be enhanced!
    async def search(self, query: SearchQuery, user_profile: Optional[UserProfile] = None) -> List[SearchResult]:
        """Main search function with performance monitoring. Uses query routing mechanism"""
        start_time = time.time()
        
        try:
            # Query type refers to different angles or aspects of coverage for the same underlying news event
            # "Breaking news" = immediate, surface-level updates
            # "Background pieces" = explanatory journalism that provides context
            # None = use hybrid RAG (default for MVP)

            if query.query_type:
            if query.query_type == "breaking":
                results = await self._breaking_news_rag(query, user_profile)
            elif query.query_type == "background": 
                results = await self._background_analysis_rag(query, user_profile)
                else:
                    results = await self._hybrid_rag(query, user_profile)

            else:
                results = await self._hybrid_rag(query, user_profile)
            
            # Performance logging
            search_time = time.time() - start_time
            print(f" Search completed in {search_time:.2f}s ({len(results)} results)")
            
            return results
            
        except Exception as e:
            print(f" Search failed: {e}")
            return []

    # ============================================================================
    # RAG STRATEGIES
    # ============================================================================

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
            # Get articles for reranking
            article_ids = [aid for aid, _ in expanded_results[:20]]
            articles = self.db.get_articles_by_ids(article_ids)
            articles_dict = {a.id: a for a in articles}
            reranked_results = await self.reranking_engine.apply_cross_encoder_rerank(
                query.text, expanded_results[:20], articles_dict)
        else:
            reranked_results = expanded_results
        
        # Step 6: Apply freshness weighting (Freshness-aware RAG)
        article_ids = [aid for aid, _ in reranked_results]
        articles = self.db.get_articles_by_ids(article_ids)
        articles_dict = {a.id: a for a in articles}
        fresh_results = self.scoring_engine.apply_freshness_boost(reranked_results, articles_dict)
        
        # Step 7: Apply user personalization (Memory RAG)  
        if user_profile:
            user_id = getattr(user_profile, "user_id", None) or "default"
            user_prefs = self.db.get_user_prefs(user_id)
            personalized_results = self.scoring_engine.apply_user_preferences(
                fresh_results, user_profile, articles_dict, user_prefs)
        else:
            personalized_results = fresh_results
        
        # Step 8: Optional MMR diversification, then ensure source diversity
        diversified_results = self.reranking_engine.apply_mmr_diversification(
            personalized_results, top_k=100, articles_dict=articles_dict)
        balanced_results = self.scoring_engine.ensure_balance_and_diversity(
            diversified_results, articles_dict)
        
        # Step 9: Convert to SearchResult objects
        final_results = self._create_search_results(balanced_results, query, user_profile)
        
        return final_results[:query.limit]

    # looks at recent news articles, 6 hours old max
    # adds urgency score and freshness score to boost articles
    # TODO currently does not implement ANY user preference or similarity of articles or entity expansion
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
    
    # prioritizes content quality over freshness, looks at the articles from last week
    # uses graph RAG to expand the results
    # no similarity of articles
    async def _background_analysis_rag(self, query: SearchQuery, user_profile: Optional[UserProfile]) -> List[SearchResult]:
        """Background Analysis RAG: Multi-factor content quality scoring"""
        
        analysis_articles = self.db.search_articles(
            query=query.text,
            content_types=[ContentType.ANALYSIS, ContentType.FEATURE],
            hours_back=24*7,  # Last week
            limit=50
        )
        
        # Convert to initial results format for graph expansion
        initial_results = [(a.id, 1.0) for a in analysis_articles]  # Base score of 1.0
        
        if self.config.enable_graph_rag:
            expanded_results = await self._apply_graph_expansion(query.text, initial_results)
        else:
            expanded_results = initial_results
        
        # Get the expanded articles
        expanded_article_ids = [aid for aid, _ in expanded_results]
        expanded_articles = self.db.get_articles_by_ids(expanded_article_ids)
        articles_dict = {a.id: a for a in expanded_articles}
        
        scored_results = []
        for article_id, base_score in expanded_results:
            article = articles_dict.get(article_id)
            if not article:
                continue
                
            # Multi-factor content quality scoring
            content_quality_score = self.content_analyzer.calculate_content_quality(article)
            
            # Content type boost based on configuration
            if article.content_type == ContentType.ANALYSIS:
                content_boost = self.config.background_analysis_boost
            else:  # ContentType.FEATURE
                content_boost = self.config.background_feature_boost
            
            # Combine base score (from graph expansion) with content quality
            final_score = base_score * content_quality_score * content_boost
            scored_results.append((article.id, final_score, article))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return self._create_search_results(scored_results, query, user_profile)[:query.limit]


    # ============================================================================
    # SEARCH METHODS
    # ============================================================================
    
    def _bm25_search(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """BM25 keyword search"""
        if not self.bm25_index:
            return []
        
        q_tokens = _tokenize(query)
        # returns one score for each article in the index
        # scores is a list of length len(bm25_articles)
        scores = self.bm25_index.get_scores(q_tokens)
        
        scored_indices = [(i, scores[i]) for i in range(len(scores))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_indices[:limit]:
            # if 0, no query term was found in the article
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


    # ============================================================================
    # GRAPH RAG AND ENTITY EXPANSION
    # ============================================================================

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


    def _ner_entities_from_text(self, text: str) -> list[str]:
        if not text: return []
        doc = _NER(text)
        keep = {"PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART"}
        return list({e.text.strip() for e in doc.ents if e.label_ in keep})

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
            from nltk.corpus import stopwords
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

    # ============================================================================
    # RESULT FORMATTING AND EXPLANATION
    # ============================================================================
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
   
    # TODO use a gpt to create a detailed explanation
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
            
        # Entity explanation: show up to two seed entities present in this article
        seed = set(self._ner_entities_from_text(query.text))
        if seed and article.entities:
            inter = [e for e in article.entities if e in seed][:2]
            if inter:
                reasons.append(f"mentions {', '.join(inter)}")

        return f"Recommended because: {', '.join(reasons[:3])}" if reasons else "Relevant content"
    