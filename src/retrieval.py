import math
import os
import pickle
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from src.utils.helpers import rrf

from src.data_models import Article, UserProfile, SearchQuery, SearchResult, ContentType
from src.config import RAGConfig
from src.storage import ArticleDB
from src.embeddings import EmbeddingSystem
from src.scoring import ScoringEngine
from src.utils.content_analysis import ContentQualityAnalyzer
from src.reranker import RerankingEngine
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
            
            # Validate cache data
            if self.bm25_index is None or len(self.bm25_articles) == 0:
                print(f" Invalid cache: index={self.bm25_index is not None}, articles={len(self.bm25_articles)}")
                self._rebuild_bm25_index()
                return
                
            print(f" Loaded BM25 cache with {len(self.bm25_articles)} articles")
        except Exception as e:
            print(f" Failed to load BM25 cache: {e}")
            self._rebuild_bm25_index()
            
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from database"""
        articles = self.db.get_recent_articles(limit=1000, hours_back=24*30)
        
        if articles:
            corpus = []
            self.bm25_articles = []
            
            for article in articles:
                text = f"{article.title} {article.description or ''} {article.content[:500]}"
                doc_tokens = _tokenize(text)
                # Only add articles with non-empty tokens
                if doc_tokens:
                    corpus.append(doc_tokens)
                    self.bm25_articles.append(article)
            
            if corpus:
                self.bm25_index = BM25Okapi(corpus)
                print(f"Built BM25 index with {len(self.bm25_articles)} articles")
            else:
                print("No valid articles found for BM25 index (all articles had empty tokens)")
                self.bm25_index = None
                self.bm25_articles = []
        else:
            print("No articles found for BM25 index")
            self.bm25_index = None
            self.bm25_articles = []
    
    def _save_bm25_cache(self):
        """Save BM25 index to cache"""
        try:
            # Only save if we have valid data
            if self.bm25_index is None or len(self.bm25_articles) == 0:
                print(" Skipping cache save: invalid index or empty articles")
                return
                
            os.makedirs(os.path.dirname(self.bm25_cache_path), exist_ok=True)
            cache_data = {
                'index': self.bm25_index,
                'articles': self.bm25_articles,
                'created_at': datetime.now(timezone.utc).isoformat()
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
    # defaılts to hybrid rag for now
    async def search(self, query: SearchQuery, user_profile: Optional[UserProfile] = None) -> List[SearchResult]:
        """Main search function with performance monitoring."""
        start_time = time.time()
        
        try:
            # Query type refers to different angles or aspects of coverage for the same underlying news event
            # "Breaking news" = immediate, surface-level updates "Background pieces" = explanatory journalism that provides context
            # None = use hybrid RAG 

            if not query.query_type:
                results = await self._hybrid_rag(query, user_profile)

            # TODO currently not used in the MVP
            #else:
            #    if query.query_type == "breaking":
            #        results = await self._breaking_news_rag(query, user_profile)
            #    elif query.query_type == "background": 
            #        results = await self._background_analysis_rag(query, user_profile)
        
            
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
        try:
            # 1) Retrieve (high depths for recall)
            print("Starting BM25 search...")
            bm25_results = self._bm25_search(query.text, self.config.BM25_K)
            print(f"BM25 results: {len(bm25_results)}")
            
            print("Starting semantic search...")
            semantic_results = self._semantic_search(query.text, self.config.DENSE_K)
            print(f"Semantic results: {len(semantic_results)}")

            # 2) Hybrid candidate pooling via RRF
            print("Starting RRF pooling...")
            pooled = self._pool_with_rrf(bm25_results, semantic_results,k_rrf=self.config.RRF_K,k_pool=self.config.POOL_K)  # -> [(doc_id, rrf_score)] desc
            print(f"Pooled results: {len(pooled)}")
            
            # 3) Optional graph expansion
            if self.config.enable_graph_rag:
                print("Starting graph expansion...")
                pooled = await self._apply_graph_expansion(query.text, pooled)
                pooled = pooled[:self.config.POOL_K]  # re-cap after expansion
                print(f"Graph expanded results: {len(pooled)}")
        except Exception as e:
            print(f"Error in _hybrid_rag: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 4) Cross-encoder re-ranking (precision)
        try:
            if self.config.enable_cross_encoder:
                print("Starting cross-encoder reranking...")
                # slice BEFORE DB/CE work
                candidate_ids = [doc_id for doc_id, _ in pooled[:self.config.CE_K]]
                print(f"Candidate IDs: {len(candidate_ids)}")
                # fetch only what you will score
                articles = self.db.get_articles_by_ids(candidate_ids)
                articles_dict = {a.id: a for a in articles}
                print(f"Articles dict: {len(articles_dict)}")
                # CE-only ordering: returns [(id, ce_logit)] desc
                ce_ranked = await self.reranking_engine.apply_cross_encoder_reranking(
                    query.text, candidate_ids, articles_dict, limit=self.config.CE_K
                )
                pooled= ce_ranked
                print(f"CE ranked results: {len(pooled)}")
            else:
                print("Skipping cross-encoder, building articles dict...")
                # For non-CE configs, still need articles_dict for personalization and diversification
                candidate_ids = [doc_id for doc_id, _ in pooled]
                articles = self.db.get_articles_by_ids(candidate_ids)
                articles_dict = {a.id: a for a in articles}
                print(f"Articles dict: {len(articles_dict)}")
        except Exception as e:
            print(f"Error in cross-encoder section: {e}")
            import traceback
            traceback.print_exc()
            raise

        # TODO
        # 5) Personalization, does not exist for MVP
        try:
            print("Starting personalization...")
            if user_profile:
                user_id = getattr(user_profile, "user_id", None) or "default"
                user_prefs = self.db.get_user_prefs(user_id)
                personalized_results = self.scoring_engine.apply_user_preferences(
                    pooled, user_profile, articles_dict, user_prefs
                )
            else:
                personalized_results = pooled
            print(f"Personalized results: {len(personalized_results)}")
        except Exception as e:
            print(f"Error in personalization section: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        # 7) Diversification / balance
        try:
            print("Starting diversification...")
            final_k = getattr(self.config, "final_k", max(100, len(personalized_results)))
            diversified_results = self.reranking_engine.apply_mmr_diversification(
                personalized_results, top_k=min(final_k, len(personalized_results)), articles_dict=articles_dict
            )
            print(f"Diversified results: {len(diversified_results)}")
        except Exception as e:
            print(f"Error in diversification section: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        # 8) Materialize output
        try:
            print("Creating final search results...")
            final_results = self._create_search_results(diversified_results, query, user_profile)
            print(f"Final results: {len(final_results)}")
        except Exception as e:
            print(f"Error in final results creation: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        return final_results[:query.limit]

    
    # TODO currently not called as the routing mech is not implemented yet
    # looks at recent news articles, 6 hours old max
    # adds urgency score and freshness score to boost articles
    # TODO currently does not implement ANY user preference or similarity of articles or entity expansion
    #async def _breaking_news_rag(self, query: SearchQuery, user_profile: Optional[UserProfile]) -> List[SearchResult]:
    #    """Breaking News RAG: Heavy freshness weighting + urgency bias"""
    #    recent_articles = self.db.search_articles(
    #        query=query.text,
    #        hours_back=6,
    #        limit=100
    #    )
    #    
    #    scored_results = []
    #    for article in recent_articles:
    #        urgency_score = article.urgency_score
    #        hours_old = (datetime.now(timezone.utc)- article.published_at).total_seconds() / 3600
    #        freshness_score = max(0, 1 - (hours_old / 6))
    #        
    #        final_score = (urgency_score * RAGConfig.breaking_news_urgency_coeff)+ (freshness_score * RAGConfig.breaking_news_freshness_coeff)
    #        scored_results.append((article.id, final_score, article))
    #    
    #    # sort based on reverse final_score to get the most recent news
    #    scored_results.sort(key=lambda x: x[1], reverse=True)
    #    return self._create_search_results(scored_results, query, user_profile)[:query.limit]
    #
    ## TODO currently not called as the routing mech is not implemented yet
    ## prioritizes content quality over freshness, looks at the articles from last week
    ## uses graph RAG to expand the results
    ## no similarity of articles
    #async def _background_analysis_rag(self, query: SearchQuery, user_profile: Optional[UserProfile]) -> List[SearchResult]:
    #    """Background Analysis RAG: Multi-factor content quality scoring"""
    #    
    #    analysis_articles = self.db.search_articles(
    #        query=query.text,
    #        content_types=[ContentType.ANALYSIS, ContentType.FEATURE],
    #        hours_back=24*7,  # Last week
    #        limit=50
    #    )
    #    
    #    # Convert to initial results format for graph expansion
    #    initial_results = [(a.id, 1.0) for a in analysis_articles]  # Base score of 1.0
    #    
    #    if self.config.enable_graph_rag:
    #        expanded_results = await self._apply_graph_expansion(query.text, initial_results)
    #    else:
    #        expanded_results = initial_results
    #    
    #    # Get the expanded articles
    #    expanded_article_ids = [aid for aid, _ in expanded_results]
    #    expanded_articles = self.db.get_articles_by_ids(expanded_article_ids)
    #    articles_dict = {a.id: a for a in expanded_articles}
    #    
    #    scored_results = []
    #    for article_id, base_score in expanded_results:
    #        article = articles_dict.get(article_id)
    #        if not article:
    #            continue
    #            
    #        # Multi-factor content quality scoring
    #        content_quality_score = self.content_analyzer.calculate_content_quality(article)
    #        
    #        # Content type boost based on configuration
    #        if article.content_type == ContentType.ANALYSIS:
    #            content_boost = self.config.background_analysis_boost
    #        else:  # ContentType.FEATURE
    #            content_boost = self.config.background_feature_boost
    #        
    #        # Combine base score (from graph expansion) with content quality
    #        final_score = base_score * content_quality_score * content_boost
    #        scored_results.append((article.id, final_score, article))
    #    
    #    scored_results.sort(key=lambda x: x[1], reverse=True)
    #    return self._create_search_results(scored_results, query, user_profile)[:query.limit]


    # ============================================================================
    # SEARCH METHODS
    # ============================================================================
    
    def _bm25_search(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """BM25 keyword search, returns descending sorted [(article_id,score)]"""
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
        
    def _pool_with_rrf(self, bm25_results, dense_results, k_rrf: int = 60, k_pool: int = 300):
        """
        Build a candidate pool from BM25 & dense retrievers using Reciprocal Rank Fusion (RRF).
        Returns a list of (doc_id, rrf_score) sorted by rrf_score desc.
        DO NOT use this order as final ranking; it's only for candidate selection.
        """
        # results are [(id, score)], sorted desc per retriever
        bm25_ids = [doc_id for doc_id, _ in bm25_results]
        dense_ids = [doc_id for doc_id, _ in dense_results]
        all_ids = list(dict.fromkeys(bm25_ids + dense_ids))  # stable union, dedup

        # Build per-retriever ranks (1-based)
        bm25_rank = {doc_id: i+1 for i, doc_id in enumerate(bm25_ids)}
        dense_rank = {doc_id: i+1 for i, doc_id in enumerate(dense_ids)}

        pooled = []
        for doc_id in all_ids:
            s = 0.0
            if doc_id in bm25_rank: s += rrf(bm25_rank[doc_id],k_rrf)
            if doc_id in dense_rank: s += rrf(dense_rank[doc_id],k_rrf)
            pooled.append((doc_id, s))

        pooled.sort(key=lambda x: x[1], reverse=True)
        return pooled[:k_pool]


    # ============================================================================
    # GRAPH RAG AND ENTITY EXPANSION
    # ============================================================================

    async def _apply_graph_expansion(self, query: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Simplified Graph RAG: Find related articles using entity relationships
        
        Core idea: 
        1. Extract entities from query
        2. Find entities that co-occur with query entities  
        3. Boost articles mentioning these related entities
        """
        if not results:
            return results

        # 1) Extract entities from query using spaCy NER
        seed_entities = set(self._ner_entities_from_text(query))
        
        # 2) If no entities in query, use entities from top results
        if not seed_entities:
            top_ids = [aid for aid, _ in results[:5]]
            articles = self.db.get_articles_by_ids(top_ids)
            for article in articles:
                if article.entities:
                    # Safely handle entities, extracting just the entity names
                    for entity in article.entities[:3]:  # Limit to 3 entities per article
                        if isinstance(entity, (tuple, list)) and len(entity) >= 1:
                            # Extract entity name (first element of tuple)
                            entity_name = entity[0] if isinstance(entity[0], str) else str(entity[0])
                            seed_entities.add(entity_name)
                        elif isinstance(entity, str):
                            seed_entities.add(entity)
            seed_entities = list(seed_entities)[:5]  # Limit total seed entities
        
        if not seed_entities:
            return results

        # 3) Find related entities that co-occur with seed entities
        related_entities = self.db.get_comention_counts(list(seed_entities), limit=20)
        
        # 4) Get articles mentioning seed or related entities
        all_entities = list(seed_entities) + list(related_entities.keys())[:10]  # Top 10 related
        graph_articles = self.db.get_articles_by_entities(all_entities, limit=50, hours_back=24*7)

        # 5) Apply simple scoring boost
        base_scores = {doc_id: score for doc_id, score in results}
        
        for article in graph_articles:
            # Safely extract entity names from article.entities
            article_entity_names = set()
            if article.entities:
                for entity in article.entities:
                    if isinstance(entity, (tuple, list)) and len(entity) >= 1:
                        # Extract entity name (first element of tuple)
                        entity_name = entity[0] if isinstance(entity[0], str) else str(entity[0])
                        article_entity_names.add(entity_name)
                    elif isinstance(entity, str):
                        article_entity_names.add(entity)
            
            # Boost if article mentions seed entities
            seed_overlap = len(seed_entities & article_entity_names)
            if seed_overlap > 0:
                boost = 0.2 * seed_overlap  # Simple linear boost
                base_scores[article.id] = base_scores.get(article.id, 0) + boost
            
            # Add new articles discovered through graph
            elif article.id not in base_scores:
                base_scores[article.id] = 0.1  # Small base score for graph-discovered articles

        # Return sorted results
        return sorted(base_scores.items(), key=lambda x: x[1], reverse=True)


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
   
    def _generate_explanation(self, article: Article, query: SearchQuery, 
                             user_profile: Optional[UserProfile]) -> str:
        """Generate simple explanation for recommendations - easy to explain in interviews"""
        reasons = []
        
        # Check if query terms appear in title
        if query.text.lower() in article.title.lower():
            reasons.append("matches your search terms")
        
        # Check freshness
        hours_old = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
        if hours_old < 6:
            reasons.append("breaking news")
        elif hours_old < 24:
            reasons.append("recent coverage")
        
        # Check content type
        if article.content_type == ContentType.ANALYSIS:
            reasons.append("in-depth analysis")
        elif article.content_type == ContentType.BREAKING_NEWS:
            reasons.append("breaking news alert")
        
        # Check source credibility
        if article.source.credibility_score > 0.8:
            reasons.append("from trusted source")
            
        # Show entity overlap (Graph RAG feature)
        seed_entities = set(self._ner_entities_from_text(query.text))
        if seed_entities and article.entities:
            # Safely extract entity names from article.entities
            article_entity_names = []
            for entity in article.entities:
                if isinstance(entity, (tuple, list)) and len(entity) >= 1:
                    # Extract entity name (first element of tuple)
                    entity_name = entity[0] if isinstance(entity[0], str) else str(entity[0])
                    article_entity_names.append(entity_name)
                elif isinstance(entity, str):
                    article_entity_names.append(entity)
            
            overlapping_entities = [e for e in article_entity_names if e in seed_entities][:2]
            if overlapping_entities:
                reasons.append(f"mentions {', '.join(overlapping_entities)}")

        return f"Recommended because: {', '.join(reasons[:3])}" if reasons else "Relevant content"
    