#!/usr/bin/env python3
"""
AI News Demo
============

A comprehensive demonstration of the AI-powered news recommendation system.
This script showcases all the enhanced AI capabilities including neural reranking,
multi-model embeddings, and advanced recommendation algorithms.

Usage:
    python scripts/demo.py

Requirements:
    - Run 'python scripts/demo.py --setup' first to import data and build index
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import ArticleDB
from src.embeddings import EmbeddingSystem
from src.recommendation_system import RecommendationSystem, RecommendationConfig
from src.providers.fixture import FixtureProvider
from src.utils.ingestion import extract_entities
from src.utils.helpers import build_bm25_index
import subprocess
import sys

# Global shared instances for performance
_db = None
_emb = None
_rec_system = None

def get_recommendation_system():
    """Get shared recommendation system instance (singleton pattern)"""
    global _db, _emb, _rec_system
    if _rec_system is None:
        print("Initializing recommendation system...")
        _db = ArticleDB("db/articles.db")
        _emb = EmbeddingSystem()
        config = RecommendationConfig(
            top_k=20
        )
        _rec_system = RecommendationSystem(_db, _emb, config)
        print("Recommendation system ready!")
    return _rec_system


def fetch_and_setup_data(featured_count=20, pool_count=100):
    """Fetch new articles and setup data, calls scripts/ingest_rss.py"""
    print("Fetching latest news and setting up data...")
    print("=" * 50)
    
    # Step 1: Fetch new articles
    print("Step 1: Fetching latest articles from RSS feeds...")
    cmd_fetch = [
        sys.executable,
        "scripts/ingest_rss.py",
        "--out_dir", "src/providers/news_fixtures",
        "--featured_count", str(featured_count),
        "--pool_count", str(pool_count),
    ]
    
    try:
        res_fetch = subprocess.run(
            cmd_fetch, 
            capture_output=True, 
            text=True, 
            check=False,
            cwd=os.getcwd(),
            env=os.environ.copy()
        )
        
        if res_fetch.returncode == 0:
            print("✅ Articles fetched successfully")
            if res_fetch.stdout:
                print("Fetch output:", res_fetch.stdout)
        else:
            print("❌ Failed to fetch articles")
            if res_fetch.stderr:
                print("Fetch error:", res_fetch.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Fetch failed: {e}")
        return False
    
    # Step 2: Setup data (import + index)
    print("\nStep 2: Setting up data...")
    return setup_data()


def setup_data():
    """Setup data by importing fixtures, saving them as articles to the database and building faiss index on recent articles"""
    print("Setting up data...")
    print("=" * 40)
    
    # Initialize components
    db = ArticleDB("db/articles.db")
    emb = EmbeddingSystem()
    provider = FixtureProvider()
    
    # Import fixtures
    print("Importing fixture data...")
    featured, candidates, _ = _load_fixture_articles(provider, 20, 100)
    articles = featured + candidates
    
    print(f"Importing {len(articles)} articles to database...")
    saved_count = db.save_articles(articles)
    print(f"Successfully imported {saved_count} articles")
    
    # Save entity relationships
    print("Saving entity relationships...")
    for article in articles:
        if article.entities and len(article.entities) > 0:
            # Check if entities are already in tuple format
            if isinstance(article.entities[0], tuple):
                # Use pre-extracted entities from ingest_rss.py (already processed with spaCy NER)
                # Entities are stored as (name, type, count) tuples
                entity_tuples = article.entities
            else:
                # Convert string entities to tuple format
                entity_tuples = [(name, "UNKNOWN", 1) for name in article.entities]
        else:
            # Fallback: extract entities if none were pre-extracted
            entity_tuples = extract_entities(article)
        
        if entity_tuples:
            db.update_entity_info(article.id, entity_tuples)
    
    # Build indexes
    print("Building FAISS index...")
    emb.rebuild_index_from_db(db)
    
    print("Building BM25 index...")

    build_bm25_index(db)
    
    print("Setup completed successfully!")
    print()
    return True


def _load_fixture_articles(provider: FixtureProvider, featured_limit: int = 20, candidate_limit: int = 100):
    """Load and cache fixture articles"""
    import asyncio
    featured_articles, candidate_articles = asyncio.run(
        provider.fetch_featured_and_candidates(
            featured_limit=featured_limit, 
            candidate_limit=candidate_limit
        )
    )
    all_articles_cache = {a.id: a for a in (featured_articles + candidate_articles)}
    return featured_articles, candidate_articles, all_articles_cache


def cmd_recommend(article_id: str, k: int = 5):
    """Get basic recommendations for an article"""
    rec = get_recommendation_system()
    
    article = rec.db.get_article_by_id(article_id)
    if not article:
        print(f"Article not found: {article_id}")
        return
    
    print(f"Basic Recommendations for: {article.title}")
    print("=" * 60)
    
    recommendations = rec.recommend_for_article(article, k=k)
    for i, (candidate, score) in enumerate(recommendations, 1):
        explanation = rec.explain_recommendation(article, candidate)
        source_name = getattr(candidate.source, "name", "Unknown") if hasattr(candidate, "source") and candidate.source else "Unknown"
        url = getattr(candidate, "url", "#")
        print(f"{i}. {score:.3f} | {candidate.title} | {url}")
        print(f"   Source: {source_name}")
        print(f"   {explanation}")
        print()


def cmd_enhanced_recommend(article_id: str, k: int = 5, model_name: str = None):
    """Get enhanced recommendations with neural reranker"""
    rec = get_recommendation_system()
    
    if model_name:
        rec.embeddings.switch_model(model_name)
    
    # Enable neural reranker for this command
    rec.config.use_neural_reranker = True
    
    article = rec.db.get_article_by_id(article_id)
    if not article:
        print(f"Article not found: {article_id}")
        return
    
    print(f"Enhanced Recommendations for: {article.title}")
    print("=" * 60)
    
    # Note: Neural reranker training requires user interaction data
    print("Using hybrid search with neural reranker configuration...")
    
    recommendations = rec.recommend_for_article(article, k=k)
    for i, (candidate, score) in enumerate(recommendations, 1):
        explanation = rec.explain_recommendation(article, candidate)
        source_name = getattr(candidate.source, "name", "Unknown") if hasattr(candidate, "source") and candidate.source else "Unknown"
        url = getattr(candidate, "url", "#")
        print(f"{i}. {score:.3f} | {candidate.title} | {url}")
        print(f"   Source: {source_name}")
        print(f"   {explanation}")
        print()


def cmd_multi_model_recommend(article_id: str, k: int = 5, models: list = None):
    """Get multi-model fusion recommendations"""
    if models is None:
        models = ["all-MiniLM-L6-v2", "news-similarity"]
    
    rec = get_recommendation_system()
    
    article = rec.db.get_article_by_id(article_id)
    if not article:
        print(f"Article not found: {article_id}")
        return
    
    print(f"Multi-Model Recommendations for: {article.title}")
    print("=" * 60)
    
    # Use multi-model search with larger pool
    query_text = f"{article.title} {article.description or ''}"
    results = rec.embeddings.multi_model_search(query_text, models=models, k=k*2, fusion_method="weighted_average")  # Get more to filter out original
    
    # Filter out the original article and take top k
    filtered_results = [(aid, score) for aid, score in results if aid != article_id][:k]
    
    for i, (aid, score) in enumerate(filtered_results, 1):
        candidate = rec.db.get_article_by_id(aid)
        if candidate:
            print(f"{i}. {score:.3f} | {candidate.title}")
            print(f"   Source: {candidate.source.name}")
            print()


def cmd_list_models():
    """List available embedding models"""
    emb = EmbeddingSystem()
    models = emb.get_model_info()
    
    print("Available Embedding Models:")
    print("=" * 40)
    for name, description in models.items():
        print(f"• {name}: {description}")
    print()


def cmd_model_info():
    """Show current model information"""
    print("Model information not available (get_stats method removed)")
    print()


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="AI News Demo")
    parser.add_argument("--setup", action="store_true", help="Setup data (import fixtures and build index)")
    parser.add_argument("--fetch-and-setup", action="store_true", help="Fetch new articles and setup data")
    parser.add_argument("--demo", action="store_true", help="Run comprehensive demo")
    parser.add_argument("--recommend", help="Get basic recommendations for article ID")
    parser.add_argument("--enhanced", help="Get enhanced recommendations for article ID")
    parser.add_argument("--multi-model", help="Get multi-model recommendations for article ID")
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--model", help="Specify embedding model to use")
    parser.add_argument("--models", nargs="+", help="Models for multi-model fusion")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model-info", action="store_true", help="Show model information")
    
    args = parser.parse_args()
    
    try:
        if args.setup:
            setup_data()
        elif args.fetch_and_setup:
            fetch_and_setup_data()
        elif args.demo:
            run_demo()
        elif args.recommend:
            cmd_recommend(args.recommend, args.k)
        elif args.enhanced:
            cmd_enhanced_recommend(args.enhanced, args.k, args.model)
        elif args.multi_model:
            cmd_multi_model_recommend(args.multi_model, args.k, args.models)
        elif args.list_models:
            cmd_list_models()
        elif args.model_info:
            cmd_model_info()
        else:
            # Default: run demo
            run_demo()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
