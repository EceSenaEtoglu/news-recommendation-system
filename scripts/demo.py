#!/usr/bin/env python3
"""
RAGify-News AI Demo
==================

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
from src.recommendation_learner import AIRecommender, RecommendationConfig
from src.providers.fixture import FixtureProvider
from src.ingestion import extract_entities
import subprocess
import sys


def fetch_and_setup_data(featured_count=20, pool_count=100):
    """Fetch new articles and setup data"""
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
    """Setup data by importing fixtures and building index"""
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
            entity_tuples = [(entity, "PERSON", 1) for entity in article.entities]
        else:
            entity_tuples = extract_entities(article)
        
        if entity_tuples:
            db.upsert_article_entities(article.id, entity_tuples)
    
    # Build index
    print("Building FAISS index...")
    emb.rebuild_index_from_db(db)
    
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


def run_demo():
    """Run the comprehensive AI demo"""
    print("RAGify-News AI Demo")
    print("=" * 60)
    print()
    
    # Initialize components
    db = ArticleDB("db/articles.db")
    emb = EmbeddingSystem()
    
    # Check if we have data
    stats = db.get_stats()
    if stats['total_articles'] == 0:
        print("No articles in database!")
        print()
        print("Please run setup first:")
        print("  python scripts/demo.py --setup")
        print()
        return
    
    print(f"Database has {stats['total_articles']} articles")
    print()
    
    # Demo 1: Embedding Models
    print("DEMO 1: Multi-Model Embeddings")
    print("=" * 50)
    
    models = emb.get_model_info()
    print("Available Models:")
    for name, desc in models.items():
        print(f"  • {name}: {desc}")
    print()
    
    # Test different models
    test_text = "Elon Musk announces new Tesla AI features for autonomous driving"
    print("Encoding test text with different models:")
    for model_name in ["all-MiniLM-L6-v2", "news-similarity"]:
        try:
            embedding = emb.encode_text(test_text, model_name)
            print(f"  {model_name}: {embedding.shape} - {embedding[:5]}...")
        except Exception as e:
            print(f"  {model_name}: Error - {e}")
    print()
    
    # Demo 2: Neural Reranker
    print("DEMO 2: Neural Reranker")
    print("=" * 50)
    
    config = RecommendationConfig(use_neural_reranker=True)
    rec = AIRecommender(db, emb, config)
    
    print(f"Neural reranker initialized: {rec._neural_reranker is not None}")
    print(f"Advanced feature extractor ready: {rec._advanced_feature_extractor is not None}")
    
    # Train the neural reranker
    print("Training neural reranker...")
    rec.train_neural_reranker()
    print("Training completed successfully!")
    print()
    
    # Demo 3: Enhanced Recommendations
    print("DEMO 3: Enhanced Recommendations")
    print("=" * 50)
    
    # Get a sample article
    articles = db.get_recent_articles(limit=1)
    if not articles:
        print("No recent articles found.")
        return
    
    seed_article = articles[0]
    print(f"Seed article: {seed_article.title}")
    print(f"Source: {seed_article.source.name}")
    print(f"Published: {seed_article.published_at.strftime('%Y-%m-%d')}")
    print()
    
    # Test different recommendation approaches
    configs = [
        ("Basic", RecommendationConfig(use_neural_reranker=False, use_mmr=False)),
        ("Neural Reranker", RecommendationConfig(use_neural_reranker=True, use_mmr=False)),
        ("MMR Diversification", RecommendationConfig(use_neural_reranker=False, use_mmr=True)),
        ("Full Enhanced", RecommendationConfig(use_neural_reranker=True, use_mmr=True))
    ]
    
    for name, config in configs:
        print(f"{name} Recommendations:")
        print("-" * 30)
        
        rec = AIRecommender(db, emb, config)
        
        # Train neural reranker if needed
        if config.use_neural_reranker:
            rec.train_neural_reranker()
        
        # Get recommendations
        recommendations = rec.recommend_for_article(seed_article, k=3)
        
        for i, (candidate, score) in enumerate(recommendations, 1):
            explanation = rec.explain_recommendation(seed_article, candidate)
            print(f"  {i}. {score:.3f} | {candidate.title[:60]}...")
            print(f"     {explanation}")
        
        print()
    
    # Demo 4: Multi-Model Fusion
    print("DEMO 4: Multi-Model Fusion")
    print("=" * 50)
    
    # Test different fusion methods
    fusion_methods = ["weighted_average", "rank_fusion", "max_score"]
    models_to_test = ["all-MiniLM-L6-v2", "news-similarity"]
    
    for method in fusion_methods:
        print(f"Fusion Method: {method}")
        print("-" * 25)
        
        # Use multi-model search
        query_text = f"{seed_article.title} {seed_article.description or ''}"
        results = emb.multi_model_search(query_text, models=models_to_test, k=3, fusion_method=method)
        
        for i, (article_id, score) in enumerate(results, 1):
            article = db.get_article_by_id(article_id)
            if article:
                print(f"  {i}. {score:.3f} | {article.title[:50]}...")
        
        print()
    
    # Demo 5: Interactive Commands
    print("DEMO 5: Interactive Commands")
    print("=" * 50)
    
    print("You can now try these interactive commands:")
    print()
    
    # Show available articles
    recent_articles = db.get_recent_articles(limit=5)
    print("Recent articles you can test with:")
    for i, article in enumerate(recent_articles, 1):
        print(f"  {i}. {article.id} - {article.title[:50]}...")
    print()
    
    print("Try these commands:")
    print(f"  python scripts/demo.py --recommend --id {seed_article.id}")
    print(f"  python scripts/demo.py --enhanced --id {seed_article.id}")
    print(f"  python scripts/demo.py --multi-model --id {seed_article.id}")
    print()
    
    print("Demo completed successfully!")
    print()


def cmd_recommend(article_id: str, k: int = 5):
    """Get basic recommendations for an article"""
    db = ArticleDB("db/articles.db")
    emb = EmbeddingSystem()
    
    # Configure for larger candidate pool
    config = RecommendationConfig(
        top_k=20,  # Increased from default 10
        mmr_pool=100,  # Increased from default 50
        use_mmr=True
    )
    rec = AIRecommender(db, emb, config)
    
    article = db.get_article_by_id(article_id)
    if not article:
        print(f"Article not found: {article_id}")
        return
    
    print(f"Basic Recommendations for: {article.title}")
    print("=" * 60)
    
    recommendations = rec.recommend_for_article(article, k=k)
    for i, (candidate, score) in enumerate(recommendations, 1):
        explanation = rec.explain_recommendation(article, candidate)
        print(f"{i}. {score:.3f} | {candidate.title}")
        print(f"   {explanation}")
        print()


def cmd_enhanced_recommend(article_id: str, k: int = 5, model_name: str = None):
    """Get enhanced recommendations with neural reranker"""
    db = ArticleDB("db/articles.db")
    emb = EmbeddingSystem()
    
    if model_name:
        emb.switch_model(model_name)
    
    # Configure for larger candidate pool
    config = RecommendationConfig(
        top_k=20,  # Increased from default 10
        mmr_pool=100,  # Increased from default 50
        use_neural_reranker=True, 
        use_mmr=True
    )
    rec = AIRecommender(db, emb, config)
    
    article = db.get_article_by_id(article_id)
    if not article:
        print(f"Article not found: {article_id}")
        return
    
    print(f"Enhanced Recommendations for: {article.title}")
    print("=" * 60)
    
    # Train neural reranker
    print("Training neural reranker...")
    rec.train_neural_reranker()
    
    recommendations = rec.recommend_for_article(article, k=k)
    for i, (candidate, score) in enumerate(recommendations, 1):
        explanation = rec.explain_recommendation(article, candidate)
        print(f"{i}. {score:.3f} | {candidate.title}")
        print(f"   {explanation}")
        print()


def cmd_multi_model_recommend(article_id: str, k: int = 5, models: list = None):
    """Get multi-model fusion recommendations"""
    if models is None:
        models = ["all-MiniLM-L6-v2", "news-similarity"]
    
    db = ArticleDB("db/articles.db")
    emb = EmbeddingSystem()
    
    # Configure for larger candidate pool
    config = RecommendationConfig(
        top_k=20,  # Increased from default 10
        mmr_pool=100,  # Increased from default 50
        use_mmr=True
    )
    
    article = db.get_article_by_id(article_id)
    if not article:
        print(f"Article not found: {article_id}")
        return
    
    print(f"Multi-Model Recommendations for: {article.title}")
    print("=" * 60)
    
    # Use multi-model search with larger pool
    query_text = f"{article.title} {article.description or ''}"
    results = emb.multi_model_search(query_text, models=models, k=k*2, fusion_method="weighted_average")  # Get more to filter out original
    
    # Filter out the original article and take top k
    filtered_results = [(aid, score) for aid, score in results if aid != article_id][:k]
    
    for i, (aid, score) in enumerate(filtered_results, 1):
        candidate = db.get_article_by_id(aid)
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
    emb = EmbeddingSystem()
    stats = emb.get_stats()
    
    print("Current Model Information:")
    print("=" * 40)
    print(f"Primary Model: {stats['model']}")
    print(f"Index Vectors: {stats['total_vectors']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Available Models: {len(stats['available_models'])}")
    print()


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="RAGify-News AI Demo")
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
