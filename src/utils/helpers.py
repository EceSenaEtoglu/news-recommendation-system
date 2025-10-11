# RRF score
def rrf(rank,k_rrf): 
      return 1.0 / (k_rrf + rank)


def build_bm25_index(db):
    """Build BM25 index from database articles"""
    from rank_bm25 import BM25Okapi
    import pickle
    import os
    from datetime import datetime, timezone
    from src.storage import ArticleDB
    
    # Get recent articles for BM25 index
    articles = db.get_recent_articles(limit=1000, hours_back=24*7)
    
    if not articles:
        print("No articles found for BM25 index")
        return
    
    # Build corpus
    corpus = []
    bm25_articles = []
    
    for article in articles:
        text = f"{article.title} {article.description or ''} {article.content[:500]}"
        # Use same tokenization as MultiRAGRetriever
        from src.retrieval import _tokenize
        doc_tokens = _tokenize(text)
        corpus.append(doc_tokens)
        bm25_articles.append(article)
    
    # Build BM25 index
    bm25_index = BM25Okapi(corpus)
    
    # Save to cache
    cache_path = "db/bm25_cache.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {
        'index': bm25_index,
        'articles': bm25_articles,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"BM25 index built with {len(bm25_articles)} articles")