import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
from .models import Article
from .storage import ArticleDB

class EmbeddingSystem:
    """Handles embeddings and semantic search using FAISS"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "db/faiss.index",
                 metadata_path: str = "db/faiss_metadata.pkl"):
        
        # Load sentence transformer model
        print(f" Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
        
        # Metadata: article_id -> article info mapping
        self.id_to_metadata = {}
        
        # reverse mapping for faster access
        self.article_id_to_faiss_id = {}  

        # Load existing index if available
        self.load_index()
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts efficiently"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # add articles to faiss index by constructing faiss id for each article
    # construct self.id_to_metadata[faiss_id] for each faiss_id created 
    def add_articles(self, articles: List[Article]) -> int:
        """Add articles to the FAISS index, fill self.id_to_metadata
        return num of added articles"""
        if not articles:
            return 0
                
        # inside add_articles, before encoding:
        unique_articles = []
        for a in articles:
            if a.id in self.article_id_to_faiss_id:
                continue  # already indexed; skip or handle updates here
            unique_articles.append(a)
        articles = unique_articles
        
        print(f" Encoding {len(unique_articles)} articles...")

        # Prepare texts for embedding (title + description)
        texts = []
        valid_articles = []
        
        # get valid articles according to length filtering
        for article in articles:
            # TODO adjust this according to how you save it!, which attributes are included etc.
            # Combine title and description for better semantic representation

            text = f"{article.title} {article.description or ''} { (article.content or '')[:500] }"
            if len(text.strip()) > 10:  # Only index meaningful content
                texts.append(text)
                valid_articles.append(article)
        
        if not texts:
            return 0
        
        # Generate embeddings
        embeddings = self.encode_texts(texts).astype('float32', copy=False)
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Update metadata mapping for the new articles
        for i, article in enumerate(valid_articles):
            faiss_id = start_id + i
            self.id_to_metadata[faiss_id] = {
                "article_id": article.id,
                "title": article.title,
                "source": article.source.name,
                "published_at": article.published_at.isoformat(),
                "content_type": article.content_type.value,
                "target_audience": article.target_audience.value,
                "urgency_score": article.urgency_score,
                "url": article.url
            }
            
            self.article_id_to_faiss_id[article.id] = faiss_id
        
        print(f" Added {len(valid_articles)} articles to semantic index")
        return len(valid_articles)
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 10,
                       score_threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Semantic search using FAISS, provides min k candidates that match given threshold
        If no threshold specified, default filters out negative cosine similarities,
        ensuring only semantically similar (not opposite) articles are returned."""
        if self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.encode_text(query).astype('float32', copy=False).reshape(1, -1)
        query_embedding = np.ascontiguousarray(query_embedding)
        faiss.normalize_L2(query_embedding)

        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > score_threshold:  # Valid match
                
                # what is id to metadata?
                metadata = self.id_to_metadata.get(idx, {}) 
                article_id = metadata.get("article_id")
                if article_id:
                    results.append((article_id, float(score)))
        
        return results
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        # Ensure directories exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True) 

        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)
        
        print(f" Saved FAISS index with {self.index.ntotal} vectors")
    
    def load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f" Loaded FAISS index with {self.index.ntotal} vectors")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.id_to_metadata = pickle.load(f)
                print(f" Loaded metadata for {len(self.id_to_metadata)} articles")

                # REBUILD REVERSE MAP HERE
                self.article_id_to_faiss_id = {
                    meta["article_id"]: faiss_id
                    for faiss_id, meta in self.id_to_metadata.items()
                    if "article_id" in meta
                }
                # Optional sanity check:
                if len(self.article_id_to_faiss_id) != len(self.id_to_metadata):
                    print(" Warning: reverse map size does not match metadata size")
                
        except Exception as e:
            print(f" Could not load existing index: {e}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.id_to_metadata = {}
            self.article_id_to_faiss_id = {}

    
    def get_stats(self) -> dict:
        """Get embedding system statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "unknown",
            "index_type": "IndexFlatIP",
            "metadata_entries": len(self.id_to_metadata)
        }
    
    def rebuild_index_from_db(self, db: ArticleDB):
        """Rebuild the entire index from database articles"""
        print(" Rebuilding semantic index from database...")
        
        # Clear existing index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.id_to_metadata = {}
        self.article_id_to_faiss_id = {}
        
        # Get all articles from database
        # TODO parameteize hours and filtering
        articles = db.get_recent_articles(limit=10000, hours_back=24*30)  # Last 30 days
        
        if articles:
            # Add articles to index
            added = self.add_articles(articles)
            
            # Save updated index
            self.save_index()
            
            print(f" Rebuilt index with {added} articles")
            return added
        else:
            print(" No articles found in database")
            return 0


def build_embeddings_from_db(db_path: str = "db/articles.db"):
    """Utility function to build embeddings from existing database"""
    db = ArticleDB(db_path)
    embeddings = EmbeddingSystem()
    
    return embeddings.rebuild_index_from_db(db)


# Script mode
if __name__ == "__main__":
    print(" Building semantic search index...")
    
    #TODO parametrize embedding count etc.
    count = build_embeddings_from_db()
    
    if count > 0:
        print(f" Successfully built index with {count} articles!")
        
        # Initialize the vector db and embedding model
        embeddings = EmbeddingSystem()
        results = embeddings.semantic_search("artificial intelligence", k=5)
        
        print(f"\n Test search for 'artificial intelligence':")
        for article_id, score in results:
            faiss_id = embeddings.article_id_to_faiss_id.get(article_id)
            if faiss_id is None:
                print(f"   {score:.3f} - (missing metadata for {article_id})")
                continue
            meta = embeddings.id_to_metadata.get(faiss_id, {})
            title = (meta.get("title") or "?")[:60]
            print(f"   {score:.3f} - {title}...")

    else:
        print(" No articles to index. Run ingestion first!")