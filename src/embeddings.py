import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from .data_models import Article


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    name: str
    model_path: str
    dimension: int
    description: str
    is_news_specific: bool = False

class EmbeddingSystem:
    """Handles embeddings and semantic search using FAISS with multi-model support"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "db/faiss.index",
                 metadata_path: str = "db/faiss_metadata.pkl"):
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Available embedding models
        self.available_models = {
            "all-MiniLM-L6-v2": EmbeddingModelConfig(
                name="all-MiniLM-L6-v2",
                model_path="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
                description="Fast general-purpose model"
            ),
            "all-mpnet-base-v2": EmbeddingModelConfig(
                name="all-mpnet-base-v2", 
                model_path="sentence-transformers/all-mpnet-base-v2",
                dimension=768,
                description="High-quality general-purpose model"
            ),
            "news-similarity": EmbeddingModelConfig(
                name="news-similarity",
                model_path="Blablablab/newsSimilarity",
                dimension=768,
                description="News-specific similarity model",
                is_news_specific=True
            ),
            "paraphrase-multilingual": EmbeddingModelConfig(
                name="paraphrase-multilingual",
                model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                dimension=384,
                description="Multilingual model"
            ),
            "msmarco-distilbert": EmbeddingModelConfig(
                name="msmarco-distilbert",
                model_path="sentence-transformers/msmarco-distilbert-base-v4",
                dimension=768,
                description="MS MARCO fine-tuned for search"
            )
        }
        
        # Initialize primary model
        self.primary_model_name = model_name
        self.models: Dict[str, SentenceTransformer] = {}
        self.current_model = self._load_model(model_name)
        
        # FAISS index
        self.index = None
        self.id_to_metadata = {}
        self.article_id_to_faiss_id = {}
        
        # Load existing index
        self.load_index()
    
    def _load_model(self, model_name: str) -> SentenceTransformer:
        """Load a specific embedding model"""
        if model_name in self.models:
            return self.models[model_name]
        
        config = self.available_models[model_name]
        print(f"Loading embedding model: {config.description}")
        
        try:
            model = SentenceTransformer(config.model_path)
            self.models[model_name] = model
            return model
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            # Fallback to default model
            if model_name != "all-MiniLM-L6-v2":
                return self._load_model("all-MiniLM-L6-v2")
            raise
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different embedding model"""
        if model_name not in self.available_models:
            print(f"Model {model_name} not available")
            return False
        
        try:
            self.current_model = self._load_model(model_name)
            self.primary_model_name = model_name
            print(f"Switched to model: {self.available_models[model_name].description}")
            return True
        except Exception as e:
            print(f"Failed to switch to model {model_name}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about available models"""
        return {name: config.description for name, config in self.available_models.items()}
    
    def encode_text(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """Encode text using specified model or current model"""
        model = self.current_model if model_name is None else self._load_model(model_name)
        return model.encode(text, convert_to_numpy=True)
    
    def encode_texts(self, texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
        """Encode multiple texts efficiently"""
        model = self.current_model if model_name is None else self._load_model(model_name)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def multi_model_search(self, 
                          query: str, 
                          models: List[str], 
                          k: int = 10,
                          fusion_method: str = "weighted_average") -> List[Tuple[str, float]]:
        """
        Perform search using multiple models and fuse results.
        
        Args:
            query: Search query
            models: List of model names to use
            k: Number of results per model
            fusion_method: "weighted_average", "rank_fusion", or "max_score"
        """
        all_results = {}
        
        for model_name in models:
            if model_name not in self.available_models:
                continue
            
            results = self.semantic_search(query, k=k, model_name=model_name)
            
            if fusion_method == "weighted_average":
                # Weight news-specific models higher
                weight = 1.5 if self.available_models[model_name].is_news_specific else 1.0
                for article_id, score in results:
                    if article_id in all_results:
                        all_results[article_id] = (all_results[article_id] + score * weight) / 2
                    else:
                        all_results[article_id] = score * weight
            
            elif fusion_method == "rank_fusion":
                # Reciprocal rank fusion
                for rank, (article_id, score) in enumerate(results):
                    rrf_score = 1.0 / (rank + 1)
                    if article_id in all_results:
                        all_results[article_id] += rrf_score
                    else:
                        all_results[article_id] = rrf_score
            
            elif fusion_method == "max_score":
                # Take maximum score across models
                for article_id, score in results:
                    if article_id in all_results:
                        all_results[article_id] = max(all_results[article_id], score)
                    else:
                        all_results[article_id] = score
        
        # Sort by score and return top k
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
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
                "urgency_score": article.urgency_score,
                "url": article.url
            }
            
            self.article_id_to_faiss_id[article.id] = faiss_id
        
        print(f" Added {len(valid_articles)} articles to semantic index")
        return len(valid_articles)
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 10,
                       score_threshold: float = 0.0,
                       model_name: Optional[str] = None) -> List[Tuple[str, float]]:
        """Semantic search using FAISS, provides min k candidates that match given threshold
        If no threshold specified, default filters out negative cosine similarities,
        ensuring only semantically similar (not opposite) articles are returned."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.encode_text(query, model_name).astype('float32', copy=False).reshape(1, -1)
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
            self.index = faiss.IndexFlatIP(self.current_model.get_sentence_embedding_dimension())
            self.id_to_metadata = {}
            self.article_id_to_faiss_id = {}

    
    def get_stats(self) -> dict:
        """Get embedding system statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.current_model.get_sentence_embedding_dimension(),
            "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "unknown",
            "index_type": "IndexFlatIP",
            "metadata_entries": len(self.id_to_metadata)
        }
    
    def rebuild_index_from_db(self, db):
        """Rebuild the entire index from database articles"""
        print(" Rebuilding semantic index from database...")
        
        # Clear existing index
        self.index = faiss.IndexFlatIP(self.current_model.get_sentence_embedding_dimension())
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
    # Deferred import to avoid circular dependency at module import time
    from .storage import ArticleDB  
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