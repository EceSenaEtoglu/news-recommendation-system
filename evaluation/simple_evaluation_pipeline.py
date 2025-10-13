"""
Simplified Evaluation Pipeline for News Recommendation System
Focused on MRR and Hit@K reporting with 3 config variants and 2 evaluation types.
Interview-friendly version of the comprehensive evaluation pipeline.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import SPICED loader functions
from evaluation.spiced_data.spiced_loader import load_combined, print_dataset_info

# Import recommendation system components
from src.recommendation_system import RecommendationSystem
from src.config import RAGConfig
from src.storage import ArticleDB
from src.embeddings import EmbeddingSystem
from src.data_models import Article


class SimpleNewsEvaluator:
    """Simplified evaluator focused on MRR and Hit@K reporting."""
    
    def __init__(self, n_test_pairs: int = 10, k_recommendations: int = 5, k_diversity: int = 10):
        """Initialize the evaluator.
        
        Args:
            n_test_pairs: Number of test pairs to evaluate (default: 10)
            k_recommendations: Number of recommendations to retrieve (default: 5)
            k_diversity: Number of recommendations for diversity evaluation (default: 10)
        """
        self.test_data = None
        self.recommendation_systems = {}
        self.system_initialized = False
        
        # Parameters
        self.n_test_pairs = n_test_pairs
        self.k_recommendations = k_recommendations
        self.k_diversity = k_diversity
        
        # Only 3 key config variants for interviews
        self.configs = self._create_3_configs()
        
        # Retrieval cache for performance
        self.retrieval_cache = {}
    
    def _create_3_configs(self) -> Dict[str, RAGConfig]:
        """Create 3 key RAGConfig variants for testing."""
        configs = {}
        
        # Basic config - minimal features
        configs['basic'] = RAGConfig(
            enable_graph_rag=False,
            enable_cross_encoder=False,
            BM25_K=100,
            DENSE_K=100,
            POOL_K=150,
            RRF_K=30,
            topic_overlap_boost=0.05,
            max_topic_bonus=0.2,
            entity_boost_weight=0.0,
            personalization_weight=0.0
        )
        
        # Enhanced config - with graph RAG
        configs['enhanced'] = RAGConfig(
            enable_graph_rag=True,
            enable_cross_encoder=False,
            BM25_K=200,
            DENSE_K=200,
            POOL_K=300,
            RRF_K=60,
            topic_overlap_boost=0.1,
            max_topic_bonus=0.5,
            entity_boost_weight=0.15,
            personalization_weight=0.1
        )
        
        # Full config - complete pipeline
        configs['full'] = RAGConfig(
            enable_graph_rag=True,
            enable_cross_encoder=True,
            BM25_K=250,
            DENSE_K=250,
            POOL_K=350,
            CE_K=120,
            RRF_K=70,
            topic_overlap_boost=0.12,
            max_topic_bonus=0.6,
            entity_boost_weight=0.1,
            personalization_weight=0.1
        )
        
        return configs
    
    def load_test_data(self) -> bool:
        """Load only test data from SPICED dataset."""
        try:
            print("Loading SPICED test data...")
            
            spiced_path = "spiced_data/spiced.csv"
            train_size = 0.7
            seed = 42
            n_hard = 500  # Small dataset for interviews
            
            # Load only test data
            self.test_data = load_combined('test', path=spiced_path, train_size=train_size, seed=seed, n_hard=n_hard//32)
            print_dataset_info(self.test_data, "Test Data")
            
            # Verify required columns
            required = {"text_1", "text_2", "URL_1", "URL_2", "Type", "label"}
            missing = required - set(self.test_data.columns)
            if missing:
                raise ValueError(f"SPICED data missing columns: {missing}")
            
            print(f"Loaded {len(self.test_data)} test pairs")
            return True
            
        except Exception as e:
            print(f"Failed to load test data: {e}")
            return False
    
    def initialize_systems(self) -> bool:
        """Initialize RecommendationSystem instances with 3 configs."""
        try:
            print("Initializing recommendation systems...")
            
            # Use test database for evaluation
            test_db_path = "test_db/test_spiced.db"
            if not os.path.exists(test_db_path):
                print("Creating test database...")
                from evaluation.test_database import SPICEDTestDatabase
                test_db = SPICEDTestDatabase(test_db_path)
                if not test_db.create_test_database():
                    raise Exception("Failed to create test database")
            
            # Initialize components
            db = ArticleDB(db_path=test_db_path)
            
            # Use test-specific FAISS paths
            faiss_index_path = "test_db/test_faiss.index"
            faiss_metadata_path = "test_db/test_faiss_metadata.pkl"
            
            # Remove existing FAISS files if they exist
            if os.path.exists(faiss_index_path):
                os.remove(faiss_index_path)
            if os.path.exists(faiss_metadata_path):
                os.remove(faiss_metadata_path)
            
            # Initialize embeddings
            embeddings = EmbeddingSystem(
                index_path=faiss_index_path,
                metadata_path=faiss_metadata_path
            )
            
            # Rebuild FAISS index
            print("Rebuilding FAISS index...")
            embeddings.rebuild_index_from_db(db)
            
            # Initialize RecommendationSystem instances
            for config_name, config in self.configs.items():
                print(f"  Initializing {config_name} system...")
                self.recommendation_systems[config_name] = RecommendationSystem(
                    db=db,
                    embeddings=embeddings,
                    config=config
                )
            
            self.system_initialized = True
            print(f"Initialized {len(self.recommendation_systems)} systems")
            return True
            
        except Exception as e:
            print(f"System initialization failed: {e}")
            return False
    
    # TODO, for better evaluation, title can be generated via LLM
    def create_article(self, text: str, url: str, topic: str = "GENERAL") -> Article:
        """Create an Article object from text data."""
        from collections import namedtuple
        Source = namedtuple("Source", ["name", "category"])
        
        # Extract title from first sentence
        title = text.split('.')[0][:100] if text else "Article"
        
        return Article(
            id=url,
            title=title,
            description=text[:200] if text else "",
            content=text,
            url=url,
            source=Source(name="SPICED Source", category=topic.upper()),
            published_at=datetime.now(),
            topics=[topic]
        )
    
    def get_recommendations(self, query_article: Article, config_name: str, limit: int = None) -> List[Dict]:
        """Get recommendations using specified config."""
        if not self.system_initialized or config_name not in self.recommendation_systems:
            return []
        
        # Use default limit if not specified
        if limit is None:
            limit = self.k_recommendations
        
        # Check cache
        cache_key = f"{query_article.id}_{config_name}_{limit}"
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        try:
            # Get recommendations
            recommendation_system = self.recommendation_systems[config_name]
            recommendations = recommendation_system.recommend_for_article(query_article, k=limit)
            
            # Convert to evaluation format
            eval_results = []
            for i, (article, score) in enumerate(recommendations):
                eval_results.append({
                    'text': (article.title or "") + " " + (article.description or ""),
                    'url': getattr(article, 'url', None),
                    'topic': getattr(article, 'topics', [None])[0] if article.topics else None,
                    'score': score,
                    'rank': i
                })
            
            # Cache results
            self.retrieval_cache[cache_key] = eval_results
            return eval_results
            
        except Exception as e:
            print(f"Error getting recommendations with {config_name}: {e}")
            return []
    
    def find_rank(self, recommendations: List[Dict], expected_url: str) -> Optional[int]:
        """Find the rank of expected URL in recommendations."""
        for i, rec in enumerate(recommendations):
            if rec.get('url') == expected_url:
                return i
        return None
    
    def calculate_mrr(self, ranks: List[Optional[int]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        rr = [(1.0/(r+1)) for r in ranks if r is not None]
        return float(np.mean(rr)) if rr else 0.0
    
    def calculate_hit_at_k(self, ranks: List[Optional[int]], k: int) -> float:
        """Calculate Hit@K metric."""
        hits = [1.0 if (r is not None and r < k) else 0.0 for r in ranks]
        return float(np.mean(hits)) if hits else 0.0
    
    def evaluate_similarity(self) -> Dict:
        """Main evaluation - MRR and Hit@K reporting."""
        print("Evaluating Similarity Detection...")
        
        # Use N positive pairs for evaluation
        test_pairs = self.test_data[self.test_data['label'] == 1].head(self.n_test_pairs)
        print(f"Testing with {len(test_pairs)} positive pairs")
        
        results = {}
        for config_name in self.configs.keys():
            print(f"  Evaluating {config_name} config...")
            ranks = []
            
            for _, pair in test_pairs.iterrows():
                # Create query article
                query_article = self.create_article(
                    pair['text_1'], 
                    pair['URL_1'], 
                    pair['Type']
                )
                
                # Get recommendations
                recommendations = self.get_recommendations(query_article, config_name)
                
                # Debug: Print first few recommendations
                if len(ranks) < 2:  # Only for first 2 queries to avoid spam
                    print(f"    Query: {pair['text_1'][:50]}...")
                    print(f"    Expected URL: {pair['URL_2']}")
                    print(f"    Got {len(recommendations)} recommendations")
                    if recommendations:
                        print(f"    First rec URL: {recommendations[0].get('url', 'No URL')}")
                
                # Find rank of expected URL
                rank = self.find_rank(recommendations, pair['URL_2'])
                ranks.append(rank)
            
            # Calculate MRR and Hit@K (using 0-based ranks internally)
            # Convert to 1-based for reporting
            ranks_1based = [r + 1 for r in ranks if r is not None] if any(r is not None for r in ranks) else []
            
            results[config_name] = {
                'MRR': self.calculate_mrr(ranks),  # MRR uses 0-based internally (1/(rank+1))
                'Hit@1': self.calculate_hit_at_k(ranks, 1),  # Hit@1 uses 0-based internally
                'Hit@3': self.calculate_hit_at_k(ranks, 3),  # Hit@3 uses 0-based internally
                f'Hit@{self.k_recommendations}': self.calculate_hit_at_k(ranks, self.k_recommendations),
                'avg_rank': np.mean(ranks_1based) if ranks_1based else None,  # Report 1-based average rank
                'found_count': sum(1 for r in ranks if r is not None)
            }
        
        return results
    
    def evaluate_diversity(self) -> Dict:
        """Simple diversity evaluation."""
        print("Evaluating Diversity...")
        
        diversity_query = "diverse news from different topics"
        query_article = self.create_article(diversity_query, "diversity_query", "GENERAL")
        
        results = {}
        for config_name in self.configs.keys():
            recommendations = self.get_recommendations(query_article, config_name, limit=self.k_diversity)
            
            # Count unique topics
            topics = [r.get('topic') for r in recommendations if r.get('topic')]
            unique_topics = len(set(topics))
            
            results[config_name] = {
                'topic_diversity': unique_topics / 7,  # 7 topics in SPICED
                'unique_topics': unique_topics,
                'total_recommendations': len(recommendations)
            }
        
        return results
    
    def compare_configs(self, similarity_results: Dict) -> Dict:
        """Simple config comparison."""
        
        # Rank by MRR
        mrr_ranking = sorted(
            similarity_results.items(), 
            key=lambda x: x[1]['MRR'], 
            reverse=True
        )
        
        # Rank by Hit@3
        hit3_ranking = sorted(
            similarity_results.items(),
            key=lambda x: x[1]['Hit@3'],
            reverse=True
        )
        
        return {
            'best_mrr': mrr_ranking[0][0] if mrr_ranking else None,
            'best_hit3': hit3_ranking[0][0] if hit3_ranking else None,
            'mrr_scores': {name: results['MRR'] for name, results in similarity_results.items()},
            'hit3_scores': {name: results['Hit@3'] for name, results in similarity_results.items()},
            'mrr_ranking': mrr_ranking,
            'hit3_ranking': hit3_ranking
        }
    
    def run_evaluation(self) -> Dict:
        """Main evaluation pipeline - 3 steps only."""
        print("Starting Simple Evaluation Pipeline")
        print("=" * 50)
        
        # Step 1: Load test data
        if not self.load_test_data():
            return {}
        
        # Step 2: Initialize systems
        if not self.initialize_systems():
            print("System initialization failed. Returning empty results.")
            return {}
        
        # Step 3: Run evaluations
        print("\nRunning Evaluations...")
        
        # Type 1: Similarity Detection (MRR + Hit@K)
        similarity_results = self.evaluate_similarity()
        
        # Type 2: Diversity (Simple metric)
        diversity_results = self.evaluate_diversity()
        
        # Config comparison
        config_comparison = self.compare_configs(similarity_results)
        
        return {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'test_pairs': len(self.test_data),
                'configs_tested': list(self.configs.keys()),
                'system_type': 'RecommendationSystem'
            },
            'similarity': similarity_results,
            'diversity': diversity_results,
            'config_comparison': config_comparison
        }
    
    def print_results(self, results: Dict):
        """Clean, interview-friendly output."""
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        # Similarity Results
        print("\nSIMILARITY DETECTION (MRR & Hit@K):")
        print("-" * 40)
        for config, metrics in results['similarity'].items():
            hit_k_key = f'Hit@{self.k_recommendations}'
            print(f"  {config:8}: MRR={metrics['MRR']:.3f}, Hit@1={metrics['Hit@1']:.3f}, Hit@3={metrics['Hit@3']:.3f}, {hit_k_key}={metrics[hit_k_key]:.3f}")
            avg_rank_str = f"{metrics['avg_rank']:.1f}" if metrics['avg_rank'] is not None else "N/A"
            print(f"           Found: {metrics['found_count']}/{self.n_test_pairs}, Avg Rank: {avg_rank_str} (1-based)")
        
        # Best Configs
        comparison = results['config_comparison']
        print(f"\nBEST CONFIGS:")
        print("-" * 20)
        print(f"  Best MRR:  {comparison['best_mrr']} ({comparison['mrr_scores'][comparison['best_mrr']]:.3f})")
        print(f"  Best Hit@3: {comparison['best_hit3']} ({comparison['hit3_scores'][comparison['best_hit3']]:.3f})")
        
        # Diversity
        print(f"\nDIVERSITY (Topic Coverage):")
        print("-" * 30)
        for config, metrics in results['diversity'].items():
            print(f"  {config:8}: {metrics['unique_topics']}/7 topics ({metrics['topic_diversity']:.2f})")
        
        # Summary
        print(f"\nSUMMARY:")
        print("-" * 15)
        print(f"  Test Pairs: {results['metadata']['test_pairs']}")
        print(f"  Configs: {', '.join(results['metadata']['configs_tested'])}")
        print(f"  Best Overall: {comparison['best_mrr']} (highest MRR)")
    
    def save_results(self, results: Dict, filename: str = 'simple_evaluation_results.json'):
        """Save evaluation results."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")


def main():
    """Main function to run simple evaluation."""
    # Initialize with custom parameters
    evaluator = SimpleNewsEvaluator(
        n_test_pairs=10,      # Number of test pairs to evaluate
        k_recommendations=5,  # Number of recommendations to retrieve
        k_diversity=10        # Number of recommendations for diversity
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        # Print results
        evaluator.print_results(results)
        
        # Save results
        evaluator.save_results(results)
        
        print("\nSimple evaluation completed successfully!")
    else:
        print("Evaluation failed!")


if __name__ == "__main__":
    main()
