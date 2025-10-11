"""
Proper Evaluation Pipeline for News Recommendation System
Using SPICED dataset and comprehensive evaluation metrics with proper train/test splits.
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
from collections import defaultdict

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import SPICED loader functions
from evaluation.spiced_data.spiced_loader import load_combined, load_intertopic, load_intratopic_and_hard_examples, print_dataset_info

class NewsRecommendationEvaluator:
    """Comprehensive evaluator for news recommendation system."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.spiced_data = None
        self.train_data = None
        self.test_data = None
        self.intratopic_train = None
        self.intratopic_test = None
        self.intertopic_train = None
        self.intertopic_test = None
        self.hard_train = None
        self.hard_test = None
        self.evaluation_results = {}
        
        # Real system components
        self.db = None
        self.embeddings = None
        self.retriever = None
        self.system_initialized = False
        
        # Retrieval cache for performance
        self.retrieval_cache = {}
        
    def load_spiced_dataset(self) -> bool:
        """Load SPICED dataset with proper train/test splits."""
        try:
            print("Loading SPICED dataset with proper splits...")

            # Point to SPICED data in spiced_data folder
            spiced_path = "spiced_data/spiced.csv"
            train_size = 0.7
            seed = 42
            n_hard = 3000

            # Combined (labeled) - heavily reduced dataset size
            self.train_data = load_combined('train', path=spiced_path, train_size=train_size, seed=seed, n_hard=n_hard//16)
            self.test_data  = load_combined('test',  path=spiced_path, train_size=train_size, seed=seed, n_hard=n_hard//16)
            print_dataset_info(self.train_data, "Combined Train")
            print_dataset_info(self.test_data,  "Combined Test")

            # Intra-topic + Hard (labeled) - heavily reduced dataset size
            self.intratopic_train, self.hard_train = load_intratopic_and_hard_examples(
                'train', path=spiced_path, train_size=train_size, seed=seed, n_hard=n_hard//16
            )
            self.intratopic_test,  self.hard_test  = load_intratopic_and_hard_examples(
                'test',  path=spiced_path, train_size=train_size, seed=seed, n_hard=n_hard//16
            )
            print_dataset_info(self.intratopic_train, "Intratopic Train")
            print_dataset_info(self.intratopic_test,  "Intratopic Test")
            print_dataset_info(self.hard_train,      "Hard Examples Train")
            print_dataset_info(self.hard_test,       "Hard Examples Test")

            # Inter-topic (labeled) - heavily reduced dataset size
            self.intertopic_train = load_intertopic('train', path=spiced_path, train_size=train_size, seed=seed)
            self.intertopic_test  = load_intertopic('test',  path=spiced_path, train_size=train_size, seed=seed)
            print_dataset_info(self.intertopic_train, "Intertopic Train")
            print_dataset_info(self.intertopic_test,  "Intertopic Test")

            # default eval split
            self.spiced_data = self.test_data

            # Sanity: we expect a label column now
            required = {"text_1","text_2","URL_1","URL_2","Type","label"}
            missing = required - set(self.spiced_data.columns)
            if missing:
                raise ValueError(f"SPICED split missing columns: {missing}")

            print("\nSPICED dataset loaded successfully!")
            print(f"Total test pairs for evaluation: {len(self.spiced_data)}")
            return True

        except Exception as e:
            print(f"Failed to load SPICED dataset: {e}")
            return False

    
    def create_evaluation_queries(self) -> List[Dict]:
        """Create evaluation queries from SPICED positives."""
        if self.spiced_data is None:
            return []

        queries = []

        # Similarity: use only positive (label==1) pairs - heavily reduced
        positives = self.spiced_data[self.spiced_data["label"] == 1].head(30)  # Limit to 30 positive pairs
        for idx, row in positives.iterrows():
            queries.append({
                'query_id': f"spiced_{idx}",
                'query_text': row['text_1'],       # full text
                'expected_url': row['URL_2'],      # match by URL
                'expected_text': row['text_2'],    # optional, for debugging
                'topic': row['Type'],
                'ground_truth_type': 'spiced_similarity',
                'evaluation_type': 'similarity_detection',
                'pair_type': row.get('pair_type', 'positive'),  # Track metadata
                'tfidf_cosine': row.get('tfidf_cosine', None)   # Track TF-IDF score if available
            })

        # Topic-based queries (balanced across topics) - heavily reduced
        topic_groups = self.spiced_data.groupby('Type')
        min_samples_per_topic = 2  # Reduced from 5
        max_urls_per_topic = 3     # Reduced from 10
        
        for topic, group in topic_groups:
            if len(group) >= min_samples_per_topic:
                # Use only positive pairs for topic queries
                topic_positives = group[group['label'] == 1] if 'label' in group.columns else group
                
                if len(topic_positives) > 0:
                    # Balance URLs across text_1 and text_2
                    urls_1 = topic_positives['URL_1'].tolist()[:max_urls_per_topic//2]
                    urls_2 = topic_positives['URL_2'].tolist()[:max_urls_per_topic//2]
                    expected_urls = urls_1 + urls_2
                    
                    queries.append({
                        'query_id': f"topic_{topic}",
                        'query_text': f"news about {topic}",
                        'expected_urls': expected_urls,
                        'topic': topic,
                        'ground_truth_type': 'topic_match',
                        'evaluation_type': 'topic_retrieval',
                        'topic_sample_size': len(topic_positives)
                    })

        # Diversity query
        queries.append({
            'query_id': 'diversity_cross_topic',
            'query_text': 'diverse news from different topics',
            'expected_topics': self.spiced_data['Type'].unique().tolist(),
            'ground_truth_type': 'diversity',
            'evaluation_type': 'diversity_measurement'
        })

        return queries

    def create_negative_evaluation_queries(self) -> List[Dict]:
        """Create evaluation queries from SPICED negative pairs."""
        if self.spiced_data is None:
            return []

        queries = []

        # Negative pairs: use only negative (label==0) pairs - heavily reduced
        negatives = self.spiced_data[self.spiced_data["label"] == 0].head(30)  # Limit to 30 negative pairs
        for idx, row in negatives.iterrows():
            queries.append({
                'query_id': f"negative_{idx}",
                'query_text': row['text_1'],       # full text
                'expected_url': row['URL_2'],      # should NOT be found
                'expected_text': row['text_2'],    # optional, for debugging
                'topic': row['Type'],
                'ground_truth_type': 'spiced_negative',
                'evaluation_type': 'negative_detection',
                'pair_type': row.get('pair_type', 'unknown')  # Track metadata
            })

        return queries

    
    
    
    def evaluate_similarity_detection(self, retriever, queries: List[Dict]) -> Dict:
        """Evaluate similarity detection with rank metrics and binary classification."""
        print("Evaluating Similarity Detection...")

        results = {
            'queries': [],
            'topic_performance': {},
            'overall_metrics': {},
            'binary_classification': {}
        }

        similarity_queries = [q for q in queries if q['evaluation_type'] == 'similarity_detection']

        ranks = []
        per_topic = defaultdict(list)
        binary_predictions = []
        binary_labels = []

        for query in similarity_queries[:15]:  # cap if needed
            try:
                start = time.time()
                real_results = self._real_retrieval_sync(query['query_text'], limit=5)
                qtime = time.time() - start

                r = self._rank_of_expected(real_results, query['expected_url'])
                ranks.append(r)
                per_topic[query['topic']].append(r)

                # Binary classification: is the expected URL in top-k?
                binary_pred = 1 if (r is not None and r < 5) else 0  # Top-5 threshold
                binary_label = 1  # All similarity queries are positive pairs
                binary_predictions.append(binary_pred)
                binary_labels.append(binary_label)

                results['queries'].append({
                    'query_id': query['query_id'],
                    'topic': query['topic'],
                    'rank': None if r is None else int(r),
                    'hit@1': 1 if (r is not None and r < 1) else 0,
                    'hit@3': 1 if (r is not None and r < 3) else 0,
                    'hit@10': 1 if (r is not None and r < 10) else 0,
                    'binary_prediction': binary_pred,
                    'query_time': qtime,
                    'retrieved_count': len(real_results)
                })
            except Exception as e:
                print(f"Error evaluating query {query['query_id']}: {e}")

        if results['queries']:
            results['overall_metrics'] = {
                'MRR': self._mrr_from_ranks(ranks),
                'Hit@1': self._hits_at_k(ranks, 1),
                'Hit@3': self._hits_at_k(ranks, 3),
                'Hit@10': self._hits_at_k(ranks, 10),
                'avg_query_time': float(np.mean([r['query_time'] for r in results['queries']]))
            }

            # Binary classification metrics
            if binary_predictions and binary_labels:
                results['binary_classification'] = self._calculate_binary_metrics(
                    binary_predictions, binary_labels
                )

        topic_metrics = {}
        for t, tranks in per_topic.items():
            topic_metrics[t] = {
                'MRR': self._mrr_from_ranks(tranks),
                'Hit@3': self._hits_at_k(tranks, 3),
                'n': len(tranks)
            }
        results['topic_performance'] = topic_metrics

        return results

    def evaluate_negative_detection(self, retriever, queries: List[Dict]) -> Dict:
        """Evaluate negative pair detection (should NOT find similar articles)."""
        print("Evaluating Negative Detection...")

        results = {
            'queries': [],
            'pair_type_performance': {},
            'overall_metrics': {},
            'false_positive_analysis': {}
        }

        negative_queries = [q for q in queries if q['evaluation_type'] == 'negative_detection']

        ranks = []
        per_pair_type = defaultdict(list)
        false_positives = []

        for query in negative_queries[:15]:  # cap if needed
            try:
                start = time.time()
                real_results = self._real_retrieval_sync(query['query_text'], limit=5)
                qtime = time.time() - start

                r = self._rank_of_expected(real_results, query['expected_url'])
                ranks.append(r)
                pair_type = query.get('pair_type', 'unknown')
                per_pair_type[pair_type].append(r)

                # False positive: found a negative pair in top-k
                fp = 1 if (r is not None and r < 5) else 0  # Top-5 threshold
                false_positives.append(fp)

                results['queries'].append({
                    'query_id': query['query_id'],
                    'topic': query['topic'],
                    'pair_type': pair_type,
                    'rank': None if r is None else int(r),
                    'false_positive': fp,
                    'query_time': qtime,
                    'retrieved_count': len(real_results)
                })
            except Exception as e:
                print(f"Error evaluating negative query {query['query_id']}: {e}")

        if results['queries']:
            results['overall_metrics'] = {
                'false_positive_rate': np.mean(false_positives),
                'avg_query_time': float(np.mean([r['query_time'] for r in results['queries']])),
                'total_negatives': len(negative_queries)
            }

            # Pair type analysis
            for pt, ptranks in per_pair_type.items():
                pt_fps = [1 if (r is not None and r < 5) else 0 for r in ptranks]
                results['pair_type_performance'][pt] = {
                    'false_positive_rate': np.mean(pt_fps),
                    'avg_rank': np.mean([r for r in ptranks if r is not None]) if any(r is not None for r in ptranks) else None,
                    'count': len(ptranks)
                }

        return results

    
    def evaluate_topic_retrieval(self, retriever, queries: List[Dict]) -> Dict:
        """Evaluate topic-based retrieval performance."""
        print("Evaluating Topic Retrieval...")
        
        results = {
            'topic_queries': [],
            'topic_metrics': {}
        }
        
        topic_queries = [q for q in queries if q['evaluation_type'] == 'topic_retrieval']
        
        for query in topic_queries:
            try:
                # Use real system for topic retrieval
                start_time = time.time()
                real_results = self._real_retrieval_sync(query['query_text'], limit=5)
                query_time = time.time() - start_time
                
                # Calculate topic relevance
                topic_relevance = self._calculate_topic_relevance(
                    query.get('expected_urls', []),
                    real_results,
                    query['topic']
                )
                
                result = {
                    'query_id': query['query_id'],
                    'topic': query['topic'],
                    'topic_relevance': topic_relevance,
                    'query_time': query_time,
                    'retrieved_count': len(real_results)
                }
                
                results['topic_queries'].append(result)
                
            except Exception as e:
                print(f"Error evaluating topic query {query['query_id']}: {e}")
        
        # Calculate topic-specific metrics
        for topic in self.spiced_data['Type'].unique():
            topic_results = [r for r in results['topic_queries'] if r['topic'] == topic]
            if topic_results:
                scores = [r['topic_relevance'] for r in topic_results]
                results['topic_metrics'][topic] = {
                    'mean_relevance': np.mean(scores),
                    'std_relevance': np.std(scores),
                    'query_count': len(topic_results)
                }
        
        return results
    
    def evaluate_diversity(self, retriever, queries: List[Dict]) -> Dict:
        """Evaluate recommendation diversity."""
        print("Evaluating Diversity...")
        
        results = {
            'diversity_queries': [],
            'diversity_metrics': {}
        }
        
        diversity_queries = [q for q in queries if q['evaluation_type'] == 'diversity_measurement']
        
        for query in diversity_queries:
            try:
                # Use real system for diversity retrieval
                real_results = self._real_retrieval_sync(query['query_text'], limit=20)
                
                # Calculate diversity metrics
                topic_diversity = self._calculate_topic_diversity(real_results)
                content_diversity = self._calculate_content_diversity(real_results)
                
                result = {
                    'query_id': query['query_id'],
                    'topic_diversity': topic_diversity,
                    'content_diversity': content_diversity,
                    'overall_diversity': (topic_diversity + content_diversity) / 2
                }
                
                results['diversity_queries'].append(result)
                
            except Exception as e:
                print(f"Error evaluating diversity query {query['query_id']}: {e}")
        
        # Calculate overall diversity metrics
        if results['diversity_queries']:
            all_diversity = [r['overall_diversity'] for r in results['diversity_queries']]
            results['diversity_metrics'] = {
                'mean_diversity': np.mean(all_diversity),
                'std_diversity': np.std(all_diversity),
                'min_diversity': np.min(all_diversity),
                'max_diversity': np.max(all_diversity)
            }
        
        return results
    
    
    

    def evaluate_difficulty_levels(self) -> Dict:
        """Evaluate performance across different difficulty levels using SPICED negatives."""
        print("Evaluating Difficulty Levels...")
        
        results = {
            'intratopic_negatives': {},
            'intertopic_negatives': {},
            'hard_negatives': {},
            'difficulty_comparison': {}
        }
        
        # Evaluate each difficulty level
        if self.intratopic_test is not None and len(self.intratopic_test) > 0:
            results['intratopic_negatives'] = self._evaluate_difficulty_subset(
                self.intratopic_test, 'intratopic', max_samples=50
            )
        
        if self.intertopic_test is not None and len(self.intertopic_test) > 0:
            results['intertopic_negatives'] = self._evaluate_difficulty_subset(
                self.intertopic_test, 'intertopic', max_samples=50
            )
        
        if self.hard_test is not None and len(self.hard_test) > 0:
            results['hard_negatives'] = self._evaluate_difficulty_subset(
                self.hard_test, 'hard', max_samples=50
            )
        
        # Compare difficulty levels
        difficulty_scores = {}
        for level, data in results.items():
            if isinstance(data, dict) and 'avg_score' in data:
                difficulty_scores[level] = data['avg_score']
        
        if difficulty_scores:
            results['difficulty_comparison'] = {
                'scores': difficulty_scores,
                'ranking': sorted(difficulty_scores.items(), key=lambda x: x[1], reverse=True),
                'hardest_level': min(difficulty_scores.items(), key=lambda x: x[1])[0] if difficulty_scores else None
            }
        
        return results

    def _evaluate_difficulty_subset(self, data: pd.DataFrame, difficulty_type: str, max_samples: int = 10) -> Dict:
        """Evaluate a subset of data for a specific difficulty level."""
        if len(data) == 0:
            return {'avg_score': 0.0, 'total_pairs': 0}
        
        # Sample data for evaluation (heavily reduced)
        sample_size = min(max_samples // 8, len(data))
        sampled_data = data.sample(n=sample_size, random_state=42)
        
        total_score = 0.0
        successful_pairs = 0
        topic_scores = defaultdict(list)
        
        for idx, row in sampled_data.iterrows():
            try:
                query_text = row['text_1'][:200] + "..."
                expected_url = row['URL_2']
                
                real_results = self._real_retrieval_sync(query_text, limit=5)
                rank = self._rank_of_expected(real_results, expected_url)
                
                # For negatives, we want LOW scores (should NOT find similar)
                # But we still calculate the same way for consistency
                similarity_score = 0.0 if rank is None else 1.0 / (rank + 1.0)
                
                total_score += similarity_score
                successful_pairs += 1
                topic_scores[row['Type']].append(similarity_score)
                
            except Exception as e:
                print(f"Error evaluating {difficulty_type} pair {idx}: {e}")
        
        avg_score = total_score / successful_pairs if successful_pairs > 0 else 0.0
        
        # Calculate topic-specific scores
        topic_metrics = {}
        for topic, scores in topic_scores.items():
            topic_metrics[topic] = {
                'mean_score': np.mean(scores),
                'count': len(scores)
            }
        
        return {
            'avg_score': avg_score,
            'total_pairs': successful_pairs,
            'sampled_pairs': sample_size,
            'difficulty_type': difficulty_type,
            'topic_metrics': topic_metrics
        }
    
    
   
    
    def _mock_retrieval(self, query_text: str, expected_text: str = "", limit: int = 10) -> List[Dict]:
        """Trivial mock for when real system isn't initialized or errors."""
        return [{
            'text': expected_text, 
            'url': None, 
            'topic': None, 
            'score': 0.0, 
            'source': 'mock', 
            'article_id': None
        } for _ in range(limit)]
    
    def _calculate_topic_relevance(self, expected_urls: List[str], results: List[Dict], topic: str) -> float:
        """Share of retrieved items whose topic matches the query topic."""
        if not results:
            return 0.0
        match_flags = [1 for r in results if r.get('topic') == topic]
        return float(np.mean(match_flags)) if match_flags else 0.0

    
    def _calculate_topic_diversity(self, results: List[Dict]) -> float:
        if not results:
            return 0.0
        got = {r.get('topic', 'unknown') for r in results}
        total = max(1, len(self.spiced_data['Type'].unique()))
        return len(got) / total

    
    def _calculate_content_diversity(self, results: List[Dict]) -> float:
        """Calculate content diversity score."""
        if not results:
            return 0.0
        
        # Simple content diversity - replace with actual metric
        texts = [result.get('text', '') for result in results]
        unique_words = set()
        for text in texts:
            unique_words.update(text.lower().split())
        
        return len(unique_words) / 1000  # Normalize
    
    def initialize_real_system(self) -> bool:
        """Initialize real system components with test database."""
        try:
            print("Initializing real system components with SPICED test database...")
            
            # Import actual system components
            from storage import ArticleDB
            from embeddings import EmbeddingSystem
            from retrieval import MultiRAGRetriever
            from data_models import SearchQuery
            
            # Use test database for evaluation
            test_db_path = "test_db/test_spiced.db"
            if not os.path.exists(test_db_path):
                print("Test database not found. Creating it...")
                from evaluation.test_database import SPICEDTestDatabase
                test_db = SPICEDTestDatabase(test_db_path)
                if not test_db.create_test_database():
                    raise Exception("Failed to create test database")
            
            # Initialize components with test database
            self.db = ArticleDB(db_path=test_db_path)
            
            # Use test-specific FAISS paths
            faiss_index_path = "test_db/test_faiss.index"
            faiss_metadata_path = "test_db/test_faiss_metadata.pkl"
            
            # Remove existing FAISS files if they exist
            if os.path.exists(faiss_index_path):
                os.remove(faiss_index_path)
            if os.path.exists(faiss_metadata_path):
                os.remove(faiss_metadata_path)
            
            # Initialize embeddings with test paths
            self.embeddings = EmbeddingSystem(
                index_path=faiss_index_path,
                metadata_path=faiss_metadata_path
            )
            
            # Rebuild FAISS index for test database
            print("Rebuilding FAISS index for test database...")
            self.embeddings.rebuild_index_from_db(self.db)
            
            # Initialize retriever with reduced cache
            self.retriever = MultiRAGRetriever(self.db, self.embeddings)
            # Reduce BM25 cache size for faster evaluation
            if hasattr(self.retriever, 'bm25_cache_size'):
                self.retriever.bm25_cache_size = 100  # Reduced from default
            
            # Test system with a simple query
            test_query = SearchQuery(text="test query", limit=5)
            # Note: search is async, so we'll test it during actual evaluation
            
            self.system_initialized = True
            print("Real system components initialized successfully with SPICED test database")
            return True
            
        except Exception as e:
            print(f"System initialization failed: {e}")
            self.system_initialized = False
            return False
    
    def _real_retrieval_sync(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Synchronous wrapper for real retrieval with caching."""
        if not self.system_initialized:
            return self._mock_retrieval(query_text, "")
        
        # Check cache first
        cache_key = f"{query_text}_{limit}"
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        try:
            import asyncio
            from data_models import SearchQuery
            
            # Create search query
            search_query = SearchQuery(text=query_text, limit=limit)
            
            # Run async search
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.retriever.search(search_query))
            finally:
                loop.close()
            
            # Convert to evaluation format
            eval_results = []
            for i, result in enumerate(results):
                article = result.article
                eval_results.append({
                    'text': (article.title or "") + " " + (article.description or ""),
                    'url': getattr(article, 'url', None),
                    'topic': getattr(article, 'topic', None),
                    'score': result.final_score,
                    'source': 'real_system',
                    'article_id': article.id,
                    'relevance_score': result.relevance_score,
                    'freshness_score': result.freshness_score,
                    'personalization_score': result.personalization_score
                })

            # Cache the results
            self.retrieval_cache[cache_key] = eval_results
            return eval_results
            
        except Exception as e:
            print(f"Error in real retrieval: {e}")
            # Fallback to mock
            return self._mock_retrieval(query_text, "")
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation pipeline."""
        print("Starting Comprehensive Evaluation Pipeline")
        print("=" * 60)
        
        # Load SPICED dataset
        if not self.load_spiced_dataset():
            return {}
        
        # Create evaluation queries
        queries = self.create_evaluation_queries()
        print(f"Created {len(queries)} evaluation queries")
        
        # Initialize real system components
        system_ready = self.initialize_real_system()
        if not system_ready:
            print("Using mock evaluation instead")
        
        # Create negative evaluation queries
        negative_queries = self.create_negative_evaluation_queries()
        all_queries = queries + negative_queries

        # Run evaluations
        evaluation_results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_queries': len(all_queries),
                'positive_queries': len(queries),
                'negative_queries': len(negative_queries),
                'spiced_pairs': len(self.spiced_data),
                'topics': list(self.spiced_data['Type'].unique()),
                'train_pairs': len(self.train_data) if self.train_data is not None else 0,
                'test_pairs': len(self.test_data) if self.test_data is not None else 0,
                'positive_ratio': len(self.spiced_data[self.spiced_data['label'] == 1]) / len(self.spiced_data) if len(self.spiced_data) > 0 else 0
            },
            'similarity_detection': self.evaluate_similarity_detection(self.retriever, queries),
            'negative_detection': self.evaluate_negative_detection(self.retriever, negative_queries),
            'topic_retrieval': self.evaluate_topic_retrieval(self.retriever, queries),
            'diversity': self.evaluate_diversity(self.retriever, queries),
            'difficulty_levels': self.evaluate_difficulty_levels(),
            'baseline_comparison': self.evaluate_baselines(all_queries)
        }
        
        # Calculate overall performance
        evaluation_results['overall_performance'] = self._calculate_overall_performance(evaluation_results)
        
        return evaluation_results

    def _rank_of_expected(self, results: List[Dict], expected_url: Optional[str]) -> Optional[int]:
        if not expected_url:
            return None
        for i, r in enumerate(results):
            if r.get('url') == expected_url:
                return i  # 0-based rank
        return None

    def _mrr_from_ranks(self, ranks: List[Optional[int]]) -> float:
        rr = [(1.0/(r+1)) for r in ranks if r is not None]
        return float(np.mean(rr)) if rr else 0.0

    def _hits_at_k(self, ranks: List[Optional[int]], k: int) -> float:
        hits = [1.0 if (r is not None and r < k) else 0.0 for r in ranks]
        return float(np.mean(hits)) if hits else 0.0

    def _calculate_binary_metrics(self, predictions: List[int], labels: List[int]) -> Dict:
        """Calculate binary classification metrics."""
        if not predictions or not labels:
            return {}
        
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

    
    def _calculate_overall_performance(self, results: Dict) -> Dict:
        """Calculate overall performance metrics including difficulty-based evaluations."""
        overall = {
            'similarity_score': 0.0,
            'topic_relevance': 0.0,
            'diversity_score': 0.0,
            'intratopic_score': 0.0,
            'intertopic_score': 0.0,
            'hard_examples_score': 0.0,
            'overall_score': 0.0
        }
        
        # Extract metrics from each evaluation
        if 'similarity_detection' in results and 'overall_metrics' in results['similarity_detection']:
            sim = results['similarity_detection']['overall_metrics']
            # prefer MRR / Hit@3
            overall['similarity_score'] = 0.5 * sim.get('MRR', 0.0) + 0.5 * sim.get('Hit@3', 0.0)
        
        # Add negative detection metrics
        if 'negative_detection' in results and 'overall_metrics' in results['negative_detection']:
            neg = results['negative_detection']['overall_metrics']
            overall['negative_detection_score'] = 1.0 - neg.get('false_positive_rate', 0.0)  # Lower FP rate is better

        
        if 'topic_retrieval' in results and 'topic_metrics' in results['topic_retrieval']:
            topic_scores = [metrics['mean_relevance'] for metrics in results['topic_retrieval']['topic_metrics'].values()]
            overall['topic_relevance'] = np.mean(topic_scores) if topic_scores else 0.0
        
        if 'diversity' in results and 'diversity_metrics' in results['diversity']:
            overall['diversity_score'] = results['diversity']['diversity_metrics'].get('mean_diversity', 0.0)
        
        # Extract difficulty-based metrics from difficulty_levels
        if 'difficulty_levels' in results:
            diff_levels = results['difficulty_levels']
            if 'intratopic_negatives' in diff_levels and 'avg_score' in diff_levels['intratopic_negatives']:
                overall['intratopic_score'] = diff_levels['intratopic_negatives']['avg_score']
            if 'intertopic_negatives' in diff_levels and 'avg_score' in diff_levels['intertopic_negatives']:
                overall['intertopic_score'] = diff_levels['intertopic_negatives']['avg_score']
            if 'hard_negatives' in diff_levels and 'avg_score' in diff_levels['hard_negatives']:
                overall['hard_examples_score'] = diff_levels['hard_negatives']['avg_score']
        
        # Calculate overall score (weighted average)
        # Give more weight to intratopic (easier) and less to hard examples
        scores = [
            overall['similarity_score'] * 0.15,
            overall.get('negative_detection_score', 0.0) * 0.15,  # Balanced with positive detection
            overall['topic_relevance'] * 0.2,
            overall['diversity_score'] * 0.1,
            overall['intratopic_score'] * 0.2,  # Same topic similarity (easier)
            overall['intertopic_score'] * 0.1,  # Cross-topic similarity (harder)
            overall['hard_examples_score'] * 0.1  # Hard examples (hardest)
        ]
        overall['overall_score'] = np.mean([s for s in scores if s > 0])  # Only count non-zero scores
        
        return overall
    
    def evaluate_baselines(self, queries: List[Dict]) -> Dict:
        """Evaluate baseline methods for comparison."""
        print("Evaluating Baseline Methods...")
        
        baseline_results = {
            'bm25_baseline': self._evaluate_bm25_baseline(queries),
            'random_baseline': self._evaluate_random_baseline(queries),
            'tfidf_baseline': self._evaluate_tfidf_baseline(queries)
        }
        
        return baseline_results
    
    def _evaluate_bm25_baseline(self, queries: List[Dict]) -> Dict:
        """Evaluate BM25 baseline with per-URL document ranking."""
        try:
            from rank_bm25 import BM25Okapi
            import re
            
            # Create per-URL documents (each URL gets its own document)
            url_documents = {}
            url_to_text = {}
            
            for idx, row in self.spiced_data.iterrows():
                # Create separate documents for each URL
                url_1_text = row['text_1']
                url_2_text = row['text_2']
                
                url_to_text[row['URL_1']] = url_1_text
                url_to_text[row['URL_2']] = url_2_text
                
                # Tokenize for BM25
                url_documents[row['URL_1']] = re.findall(r'\w+', url_1_text.lower())
                url_documents[row['URL_2']] = re.findall(r'\w+', url_2_text.lower())
            
            # Create corpus and URL mapping
            corpus = list(url_documents.values())
            url_to_index = {url: idx for idx, url in enumerate(url_documents.keys())}
            
            # Initialize BM25
            bm25 = BM25Okapi(corpus)
            
            ranks = []
            successful_queries = 0
            
            for query in queries[:5]:  # Heavily reduced
                if query.get('evaluation_type') == 'similarity_detection':
                    query_text = query['query_text']
                    query_tokens = re.findall(r'\w+', query_text.lower())
                    
                    # Get BM25 scores for all documents
                    scores = bm25.get_scores(query_tokens)
                    
                    # Find expected URL in corpus
                    expected_url = query.get('expected_url')
                    if expected_url in url_to_index:
                        expected_idx = url_to_index[expected_url]
                        
                        # Rank the expected document among all documents
                        expected_score = scores[expected_idx]
                        rank = sum(1 for score in scores if score > expected_score)
                        ranks.append(rank)
                        successful_queries += 1
            
            # Calculate MRR and Hit@K
            mrr = self._mrr_from_ranks(ranks)
            hit_at_3 = self._hits_at_k(ranks, 3)
            hit_at_10 = self._hits_at_k(ranks, 10)
            
            return {
                'method': 'BM25 (per-URL ranking)',
                'MRR': mrr,
                'Hit@3': hit_at_3,
                'Hit@10': hit_at_10,
                'total_queries': successful_queries,
                'avg_rank': np.mean(ranks) if ranks else 0,
                'total_documents': len(corpus)
            }
            
        except ImportError:
            return {'method': 'BM25', 'MRR': 0.0, 'error': 'rank_bm25 not installed'}
        except Exception as e:
            return {'method': 'BM25', 'MRR': 0.0, 'error': str(e)}
    
    def _evaluate_random_baseline(self, queries: List[Dict]) -> Dict:
        """Evaluate random baseline with proper ranking."""
        import random
        random.seed(42)
        
        ranks = []
        successful_queries = 0
        
        for query in queries[:10]:  # Heavily reduced
            if query.get('evaluation_type') == 'similarity_detection':
                # Random rank between 0 and corpus size
                corpus_size = len(self.spiced_data) if self.spiced_data is not None else 1000
                random_rank = random.randint(0, corpus_size - 1)
                ranks.append(random_rank)
                successful_queries += 1
        
        # Calculate MRR and Hit@K
        mrr = self._mrr_from_ranks(ranks)
        hit_at_3 = self._hits_at_k(ranks, 3)
        hit_at_10 = self._hits_at_k(ranks, 10)
        
        return {
            'method': 'Random',
            'MRR': mrr,
            'Hit@3': hit_at_3,
            'Hit@10': hit_at_10,
            'total_queries': successful_queries,
            'avg_rank': np.mean(ranks) if ranks else 0
        }
    
    def _evaluate_tfidf_baseline(self, queries: List[Dict]) -> Dict:
        """Evaluate TF-IDF baseline with pairwise similarity."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Prepare corpus
            corpus = []
            for idx, row in self.spiced_data.iterrows():
                text = row['text_1'] + " " + row['text_2']
                corpus.append(text)
            
            # Initialize TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            similarities = []
            successful_queries = 0
            
            for query in queries[:5]:  # Heavily reduced
                if query.get('evaluation_type') == 'similarity_detection':
                    query_text = query['query_text']
                    expected_text = query.get('expected_text', '')
                    
                    # Transform query and expected text
                    query_vector = vectorizer.transform([query_text])
                    expected_vector = vectorizer.transform([expected_text])
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_vector, expected_vector)[0][0]
                    similarities.append(similarity)
                    successful_queries += 1
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            return {
                'method': 'TF-IDF (pairwise similarity)',
                'avg_similarity': avg_similarity,
                'total_queries': successful_queries,
                'min_similarity': np.min(similarities) if similarities else 0.0,
                'max_similarity': np.max(similarities) if similarities else 0.0,
                'std_similarity': np.std(similarities) if similarities else 0.0
            }
            
        except ImportError:
            return {'method': 'TF-IDF', 'avg_similarity': 0.0, 'error': 'scikit-learn not installed'}
        except Exception as e:
            return {'method': 'TF-IDF', 'avg_similarity': 0.0, 'error': str(e)}
    
    
    def save_results(self, results: Dict, filename: str = 'evaluation_results.json'):
        """Save evaluation results."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\nEvaluation Summary")
        print("=" * 40)
        
        metadata = results.get('metadata', {})
        print(f"Evaluation Date: {metadata.get('evaluation_date', 'N/A')}")
        print(f"Total Queries: {metadata.get('total_queries', 0)}")
        print(f"SPICED Pairs: {metadata.get('spiced_pairs', 0)}")
        print(f"Topics: {', '.join(metadata.get('topics', []))}")
        
        # Overall performance
        overall = results.get('overall_performance', {})
        print(f"\nOverall Performance:")
        print(f"  Similarity Score: {overall.get('similarity_score', 0.0):.3f}")
        print(f"  Topic Relevance: {overall.get('topic_relevance', 0.0):.3f}")
        print(f"  Diversity Score: {overall.get('diversity_score', 0.0):.3f}")
        print(f"  Overall Score: {overall.get('overall_score', 0.0):.3f}")
        
        # Topic-specific performance
        if 'topic_retrieval' in results and 'topic_metrics' in results['topic_retrieval']:
            print(f"\nTopic Performance:")
            for topic, metrics in results['topic_retrieval']['topic_metrics'].items():
                print(f"  {topic}: {metrics['mean_relevance']:.3f} ({metrics['query_count']} queries)")


def main():
    """Main function to run evaluation pipeline."""
    evaluator = NewsRecommendationEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        # Save results
        evaluator.save_results(results)
        
        # Print summary
        evaluator.print_summary(results)
        
        print("\nEvaluation pipeline completed successfully!")
    else:
        print("Evaluation pipeline failed!")


if __name__ == "__main__":
    main()
