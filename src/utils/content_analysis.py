"""
Content quality analysis utilities for news recommendations.
"""

from typing import List
from data_models import Article
from config import RAGConfig


class ContentQualityAnalyzer:
    """Analyzes content quality for news articles"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def calculate_content_quality(self, article: Article) -> float:
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
