# src/summarization.py
import re
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class NewsSummarizer:
    """News article summarization using transformer models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer with a pre-trained model.
        
        Args:
            model_name: Hugging Face model name for summarization
        """
        self.model_name = model_name
        self.summarizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model"""
        try:
            # Use BART-large-CNN with optimized settings for longer summaries
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1,
                max_length=300,        # Longer summaries for news
                min_length=150,        # Higher minimum to force longer summaries
                do_sample=False,
                length_penalty=3.0,    # Much stronger encouragement for longer summaries
                repetition_penalty=1.5, # Reduce repetition
                num_beams=4           # Better quality generation
            )
            print(f" Loaded summarization model: {self.model_name}")
        except Exception as e:
            print(f" Failed to load summarization model: {e}")
            print(" Falling back to extractive summarization")
            self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """
        Summarize the given text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summarized text
        """
        if not text or len(text.strip()) < 50:
            return text[:200] + "..." if len(text) > 200 else text
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        if self.summarizer is None:
            return self._extractive_summarize(cleaned_text)
        
        try:
            # Truncate text if too long (most models have input limits)
            if len(cleaned_text) > 1024:
                cleaned_text = cleaned_text[:1024]
            
            # Generate summary
            result = self.summarizer(
                cleaned_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary = result[0]['summary_text']
            return self._post_process_summary(summary)
            
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
            return self._extractive_summarize(cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for summarization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
        
        return text.strip()
    
    def _extractive_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Fallback extractive summarization"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Take first few sentences
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences)
        
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def _post_process_summary(self, summary: str) -> str:
        """Post-process the generated summary"""
        # Ensure proper capitalization
        summary = summary.strip()
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure it ends with proper punctuation
        if summary and not summary[-1] in '.!?':
            summary += '.'
        
        return summary

# Global summarizer instance
_summarizer_instance: Optional[NewsSummarizer] = None

def get_summarizer() -> NewsSummarizer:
    """Get or create the global summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = NewsSummarizer()
    return _summarizer_instance

def summarize_article(content: str, title: str = "", max_length: int = 300) -> str:
    """
    Convenience function to summarize an article.
    
    Args:
        content: Article content
        title: Article title (optional)
        max_length: Maximum summary length
        
    Returns:
        Summarized text
    """
    summarizer = get_summarizer()
    
    # Combine title and content for better context
    if title:
        full_text = f"{title}. {content}"
    else:
        full_text = content
    
    return summarizer.summarize(full_text, max_length=max_length)
