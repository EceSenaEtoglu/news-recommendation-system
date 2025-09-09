# src/providers/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from ..data_models import Article, Source

class ArticleProvider(ABC):
    """Base class for article providers"""
    
    @abstractmethod
    async def fetch_articles(self, limit: int = 100) -> List[Article]:
        """Fetch articles from the provider"""
        pass
    
    @abstractmethod
    def get_sources(self) -> List[Source]:
        """Get available sources"""
        pass