from .base import ArticleProvider
from .newsapi import NewsAPIProvider

# the factory pattern
# TODO add other providers and call them
def create_provider(api_key: str) -> NewsAPIProvider:
    """Create a NewsAPI provider - simple factory"""
    return NewsAPIProvider(api_key)



# control what gets imported
__all__ = ["ArticleProvider", "NewsAPIProvider", "create_provider"]