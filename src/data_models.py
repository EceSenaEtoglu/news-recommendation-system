from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional
from enum import Enum

class ContentType(Enum):
    BREAKING_NEWS = "breaking"
    ANALYSIS = "analysis"
    OPINION = "opinion"
    FACTUAL = "factual"
    FEATURE = "feature"


class SourceCategory(Enum):
    WORLD = "world"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    POLITICS = "politics"
    GENERAL = "general"


@dataclass
class Source:
    """News source metadata"""
    id: str
    name: str
    url: str
    category: SourceCategory = SourceCategory.GENERAL
    country: str = "us"
    language: str = "en"
    # Used in reranking
    credibility_score: float = 0.5  # 0-1 scale

@dataclass
class Article:
    """Core article data structure"""
    id: str
    title: str
    content: str
    url: str
    source: Source
    published_at: datetime
    author: Optional[str] = None
    description: Optional[str] = None
    url_to_image: Optional[str] = None
    summary: Optional[str] = None
    
    # Article level labels
    # used in retrieval system
    content_type: ContentType = ContentType.FACTUAL
    urgency_score: float = 0.5  # 0=evergreen, 1=breaking news
    
    
    # Computed fields
    embedding: Optional[List[float]] = None
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    # called automatically after the object is created
    def __post_init__(self):
        """Clean and validate article data"""
        # Remove null bytes that can break SQLite
        if self.content:
            self.content = self.content.replace('\x00', '')
        if self.title:
            self.title = self.title.replace('\x00', '')
        if self.description:
            self.description = self.description.replace('\x00', '')
            
        # bias detection
        self._detect_content_bias()

            
    # TODO, should be swiched with NER or classification
    # sets the content type and urgency score based on simple rules
    def _detect_content_bias(self):
        """Smart bias detection based on article content and metadata"""
        text = f"{self.title} {self.description}".lower()
        
        # Detect content type from keywords
        breaking_keywords = ["breaking", "urgent", "developing", "just in", "alert"]
        analysis_keywords = ["analysis", "explained", "why", "how", "deep dive", "investigation"]
        opinion_keywords = ["opinion", "editorial", "comment", "perspective", "view"]
        
        if any(word in text for word in breaking_keywords):
            self.content_type = ContentType.BREAKING_NEWS
            self.urgency_score = 0.9
        elif any(word in text for word in opinion_keywords):
            self.content_type = ContentType.OPINION
        elif any(word in text for word in analysis_keywords):
            self.content_type = ContentType.ANALYSIS
        
        # Detect urgency from publish time (articles published within 2 hours are "breaking")
        URGENCY_HOUR_INTERVAL = 2
        # Normalize to timezone-aware computation (UTC)
        pub = self.published_at
        if isinstance(pub, datetime):
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            else:
                pub = pub.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        hours_old = (now_utc - pub).total_seconds() / 3600
        if hours_old <URGENCY_HOUR_INTERVAL:
            self.urgency_score = min(self.urgency_score + 0.3, 1.0)

@dataclass
class UserProfile:
    """Simple user profile for personalization"""
    user_id: str
    preferred_topics: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    blocked_sources: List[str] = field(default_factory=list)
    preferred_content_types: List[ContentType] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


# RAG components
#-------------------------
@dataclass
# TODO how to init the query type
# Query type refers to different angles or aspects of coverage for the same underlying news event
class SearchQuery:
    """Search query with metadata"""
    text: str
    user_id: Optional[str] = None
    query_type: str = "general"  # general, breaking, background
    limit: int = 10

    # TODO to avoid filter bubble
    include_opposing_views: bool = True 
    
@dataclass
class SearchResult:
    """Search result with relevance scoring"""
    article: Article
    relevance_score: float
    freshness_score: float = 0.0
    personalization_score: float = 0.0
    final_score: float = 0.0
    explanation: Optional[str] = None  # Explain Why this search was recommended

# Entity for graphrag in the future, right now pure sqllite database tables are used
#@dataclass
#class Entity:
#    """Named entity for graph construction"""
#    name: str
#    entity_type: str  # PERSON, ORG, GPE, etc.
#    mentions: int = 1
#    first_seen: datetime = field(default_factory=datetime.now)
#    articles: List[str] = field(default_factory=list)  # Article IDs