# src/providers/fixtures.py
from __future__ import annotations
import json, glob, hashlib, random
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from .base import ArticleProvider
from ..data_models import Article, Source, ContentType, SourceCategory

try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None

def _parse_dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    if dateparser:
        try:
            dt = dateparser.parse(val)
            if not dt: return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    try:
        if val.endswith("Z"):
            val = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _first(v):
    if isinstance(v, list) and v:
        return v[0]
    return v

class FixtureProvider(ArticleProvider):
    """
    Reads local JSON files that mimic API responses.
    Useful for portfolio/MVP because real APIs don't return full text for free.
    """

    def __init__(
        self,
        folder: str = "src/providers/news_fixtures",
        *,
        language: str = "en",
        shuffle: bool = True,
        seed: Optional[int] = 1337,
        max_per_file: Optional[int] = None,
    ):
        self.folder = folder
        self.language = language
        self.shuffle = shuffle
        self.rng = random.Random(seed) if seed is not None else random

    async def fetch_articles(
        self,
        limit: int = 100,
        *,
        q: Optional[str] = None,
        categories: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        full_content: bool = True,  # ignored; fixtures are expected to have content
    ) -> List[Article]:
        paths = sorted(glob.glob(f"{self.folder}/*.json"))
        items: List[Dict[str, Any]] = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
                    # Handle new structure with metadata
                    if isinstance(chunk, dict) and "articles" in chunk:
                        articles = chunk["articles"]
                        if isinstance(articles, list):
                            items.extend(articles)
                    elif isinstance(chunk, list):
                        # Handle old structure (direct list)
                        items.extend(chunk)
            except Exception as e:
                print(f"[FixtureProvider] failed to read {p}: {e}")

        # Optional shuffle so it feels like a live feed
        if self.shuffle:
            self.rng.shuffle(items)

        # Lightweight filters (q/category/country/domain)
        def _passes(item: Dict[str, Any]) -> bool:
            if q:
                hay = " ".join([
                    str(item.get("title","")),
                    str(item.get("description","")),
                    str(item.get("content",""))
                ]).lower()
                if q.lower() not in hay:
                    return False
            if categories:
                cats = item.get("category") or []
                if isinstance(cats, str): cats = [cats]
                if cats and not set(c.lower() for c in cats) & set(x.lower() for x in categories):
                    return False
            if countries:
                c = _first(item.get("country"))
                if c and str(c).lower() not in [x.lower() for x in countries]:
                    return False
            if domains:
                # accept if source_url domain or link host matches
                src = item.get("source_url","") or ""
                link = item.get("link","") or item.get("url","") or ""
                if not any(d.lower() in src.lower() or d.lower() in link.lower() for d in domains):
                    return False
            return True

        out: List[Article] = []
        for raw in items:
            if not _passes(raw):
                continue
            art = self._to_article(raw)
            if art:
                out.append(art)
                if len(out) >= limit:
                    break
        return out

    async def fetch_featured_and_candidates(
        self,
        *,
        featured_limit: int = 2,
        candidate_limit: int = 50,
        q: Optional[str] = None,
    ) -> Tuple[List[Article], List[Article]]:
        """Return two lists: (featured, candidates).
        - Featured are loaded from featured.json if present; otherwise first N fixtures.
        - Candidates are other articles (excluding featured) for recommendation.
        """
        # Load all fixtures
        paths = sorted(glob.glob(f"{self.folder}/*.json"))
        items: List[Dict[str, Any]] = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
                    # Handle new structure with metadata
                    if isinstance(chunk, dict) and "articles" in chunk:
                        items.extend(chunk["articles"])
                    elif isinstance(chunk, list):
                        items.extend(chunk)
            except Exception as e:
                print(f"[FixtureProvider] failed to read {p}: {e}")

        # Optional shuffle for non-featured pool
        pool_items = list(items)

        # Try to load explicit featured file
        featured_items: List[Dict[str, Any]] = []
        fpath = f"{self.folder}/featured.json"
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                fj = json.load(f)
                # Handle new structure with metadata
                if isinstance(fj, dict) and "articles" in fj:
                    featured_items = fj["articles"][:featured_limit]
                elif isinstance(fj, list):
                    featured_items = fj[:featured_limit]
        except Exception:
            # Fall back to first N items deterministically
            featured_items = pool_items[:featured_limit]

        # Convert and filter based on the limit
        def _to_list(dicts: List[Dict[str, Any]], lim: int) -> List[Article]:
            out: List[Article] = []
            for d in dicts:
                a = self._to_article(d)
                if a:
                    out.append(a)
                    if len(out) >= lim:
                        break
            return out

        featured = _to_list(featured_items, featured_limit)

        # Build candidate pool excluding featured ids
        featured_ids = {a.get("link") or a.get("url") or a.get("title") for a in featured_items}
        pool_filtered = [d for d in pool_items if (d.get("link") or d.get("url") or d.get("title")) not in featured_ids]

        if self.shuffle:
            self.rng.shuffle(pool_filtered)

        candidates_all = _to_list(pool_filtered, candidate_limit)
        return featured, candidates_all

    def get_sources(self) -> List[Source]:
        # Optional: derive a unique set of sources by scanning fixtures.
        return []

    def _to_article(self, item: Dict[str, Any]) -> Optional[Article]:
        title = item.get("title")
        if not title:
            return None

        content = (item.get("content") or item.get("full_content") or "").strip()
        description = (item.get("description") or "").strip()
        url = item.get("link") or item.get("url") or ""
        image_url = item.get("image_url") or item.get("image")

        source_id = str(item.get("source_id") or item.get("source") or "unknown")
        source_name = str(item.get("source") or item.get("source_id") or "Unknown")
        source_url = str(item.get("source_url") or "")
        lang = (item.get("language") or self.language or "en")[:5]
        country = _first(item.get("country"))
        category_str = _first(item.get("category")) or "general"
        # Map string to enum
        try:
            category = SourceCategory(category_str.lower())
        except ValueError:
            category = SourceCategory.GENERAL

        src = Source(
            id=source_id,
            name=source_name,
            url=source_url,
            category=category,
            country=str(country or "us"),
            language=lang,
            credibility_score=0.6,
        )

        published_at = _parse_dt(item.get("pubDate")) or datetime.now(timezone.utc)

        creator = item.get("creator")
        if isinstance(creator, list):
            author = ", ".join([c for c in creator if c])
        else:
            author = creator if isinstance(creator, str) else None

        # Deterministic ID from URL or title
        base = url if url else title
        art_id = hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]

        return Article(
            id=art_id,
            title=title,
            content=content or description,  # never empty
            description=description,
            url=url,
            url_to_image=image_url,
            source=src,
            published_at=published_at,
            author=author,
            content_type=ContentType.FACTUAL,
            topics=[category.value],  # map category → topics for free "topic" signal
            entities=item.get("entities", []),  # Extract entities with types (name, type, count)
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from fixture files including last update timestamp"""
        metadata = {
            "last_updated": None,
            "featured_count": 0,
            "pool_count": 0,
            "total_articles": 0
        }
        
        # Try to read metadata from featured.json
        featured_path = f"{self.folder}/featured.json"
        try:
            with open(featured_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "metadata" in data:
                    metadata.update(data["metadata"])
        except Exception:
            pass
            
        return metadata
