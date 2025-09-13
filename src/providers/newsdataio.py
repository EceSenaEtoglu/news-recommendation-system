# src/providers/newsdataio.py
from __future__ import annotations
import os
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from .base import ArticleProvider
from ..data_models import Article, Source, ContentType

try:
    # robust parsing for various pubDate formats (ISO/RFC-2822)
    from dateutil import parser as dateparser
except Exception:
    dateparser = None


class NewsDataIOProvider(ArticleProvider):
    """
    Minimal NewsData.io wrapper (MVP-friendly).

    - Endpoint: /api/1/latest (last 48h)
    - Full text: full_content=1
    - Pagination: response['nextPage'] -> request param page=<token>
    - Category is mapped to Article.topics (free and reliable as a coarse topic)
    """

    BASE_URL = "https://newsdata.io/api/1/latest"

    def __init__(self, api_key: str, *, language: str = "en", timeout: int = 15):
        self.api_key = api_key
        self.language = language
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "portfolio-news-mvp/1.0"})

    # ---------------- ArticleProvider API ---------------- #

    async def fetch_articles(
        self,
        limit: int = 100,
        *,
        q: Optional[str] = None,
        categories: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        full_content: bool = True,
        page: Optional[str] = None,
    ) -> List[Article]:
        """
        Fetch up to `limit` recent articles. Iterates pages until limit or no nextPage.
        Filters are optional and map to NewsData.io params.
        """
        params: Dict[str, Any] = {
            "apikey": self.api_key,
            "language": self.language,
            "full_content": 1 if full_content else 0,
        }
        if q:
            params["q"] = q
        if categories:
            params["category"] = ",".join(categories)
        if countries:
            params["country"] = ",".join(countries)
        if domains:
            params["domain"] = ",".join(domains)
        if page:
            params["page"] = page  # resume from a pagination token

        out: List[Article] = []
        next_page: Optional[str] = None

        while len(out) < limit:
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                print(f"[NewsDataIOProvider] request error: {e}")
                break

            results = payload.get("results") or payload.get("data") or []
            if not results:
                break

            for item in results:
                art = self._to_article(item)
                if art:
                    out.append(art)
                    if len(out) >= limit:
                        break

            # pagination: nextPage -> page
            next_page = payload.get("nextPage")
            if not next_page or len(out) >= limit:
                break
            params["page"] = next_page  # continue

        return out

    def get_sources(self) -> List[Source]:
        """
        Optional: you can implement a /sources call later.
        For MVP we infer Source per article.
        """
        return []

    # ---------------- internals ---------------- #

    def _to_article(self, item: Dict[str, Any]) -> Optional[Article]:
        """
        Map a NewsData.io item to your Article model.
        Fields commonly present:
          title, description, content/full_content, link/url, image_url,
          pubDate, source_id, source, source_url, language, country,
          category (str or list), creator (str or list)
        """
        title = item.get("title")
        if not title:
            return None

        # Prefer full body when available
        content = item.get("content") or item.get("full_content") or ""
        description = item.get("description") or ""
        url = item.get("link") or item.get("url") or ""
        image_url = item.get("image_url") or item.get("image")

        # Source
        source_id = str(item.get("source_id") or item.get("source") or "unknown")
        source_name = str(item.get("source") or item.get("source_id") or "Unknown")
        source_url = str(item.get("source_url") or "")
        lang = (item.get("language") or self.language or "en")[:5]
        country = self._first(item.get("country"))
        category = self._first(item.get("category")) or "general"  # ← will feed Article.topics

        src = Source(
            id=source_id,
            name=source_name,
            url=source_url,
            category=category,
            country=str(country or "us"),
            language=lang,
            credibility_score=0.6,  # TODO: calibrate per source if you maintain a list
        )

        published_at = self._parse_dt(item.get("pubDate")) or datetime.now(timezone.utc)

        # Authors
        creator = item.get("creator")
        if isinstance(creator, list):
            author = ", ".join([c for c in creator if c])
        else:
            author = creator if isinstance(creator, str) else None

        # NOTE: content_type/target_audience will be upgraded by your classifiers later.
        return Article(
            id=str(hash(url or title)),
            title=title,
            content=content or description,
            description=description,
            url=url,
            url_to_image=image_url,
            source=src,
            published_at=published_at,        # tz-aware UTC
            author=author,
            content_type=ContentType.FACTUAL, # default; classifier can overwrite
            topics=[category],                # ← map NewsData category to Article.topics
        )

    @staticmethod
    def _first(v):
        if isinstance(v, list) and v:
            return v[0]
        return v

    @staticmethod
    def _parse_dt(val: Optional[str]) -> Optional[datetime]:
        if not val:
            return None
        if dateparser:
            try:
                dt = dateparser.parse(val)
                if not dt:
                    return None
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                return None
        try:
            # simple ISO fallback
            if val.endswith("Z"):
                val = val.replace("Z", "+00:00")
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
