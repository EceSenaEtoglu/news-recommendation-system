import os
import json
import time
import argparse
from urllib.parse import urlparse
from datetime import datetime, timezone

import feedparser
from newspaper import Article as NPArticle


DEFAULT_FEEDS = [
    "https://www.reuters.com/rssFeed/worldNews",
    "https://www.reuters.com/finance/markets/rss",
    "https://www.theverge.com/rss/index.xml",
    "https://www.techcrunch.com/feed/",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]


def extract_fulltext(url: str, timeout: int = 15) -> tuple[str, str]:
    art = NPArticle(url)
    art.download()
    art.parse()
    title = art.title or ""
    text = art.text or ""
    return title, text


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="src/providers/news_fixtures", help="Output fixtures folder")
    ap.add_argument("--featured_count", type=int, default=2)
    ap.add_argument("--pool_count", type=int, default=50)
    ap.add_argument("--feeds", nargs="*", default=DEFAULT_FEEDS)
    ap.add_argument("--throttle", type=float, default=1.0, help="Seconds between downloads")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    items = []
    for feed_url in args.feeds:
        try:
            d = feedparser.parse(feed_url)
        except Exception as e:
            print("Failed to parse feed:", feed_url, e)
            continue
        for e in d.entries:
            link = e.get("link") or e.get("id")
            if not link:
                continue
            title = e.get("title") or ""
            desc = e.get("summary") or ""
            pub = e.get("published") or e.get("updated")
            image = None
            try:
                t, content = extract_fulltext(link)
                if t and not title:
                    title = t
                if not content or len(content) < 200:
                    # Skip very short pages
                    continue
            except Exception as ex:
                print("Extract failed:", link, ex)
                continue

            items.append({
                "title": title,
                "description": desc,
                "content": content,
                "url": link,
                "image_url": image,
                "source": urlparse(link).netloc,
                "source_id": urlparse(link).netloc.split(":")[0],
                "source_url": link,
                "language": "en",
                "country": "us",
                "category": "general",
                "pubDate": pub or iso_now(),
                "creator": None
            })
            print("OK:", title[:60])
            time.sleep(args.throttle)

    # Dedup by URL
    seen = set()
    deduped = []
    for it in items:
        key = it.get("url")
        if key and key not in seen:
            seen.add(key)
            deduped.append(it)

    # Write featured and pool
    featured = deduped[: args.featured_count]
    pool = deduped[args.featured_count : args.featured_count + args.pool_count]

    with open(os.path.join(args.out_dir, "featured.json"), "w", encoding="utf-8") as f:
        json.dump(featured, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "pool.json"), "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(featured)} featured and {len(pool)} pool items to {args.out_dir}")


if __name__ == "__main__":
    main()


