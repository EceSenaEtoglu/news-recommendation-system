import os
import json
import time
import argparse
from urllib.parse import urlparse
import random
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
from newspaper import Article as NPArticle
from newspaper import Config as NPConfig


# THIS SCRIPT CREATES featured.json and pool.json

# Import entity extraction function
from src.ingestion import extract_entities
from src.config import MAX_ENTITIES_PER_ARTICLE


DEFAULT_FEEDS = [
    # World / General
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.reuters.com/rssFeed/worldNews",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.npr.org/1004/rss.xml",        # NPR World
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",  # World news
    "https://www.ft.com/world?format=rss",

    # US / Politics / Business
    "https://www.reuters.com/finance/markets/rss",
    "https://feeds.npr.org/1001/rss.xml",        # NPR Top Stories
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top news
    "https://www.politico.com/rss/politics-news.xml",

    # Tech
    "https://www.theverge.com/rss/index.xml",
    "https://techcrunch.com/feed/",
    "https://arstechnica.com/feed/",
    "https://www.engadget.com/rss.xml",
    "https://www.wired.com/feed/rss",
    "https://hnrss.org/frontpage",

    # Science / Misc (optional but useful for volume)
    "https://rss.slashdot.org/Slashdot/slashdot",
]

# Map some feeds to coarse categories for better diversity
FEED_CATEGORY = {
    "https://feeds.bbci.co.uk/news/world/rss.xml": "world",
    "https://www.reuters.com/rssFeed/worldNews": "world",
    "https://www.theguardian.com/world/rss": "world",
    "https://www.aljazeera.com/xml/rss/all.xml": "world",
    "https://feeds.npr.org/1004/rss.xml": "world",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html": "world",
    "https://www.ft.com/world?format=rss": "world",

    "https://www.reuters.com/finance/markets/rss": "business",
    "https://feeds.npr.org/1001/rss.xml": "business",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html": "business",
    "https://www.politico.com/rss/politics-news.xml": "politics",

    "https://www.theverge.com/rss/index.xml": "technology",
    "https://techcrunch.com/feed/": "technology",
    "https://arstechnica.com/feed/": "technology",
    "https://www.engadget.com/rss.xml": "technology",
    "https://www.wired.com/feed/rss": "technology",
    "https://hnrss.org/frontpage": "technology",

    "https://rss.slashdot.org/Slashdot/slashdot": "science",
}

# Domain-based category mapping for better coverage
DOMAIN_CATEGORY = {
    # Technology
    "techcrunch.com": "technology",
    "www.techcrunch.com": "technology", 
    "theverge.com": "technology",
    "www.theverge.com": "technology",
    "arstechnica.com": "technology", 
    "www.arstechnica.com": "technology",
    "engadget.com": "technology",
    "www.engadget.com": "technology",
    "wired.com": "technology",
    "www.wired.com": "technology",
    "hnrss.org": "technology",
    
    # World/News
    "bbc.com": "world",
    "www.bbc.com": "world",
    "reuters.com": "world", 
    "www.reuters.com": "world",
    "theguardian.com": "world",
    "www.theguardian.com": "world",
    "aljazeera.com": "world",
    "www.aljazeera.com": "world",
    "npr.org": "world",
    "www.npr.org": "world",
    
    # Business
    "cnbc.com": "business",
    "www.cnbc.com": "business",
    "ft.com": "business", 
    "www.ft.com": "business",
    
    # Politics
    "politico.com": "politics",
    "www.politico.com": "politics",
    
    # Science
    "slashdot.org": "science",
    "www.slashdot.org": "science",
    "yro.slashdot.org": "science",
    
    # General/Other
    "bmj.com": "science",
    "www.bmj.com": "science",
    "techxplore.com": "technology",
    "www.techxplore.com": "technology",
    
    # Additional mappings for sources in database
    "sibellavia.lol": "general",
    "worksinprogress.co": "general",
    "oldvcr.blogspot.com": "technology",
}

# use the newspaper3k to extract the full text of the article
def extract_fulltext(url: str, np_config: NPConfig) -> tuple[str, str]:
    art = NPArticle(url, config=np_config)
    art.download()
    art.parse()
    return (art.title or ""), (art.text or "")

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def calculate_optimal_max_per_feed(featured_count, pool_count, num_feeds, safety_factor=2.5, min_per_feed=5, max_per_feed=25):
    """
    Calculate optimal max_per_feed based on requirements with safety factor.
    
    Args:
        featured_count: Number of featured articles needed
        pool_count: Number of pool articles needed  
        num_feeds: Number of RSS feeds available
        safety_factor: Multiplier for buffer (2.5 = 150% buffer)
        min_per_feed: Minimum articles per feed (for diversity)
        max_per_feed: Maximum articles per feed (for performance)
    
    Returns:
        Optimal max_per_feed value
    """
    needed_articles = featured_count + pool_count
    
    # Calculate base requirement
    base_per_feed = needed_articles / num_feeds
    
    # Apply safety factor for buffer (handles feed failures, quality filtering, etc.)
    optimal_per_feed = int(base_per_feed * safety_factor)
    
    # Apply constraints
    optimal_per_feed = max(min_per_feed, min(optimal_per_feed, max_per_feed))
    
    return optimal_per_feed


def validate_configuration(featured_count, pool_count, max_per_feed, num_feeds):
    """Validate that configuration makes sense and provide feedback."""
    total_possible = num_feeds * max_per_feed
    needed = featured_count + pool_count
    
    print(f"Configuration analysis:")
    print(f"  Featured articles: {featured_count}")
    print(f"  Pool articles: {pool_count}")
    print(f"  Total needed: {needed}")
    print(f"  Available feeds: {num_feeds}")
    print(f"  Max per feed: {max_per_feed}")
    print(f"  Total possible: {total_possible}")
    
    if total_possible < needed:
        print(f"ERROR: Insufficient articles ({total_possible} possible, {needed} needed)")
        print(f"  Consider increasing max_per_feed or reducing requirements")
        return False
    
    buffer_articles = total_possible - needed
    buffer_percentage = (buffer_articles / needed) * 100
    
    if buffer_percentage < 50:
        print(f"WARNING: Low buffer ({buffer_percentage:.1f}%, {buffer_articles} extra articles)")
        print(f"  Consider increasing max_per_feed for better reliability")
    else:
        print(f"Configuration OK: {buffer_percentage:.1f}% buffer ({buffer_articles} extra articles)")
    
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="src/providers/news_fixtures", help="Output fixtures folder")
    ap.add_argument("--featured_count", type=int, default=2)
    ap.add_argument("--pool_count", type=int, default=50)
    ap.add_argument("--feeds", nargs="*", default=DEFAULT_FEEDS)
    ap.add_argument("--throttle", type=float, default=0.0, help="Seconds between FEED downloads (not per article)")
    ap.add_argument("--workers", type=int, default=12, help="Parallel workers for article downloads")
    ap.add_argument("--min_chars", type=int, default=160, help="Minimum content length to keep an article")
    ap.add_argument("--max_per_feed", type=int, default=None, help="Max items to take from each feed (auto-calculated if not specified)")
    ap.add_argument("--safety_factor", type=float, default=2.5, help="Safety factor for max_per_feed calculation (2.5 = 150% buffer)")
    ap.add_argument("--user_agent", default="Mozilla/5.0 (compatible; NewsFetcher/1.0; +https://example.com/bot)",
                    help="User agent for article fetching")
    ap.add_argument("--request_timeout", type=int, default=20, help="Timeout for article HTTP requests (seconds)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling featured")
    args = ap.parse_args()
    
    # Auto-calculate max_per_feed if not specified
    if args.max_per_feed is None:
        num_feeds = len(args.feeds)
        args.max_per_feed = calculate_optimal_max_per_feed(
            featured_count=args.featured_count,
            pool_count=args.pool_count,
            num_feeds=num_feeds,
            safety_factor=args.safety_factor
        )
        print(f"Auto-calculated max_per_feed: {args.max_per_feed}")
    
    # Validate configuration
    num_feeds = len(args.feeds)
    if not validate_configuration(args.featured_count, args.pool_count, args.max_per_feed, num_feeds):
        print("Configuration validation failed. Exiting.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Configure newspaper globally
    np_config = NPConfig()
    np_config.browser_user_agent = args.user_agent
    np_config.request_timeout = args.request_timeout
    # for now only text is needed
    np_config.fetch_images = False  
    np_config.memoize_articles = False

    # 1) Gather candidate entries from all feeds first
    candidates = []
    print(f"Fetching from {len(args.feeds)} RSS feeds...")
    for feed_url in args.feeds:
        try:
            print(f"Parsing feed: {feed_url}")
            d = feedparser.parse(feed_url)
            print(f"  Found {len(d.entries)} entries")
        except Exception as e:
            print("Failed to parse feed:", feed_url, e)
            continue

        taken = 0
        for e in d.entries:
            link = e.get("link") or e.get("id")
            if not link:
                continue
            candidates.append({
                "feed": feed_url,
                "link": link,
                "title": e.get("title") or "",
                "desc": e.get("summary") or "",
                "pub": e.get("published") or e.get("updated") or None
            })
            taken += 1
            if args.max_per_feed > 0 and taken >= args.max_per_feed:
                break
        if args.throttle > 0:
            time.sleep(args.throttle)

    # 2) De-dup by URL before fetching full text
    seen = set()
    unique_candidates = []
    for c in candidates:
        u = c["link"]
        if u and u not in seen:
            seen.add(u)
            unique_candidates.append(c)

    print(f"Found {len(unique_candidates)} unique candidate links across {len(args.feeds)} feeds.")
    if len(unique_candidates) == 0:
        print("ERROR: No articles found from any RSS feeds!")
        return

    # 3) Fetch full text in parallel
    items = []
    # returns a structured
    def process(c):
        link = c["link"]
        try:
            t, content = extract_fulltext(link, np_config)
            title = t or c["title"]
            if not content or len(content) < args.min_chars:
                return None
            pub = c["pub"] or iso_now()
            netloc = urlparse(link).netloc
            
            # Extract entities using spaCy NER
            try:
                # Create a temporary article-like object for entity extraction
                temp_article = type('Article', (), {
                    'title': title or "",
                    'description': c["desc"] or "",
                    'content': content
                })()
                entity_tuples = extract_entities(temp_article)
                entities = [name for name, _, _ in entity_tuples[:MAX_ENTITIES_PER_ARTICLE]]
            except Exception as e:
                print(f"Entity extraction failed for {link}: {e}")
                entities = []
            
            return {
                "title": title or "",
                "description": c["desc"] or "",
                "content": content,
                "url": link,
                "image_url": None,
                "source": netloc,
                "source_id": netloc.split(":")[0],
                "source_url": link,
                "language": "en",
                "country": "us",
                "category": "general",
                "pubDate": pub,
                "creator": None,
                "entities": entities  # Add extracted entities
            }
        except Exception as ex:
            print("Extract failed:", link, ex)
            return None

    print(f"Extracting full text from {len(unique_candidates)} articles...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        # Submit all article processing tasks to the thread pool
        # Each task fetches full text content from a URL and returns a structured dictionary
        futures = [ex.submit(process, c) for c in unique_candidates]
        
        # Collect results as they complete
        # This allows processing multiple articles concurrently for better performance
        for fut in as_completed(futures):
            r = fut.result()
            if r:  # Only append successful results (failed fetches return None)
                items.append(r)
    
    print(f"Successfully extracted {len(items)} articles with sufficient content.")

    # 4) Final de-dup by URL
    deduped = []
    seen = set()
    for it in items:
        u = it["url"]
        if u not in seen:
            seen.add(u)
            deduped.append(it)

    # TODO sort newest first 


    # ------Following code block constructs the featured and pool articles------

    # By randomizing, select articles for featured and pool that has different categories
    rng = random.Random(args.seed)
    by_category = {}

    # traverse articles and group by category
    for article in deduped:
        # Determine category
        feed_url = article.get("feed", "")
        source_url = article.get("source_url", "")
        
        # Extract domain from source_url for mapping
        source_domain = source_url.replace("https://", "").replace("http://", "").split("/")[0]
        
        category = (FEED_CATEGORY.get(source_url) or 
                   FEED_CATEGORY.get(feed_url) or 
                   DOMAIN_CATEGORY.get(source_domain) or  # Try domain mapping
                   article.get("category") or 
                   "general")
        article["category"] = category
        by_category.setdefault(category, []).append(article)
    
    # Shuffle articles within each category
    for category_articles in by_category.values():
        rng.shuffle(category_articles)
    
    # Round-robin selection for featured articles
    category_names = sorted(by_category.keys())
    featured = []
    category_indices = {category: 0 for category in category_names}
    
    while len(featured) < args.featured_count and category_names:
        for category in list(category_names):
            category_articles = by_category[category]
            current_index = category_indices[category]
            
            if current_index < len(category_articles):
                featured.append(category_articles[current_index])
                category_indices[category] = current_index + 1
                
                if len(featured) >= args.featured_count:
                    break
            else:
                # Remove category if no more articles
                category_names.remove(category)

    # Pool = remaining items excluding featured URLs, ensuring pool and featured are different
    featured_urls = {article.get("url") for article in featured}
    pool = [article for article in deduped if article.get("url") not in featured_urls][: args.pool_count]

    # Clear and save the extracted articles
    print(f"Clearing old fixture files...")
    
    # Clear existing files first
    featured_path = os.path.join(args.out_dir, "featured.json")
    pool_path = os.path.join(args.out_dir, "pool.json")
    
    # Write empty files first to ensure clean state
    with open(featured_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    
    # Create metadata with timestamp
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Now write the new articles with metadata
    featured_data = {
        "metadata": {
            "last_updated": timestamp,
            "featured_count": len(featured),
            "pool_count": len(pool),
            "total_articles": len(featured) + len(pool)
        },
        "articles": featured
    }
    
    pool_data = {
        "metadata": {
            "last_updated": timestamp,
            "featured_count": len(featured),
            "pool_count": len(pool),
            "total_articles": len(featured) + len(pool)
        },
        "articles": pool
    }
    
    with open(featured_path, "w", encoding="utf-8") as f:
        json.dump(featured_data, f, ensure_ascii=False, indent=2)

    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(featured)} featured and {len(pool)} pool items to {args.out_dir}")

if __name__ == "__main__":
    main()
