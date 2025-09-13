#!/usr/bin/env python3
"""
Reset fixture files to clean state
=================================

This script clears the fixture files to start fresh.
"""

import os
import json


def main():
    """Reset fixture files"""
    fixtures_dir = "src/providers/news_fixtures"
    
    # Create empty fixture files
    empty_featured = []
    empty_pool = []
    
    # Write empty files
    with open(os.path.join(fixtures_dir, "featured.json"), "w", encoding="utf-8") as f:
        json.dump(empty_featured, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(fixtures_dir, "pool.json"), "w", encoding="utf-8") as f:
        json.dump(empty_pool, f, ensure_ascii=False, indent=2)
    
    print("Fixture files reset successfully!")
    print("Now you can fetch fresh articles with:")
    print("  python scripts/ingest_rss.py --out_dir src/providers/news_fixtures --featured_count 20 --pool_count 100")


if __name__ == "__main__":
    main()
