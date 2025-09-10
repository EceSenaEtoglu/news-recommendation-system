import argparse
import asyncio

from src.embeddings import EmbeddingSystem
from src.providers.fixture import FixtureProvider


async def build_index_from_fixtures(out_index: str, out_meta: str, folder: str, limit: int):
    provider = FixtureProvider(folder=folder, shuffle=False)
    featured, candidates = await provider.fetch_featured_and_candidates(
        featured_limit=2, candidate_limit=limit
    )
    articles = featured + candidates

    emb = EmbeddingSystem(index_path=out_index, metadata_path=out_meta)
    # reset index
    added = emb.add_articles(articles)
    emb.save_index()
    stats = emb.get_stats()
    print(f"Added {added} articles to FAISS. Stats: {stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="src/providers/news_fixtures")
    ap.add_argument("--index", default="db/faiss.index")
    ap.add_argument("--meta", default="db/faiss_metadata.pkl")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    asyncio.run(build_index_from_fixtures(args.index, args.meta, args.folder, args.limit))


if __name__ == "__main__":
    main()


