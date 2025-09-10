import argparse
from datetime import datetime

from src.storage import ArticleDB
from src.embeddings import EmbeddingSystem
from src.recommendation_learner import AIRecommender, RecommendationConfig
from src.reranker import TrainableLogisticReranker
from src.providers.fixture import FixtureProvider


def cmd_build_index(db: ArticleDB, emb: EmbeddingSystem):
    emb.rebuild_index_from_db(db)


def cmd_show_featured(provider: FixtureProvider):
    import asyncio
    featured, candidates = asyncio.run(provider.fetch_featured_and_candidates())
    print("Featured:")
    for a in featured:
        print("-", a.title)


def cmd_recommend(db: ArticleDB, emb: EmbeddingSystem, article_id: str, use_mmr: bool):
    rec = AIRecommender(db, emb, RecommendationConfig(use_mmr=use_mmr))
    art = db.get_article_by_id(article_id)
    if not art:
        print("Article not found:", article_id)
        return
    out = rec.recommend_for_article(art, k=5)
    for cand, score in out:
        expl = rec.explain_recommendation(art, cand)
        print(f"{score:.3f} | {cand.title} | {expl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-index", "show-featured", "recommend", "recommend-fixtures"]) 
    ap.add_argument("--db", default="db/articles.db")
    ap.add_argument("--id", help="article id for recommend")
    ap.add_argument("--use_mmr", action="store_true")
    ap.add_argument("--reranker_weights", help="Path to .npz of logistic reranker weights")
    args = ap.parse_args()

    db = ArticleDB(args.db)
    emb = EmbeddingSystem()
    provider = FixtureProvider()

    if args.cmd == "build-index":
        cmd_build_index(db, emb)
    elif args.cmd == "show-featured":
        cmd_show_featured(provider)
    elif args.cmd == "recommend":
        if not args.id:
            print("--id is required for recommend")
            return
        # Optional reranker attach
        if args.reranker_weights:
            try:
                import numpy as np
                model = TrainableLogisticReranker()
                data = np.load(args.reranker_weights)
                model.w = data["w"]
                # monkey-patch into recommender inside call
                rec = AIRecommender(db, emb, RecommendationConfig(use_mmr=args.use_mmr))
                rec.set_logistic_reranker(model)
                art = db.get_article_by_id(args.id)
                if not art:
                    print("Article not found:", args.id)
                    return
                out = rec.recommend_for_article(art, k=5)
                for cand, score in out:
                    expl = rec.explain_recommendation(art, cand)
                    print(f"{score:.3f} | {cand.title} | {expl}")
                return
            except Exception as e:
                print("Failed to load reranker weights:", e)
        cmd_recommend(db, emb, args.id, args.use_mmr)
    elif args.cmd == "recommend-fixtures":
        # Recommend using fixtures + FAISS only (no DB lookups)
        import asyncio
        featured, candidates = asyncio.run(provider.fetch_featured_and_candidates())
        all_articles = {a.id: a for a in (featured + candidates)}
        if not args.id or args.id not in all_articles:
            print("--id must be one of the fixture article IDs. Use show-featured or print IDs from your loader.")
            return
        # Ensure FAISS is built from fixtures
        # Users should have run scripts/index_fixtures_to_faiss.py beforehand
        rec = AIRecommender(db, emb, RecommendationConfig(use_mmr=args.use_mmr))
        seed = all_articles[args.id]
        # Optional reranker attach (fixtures path)
        if args.reranker_weights:
            try:
                import numpy as np
                model = TrainableLogisticReranker()
                data = np.load(args.reranker_weights)
                model.w = data["w"]
                rec.set_logistic_reranker(model)
            except Exception as e:
                print("Failed to load reranker weights:", e)
        out = rec.recommend_for_article(seed, k=5)
        for cand, score in out:
            expl = rec.explain_recommendation(seed, cand)
            print(f"{score:.3f} | {cand.title} | {expl}")


if __name__ == "__main__":
    main()


