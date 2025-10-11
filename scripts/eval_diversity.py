import argparse
import asyncio
import numpy as np

from src.embeddings import EmbeddingSystem
from src.providers.fixture import FixtureProvider
from src.recommendation_system import RecommendationSystem, RecommendationConfig


def avg_pairwise_cosine(emb: EmbeddingSystem, articles):
    if len(articles) < 2:
        return 0.0
    texts = []
    for a in articles:
        desc = a.description or ""
        head = (a.content or "")[:500]
        texts.append(f"{a.title} {desc} {head}")
    X = emb.encode_texts(texts).astype('float32', copy=False)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    n = S.shape[0]
    triu = S[np.triu_indices(n, k=1)]
    return float(np.mean(triu))


async def main_async(k: int, seeds: int):
    provider = FixtureProvider(shuffle=False)
    emb = EmbeddingSystem()
    featured, candidates = await provider.fetch_featured_and_candidates()
    pool = candidates + featured
    seeds_list = pool[:seeds]

    base_cfg = RecommendationConfig()
    mmr_cfg = RecommendationConfig()  # MMR is handled in MultiRAGRetriever

    base = RecommendationSystem(None, emb, base_cfg)  # db not needed for fixtures text
    mmr  = RecommendationSystem(None, emb, mmr_cfg)

    print("seed_id | base_cos | mmr_cos")
    for s in seeds_list:
        base_out = [a for a, _ in base.recommend_for_article(s, k=k)]
        mmr_out  = [a for a, _ in mmr.recommend_for_article(s, k=k)]
        bc = avg_pairwise_cosine(emb, base_out)
        mc = avg_pairwise_cosine(emb, mmr_out)
        print(f"{s.id} | {bc:.3f} | {mc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(main_async(args.k, args.seeds))


if __name__ == "__main__":
    main()


