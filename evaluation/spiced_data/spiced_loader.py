"""
SPICED Dataset Loader Functions (fixed)
- Adds labels (1/0)
- Prevents split leakage (split positives first; then make negatives per split)
- Generates inter/intra/hard negatives in line with the paper
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOPIC_COL = "Type"
REQ_COLS = ["text_1", "text_2", "URL_1", "URL_2", TOPIC_COL]


# ---------------------------
# utilities
# ---------------------------

def _read_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQ_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"SPICED base CSV missing columns: {missing}")
    df = df.copy()
    df["label"] = 1  # all gold pairs are similar
    return df


def _split_pos(df_pos: pd.DataFrame, train_size: float, seed: int):
    # stratify by topic when possible
    stratify = df_pos[TOPIC_COL] if df_pos[TOPIC_COL].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df_pos, train_size=train_size, random_state=seed, shuffle=True, stratify=stratify
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _make_neg_intertopic(pos_split: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    For each positive pair row r, sample a different-topic row s,
    then create a negative by pairing (r.text_1, s.text_2).
    """
    rng = np.random.RandomState(seed)
    by_topic = {t: g for t, g in pos_split.groupby(TOPIC_COL)}
    topics = list(by_topic.keys())

    rows = []
    for _, r in pos_split.iterrows():
        # choose a different topic
        other = [t for t in topics if t != r[TOPIC_COL]]
        if not other: 
            continue
        tgt_topic = other[rng.randint(len(other))]
        s = by_topic[tgt_topic].iloc[rng.randint(len(by_topic[tgt_topic]))]

        rows.append({
            "text_1": r["text_1"], "URL_1": r["URL_1"],
            "text_2": s["text_2"], "URL_2": s["URL_2"],
            TOPIC_COL: r[TOPIC_COL],
            "label": 0,
            "pair_type": "intertopic_neg"
        })
    return pd.DataFrame(rows)


def _make_neg_intratopic(pos_split: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Same-topic negatives: for each topic, mismatch text_1 with a different row's text_2.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for t, g in pos_split.groupby(TOPIC_COL):
        if len(g) < 2:
            continue
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # rotate text_2 to mismatch
        g2 = g.copy()
        g2 = g2.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        for i in range(len(g)):
            r1 = g.iloc[i]
            r2 = g2.iloc[i]
            # skip if by chance it matched the true pair
            if r1["URL_2"] == r2["URL_2"]:
                continue
            rows.append({
                "text_1": r1["text_1"], "URL_1": r1["URL_1"],
                "text_2": r2["text_2"], "URL_2": r2["URL_2"],
                TOPIC_COL: t,
                "label": 0,
                "pair_type": "intratopic_neg"
            })

    return pd.DataFrame(rows)


def _mine_hard_negatives(pos_split: pd.DataFrame, k_per_topic: int, seed: int) -> pd.DataFrame:
    """
    Hard negatives: same-topic mismatches with *high* TF-IDF cosine between (text_1, text_2).
    We mine top-K per topic for speed (adjust as needed).
    """
    rng = np.random.RandomState(seed)
    rows = []

    for t, g in pos_split.groupby(TOPIC_COL):
        if len(g) < 3:
            continue

        # build TF-IDF on candidate pool (text_1 + text_2)
        left_texts = g["text_1"].tolist()
        right_texts = g["text_2"].tolist()

        # sample to keep it light if huge
        max_pool = min(400, len(g))
        g_s = g.sample(n=max_pool, random_state=seed).reset_index(drop=True)
        L = g_s["text_1"].tolist()
        R = g_s["text_2"].tolist()

        vec = TfidfVectorizer(max_features=20000, stop_words="english")
        X_L = vec.fit_transform(L)
        X_R = vec.transform(R)
        sims = cosine_similarity(X_L, X_R)

        # remove diagonals where it might align true pairs; also avoid true URL matches
        candidates = []
        for i in range(sims.shape[0]):
            for j in range(sims.shape[1]):
                r1 = g_s.iloc[i]
                r2 = g_s.iloc[j]
                if r1["URL_2"] == r2["URL_2"]:
                    continue  # true pair
                candidates.append((sims[i, j], r1, r2))

        # pick top-K similar mismatches
        candidates.sort(key=lambda x: x[0], reverse=True)
        take = candidates[:k_per_topic]

        for sim, r1, r2 in take:
            rows.append({
                "text_1": r1["text_1"], "URL_1": r1["URL_1"],
                "text_2": r2["text_2"], "URL_2": r2["URL_2"],
                TOPIC_COL: t,
                "label": 0,
                "pair_type": "hard_neg",
                "tfidf_cosine": float(sim)
            })

    return pd.DataFrame(rows)


def _union_and_dedup(*dfs) -> pd.DataFrame:
    keep = []
    for d in dfs:
        if d is not None and len(d):
            keep.append(d)
    if not keep:
        return pd.DataFrame(columns=REQ_COLS + ["label", "pair_type"])
    out = pd.concat(keep, ignore_index=True)
    # Dedup by (URL_1, URL_2, label)
    out = out.drop_duplicates(subset=["URL_1", "URL_2", "label"]).reset_index(drop=True)
    return out


# ---------------------------
# public loaders (same API)
# ---------------------------

def load_combined(mode: str = "train", path: str = "evaluation/spiced_data/spiced.csv",
                  train_size: float = 0.7, seed: int = 42, n_hard: int = 3000) -> pd.DataFrame:
    pos = _read_gold(path)
    pos_tr, pos_te = _split_pos(pos, train_size, seed)

    if mode == "train":
        inter_neg = _make_neg_intertopic(pos_tr, seed)
        intra_neg = _make_neg_intratopic(pos_tr, seed)
        # distribute hard across topics roughly evenly
        hard_neg = _mine_hard_negatives(pos_tr, k_per_topic=max(1, n_hard // max(1, pos_tr[TOPIC_COL].nunique())), seed=seed)
        combined = _union_and_dedup(pos_tr, inter_neg, intra_neg, hard_neg)
        combined["pair_type"] = combined.get("pair_type", "combined")
        return combined
    else:
        inter_neg = _make_neg_intertopic(pos_te, seed)
        intra_neg = _make_neg_intratopic(pos_te, seed)
        hard_neg = _mine_hard_negatives(pos_te, k_per_topic=max(1, n_hard // max(1, pos_te[TOPIC_COL].nunique())), seed=seed)
        combined = _union_and_dedup(pos_te, inter_neg, intra_neg, hard_neg)
        combined["pair_type"] = combined.get("pair_type", "combined")
        return combined


def load_intertopic(mode: str = "train", path: str = "evaluation/spiced_data/spiced.csv",
                    train_size: float = 0.7, seed: int = 42) -> pd.DataFrame:
    pos = _read_gold(path)
    pos_tr, pos_te = _split_pos(pos, train_size, seed)
    if mode == "train":
        neg = _make_neg_intertopic(pos_tr, seed)
        return _union_and_dedup(pos_tr, neg)
    else:
        neg = _make_neg_intertopic(pos_te, seed)
        return _union_and_dedup(pos_te, neg)


def load_intratopic_and_hard_examples(mode: str = "train", path: str = "evaluation/spiced_data/spiced.csv",
                                      train_size: float = 0.7, seed: int = 42, n_hard: int = 3000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos = _read_gold(path)
    pos_tr, pos_te = _split_pos(pos, train_size, seed)

    if mode == "train":
        intra_neg = _make_neg_intratopic(pos_tr, seed)
        intra_set = _union_and_dedup(pos_tr, intra_neg)
        hard_neg = _mine_hard_negatives(pos_tr, k_per_topic=max(1, n_hard // max(1, pos_tr[TOPIC_COL].nunique())), seed=seed)
        return intra_set, hard_neg.reset_index(drop=True)
    else:
        intra_neg = _make_neg_intratopic(pos_te, seed)
        intra_set = _union_and_dedup(pos_te, intra_neg)
        hard_neg = _mine_hard_negatives(pos_te, k_per_topic=max(1, n_hard // max(1, pos_te[TOPIC_COL].nunique())), seed=seed)
        return intra_set, hard_neg.reset_index(drop=True)


# ---------------------------
# diagnostics
# ---------------------------

def get_dataset_stats(df: pd.DataFrame, dataset_name: str = "Dataset"):
    stats = {
        "name": dataset_name,
        "total_pairs": int(len(df)),
        "topics": int(df[TOPIC_COL].nunique()) if TOPIC_COL in df.columns else 0,
        "topic_distribution": df[TOPIC_COL].value_counts().to_dict() if TOPIC_COL in df.columns else {},
        "avg_text_length_1": float(df["text_1"].str.len().mean()) if "text_1" in df.columns else 0.0,
        "avg_text_length_2": float(df["text_2"].str.len().mean()) if "text_2" in df.columns else 0.0,
        "pos_ratio": float(df["label"].mean()) if "label" in df.columns and len(df) else 0.0,
    }
    return stats


def print_dataset_info(df: pd.DataFrame, dataset_name: str = "Dataset"):
    s = get_dataset_stats(df, dataset_name)
    print(f"\n{dataset_name} Statistics:")
    print(f"  Total pairs: {s['total_pairs']}")
    print(f"  Topics: {s['topics']}")
    print(f"  Pos ratio: {s['pos_ratio']:.3f}")
    print(f"  Avg len text_1: {s['avg_text_length_1']:.1f}")
    print(f"  Avg len text_2: {s['avg_text_length_2']:.1f}")
    if s["topic_distribution"]:
        print("  Topic distribution:")
        for k, v in s["topic_distribution"].items():
            print(f"    {k}: {v}")
