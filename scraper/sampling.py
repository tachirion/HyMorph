from scraper._base import RELATION_HIERARCHY, log
from enum import Enum
from pathlib import Path
import pandas as pd
import random


# ===================== NEGATIVE SAMPLING =====================
class NegativeStrategy(str, Enum):
    RANDOM = "random"  # uniform random node corruption
    TYPED = "typed"  # preserve POS type of corrupted endpoint
    RELCORR = "relcorr"  # same relation, wrong target (hardest negatives)


class NegativeSampler:
    """
    Generates negative-sample edges for GNN link-prediction training.
    Output schema matches EDGE_FIELDS plus a 'label' column (0 = negative).
    """

    def __init__(self, ratio: int = 5, strategy: NegativeStrategy = NegativeStrategy.TYPED, seed: int = 42):
        self.ratio = ratio
        self.strategy = strategy
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def generate(self, edges_file: Path, nodes_file: Path, out_file: Path) -> pd.DataFrame:
        """
        Read positive edges and nodes from CSV; write negatives to out_file.
        Returns the negative-sample DataFrame.
        """
        pos_df = pd.read_csv(edges_file, encoding="utf-8-sig", dtype=str).fillna("")
        node_df = pd.read_csv(nodes_file, encoding="utf-8-sig", dtype=str).fillna("")

        all_words = node_df["word"].tolist()
        word_to_pos = dict(zip(node_df["word"], node_df["pos"]))

        pos_to_words = {}
        for w, p in word_to_pos.items():
            pos_to_words.setdefault(p, []).append(w)

        pos_keys = set(zip(pos_df["source"], pos_df["relation"], pos_df["target"]))

        negatives = []

        for _, row in pos_df.iterrows():
            src, rel, tgt = row["source"], row["relation"], row["target"]
            src_pos = row.get("source_pos", "unknown")
            tgt_pos = row.get("target_pos", "unknown")
            rel_cls = row.get("relation_class", RELATION_HIERARCHY.get(rel, "other"))

            for _ in range(self.ratio * 3):
                if len(negatives) and len(negatives) % (self.ratio * len(pos_df)) == 0:
                    break

                corrupt_source = self.rng.random() < 0.5

                if self.strategy == NegativeStrategy.RANDOM:
                    candidate = self.rng.choice(all_words)
                elif self.strategy == NegativeStrategy.TYPED:
                    pool_pos  = src_pos if corrupt_source else tgt_pos
                    pool = pos_to_words.get(pool_pos, all_words)
                    candidate = self.rng.choice(pool) if pool else self.rng.choice(all_words)
                else:
                    corrupt_source = False
                    pool = pos_to_words.get(tgt_pos, all_words)
                    candidate = self.rng.choice(pool) if pool else self.rng.choice(all_words)

                neg_src = candidate if corrupt_source else src
                neg_tgt = candidate if not corrupt_source else tgt

                key = (neg_src, rel, neg_tgt)
                if key in pos_keys or neg_src == neg_tgt:
                    continue
                pos_keys.add(key)

                negatives.append({
                    "source": neg_src,
                    "source_pos": word_to_pos.get(neg_src, "unknown"),
                    "relation": rel,
                    "relation_class": rel_cls,
                    "target": neg_tgt,
                    "target_pos": word_to_pos.get(neg_tgt, "unknown"),
                    "label": 0
                })

                if len(negatives) >= len(pos_df) * self.ratio:
                    break

        neg_df = pd.DataFrame(negatives)
        pos_df["label"] = 1
        combined = pd.concat([pos_df, neg_df], ignore_index=True)
        combined.to_csv(out_file, index=False, encoding="utf-8-sig")
        log.info(f"Negative sampling | positives={len(pos_df)} | negatives={len(neg_df)} | strategy={self.strategy} | ratio={self.ratio} | -> {out_file}")
        return neg_df


def split_for_gnn(combined_file: Path, train: float = 0.8, val: float = 0.1, test: float = 0.1, seed: int = 42) -> None:
    """
    Stratified split of the labeled edge file (positive + negative) into
    train / val / test sets, stratified by relation type so rare relations appear in all splits.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1"
    df = pd.read_csv(combined_file, encoding="utf-8-sig", dtype=str).fillna("")

    splits = {"train": [], "val": [], "test": []}

    for rel, group in df.groupby("relation"):
        rows = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(rows)
        i1 = int(n * train)
        i2 = i1 + int(n * val)
        splits["train"].append(rows.iloc[:i1])
        splits["val"].append(rows.iloc[i1:i2])
        splits["test"].append(rows.iloc[i2:])

    base = combined_file.parent
    for name, frames in splits.items():
        out = base / f"{combined_file.stem}_{name}.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        log.info(f"Split '{name}' -> {out} ({sum(len(f) for f in frames)} rows)")
