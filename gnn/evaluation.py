import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, normalize
from gnn._base import log, RELATION_TYPES, _CI_METRICS, _CI_BOOTSTRAP_ITERS, _ENSEMBLE_CI_METRICS


# =========================== EVALUATIONS ===========================
def evaluate_link_ranking(z: torch.Tensor, pos_edges: torch.Tensor, num_nodes: int, existing_edges: Optional[set] = None, decoder = None, n_corrupted: int = 300, seed: int = 42) -> Tuple[float, float, float, float]:
    assert n_corrupted >= 10
    rng = np.random.default_rng(seed)
    existing_edges = existing_edges or set()
    reciprocal_ranks, hits1, hits5, hits10 = [], [], [], []
    z_cpu = z.cpu()

    for i in range(pos_edges.size(1)):
        u, v = int(pos_edges[0, i]), int(pos_edges[1, i])
        excluded = {v} | {t for (s, t) in existing_edges if s == u}
        neg_candidates: List[int] = []
        while len(neg_candidates) < n_corrupted - 1:
            c = int(rng.integers(0, num_nodes))
            if c not in excluded:
                neg_candidates.append(c)
                excluded.add(c)
        corrupted = [v] + neg_candidates
        candidates = torch.tensor(corrupted, dtype=torch.long)
        if decoder is not None:
            u_proj = decoder.W_src(z_cpu[u].unsqueeze(0)).expand(len(candidates), -1)
            v_proj = decoder.W_tgt(z_cpu[candidates])
            scores = (u_proj * v_proj).sum(dim=1).numpy()
        else:
            u_embed = z_cpu[u].unsqueeze(0).expand(len(candidates), -1)
            scores = (u_embed * z_cpu[candidates]).sum(dim=1).numpy()

        rank = int(np.where(np.argsort(-scores) == 0)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        hits1.append(float(rank <= 1))
        hits5.append(float(rank <= 5))
        hits10.append(float(rank <= 10))

    return (float(np.mean(reciprocal_ranks)), float(np.mean(hits1)), float(np.mean(hits5)), float(np.mean(hits10)))


def evaluate_embeddings(embeddings: np.ndarray, nodes: pd.DataFrame, model_name: str, n_clusters: Optional[int] = None, seed: int = 42) -> dict:
    """
    Evaluates embeddings via clustering quality (ARI) and neighbor retrieval.
    """
    log.info(f"Embedding evaluation: {model_name}")

    normed = normalize(embeddings, norm="l2")
    pos_labels = LabelEncoder().fit_transform(nodes["pos"].values)

    k = int(nodes["pos"].nunique()) if n_clusters is None else n_clusters
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=20, init="k-means++").fit(normed)

    ari = adjusted_rand_score(pos_labels, kmeans.labels_)
    nmi = normalized_mutual_info_score(pos_labels, kmeans.labels_, average_method="arithmetic")

    max_sil = 10_000
    if len(normed) > max_sil:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(normed), max_sil, replace=False)
        sil = float(silhouette_score(normed[idx], kmeans.labels_[idx], metric="cosine"))
    else:
        sil = float(silhouette_score(normed, kmeans.labels_, metric="cosine"))

    log.info(f"K-Means k={k}: ARI={ari:.4f}  NMI={nmi:.4f}  Silhouette={sil:.4f}")

    SPOT_CHECK_SEEDS = ["գիր", "գրիչ", "գրություն", "սիրել", "սիրուն", "սիրահար", "լույս", "լուսին", "լուսավոր"]
    word_list = nodes["word"].tolist()
    word2idx = {w: i for i, w in enumerate(word_list)}
    nn_results = {}
    for seed_word in SPOT_CHECK_SEEDS:
        if seed_word not in word2idx:
            log.warning(f"Seed word '{seed_word}' not in vocabulary")
            continue
        idx = word2idx[seed_word]
        sims = normed @ normed[idx]
        top = np.argpartition(-sims, 6)[:6]
        top = top[np.argsort(-sims[top])]
        neighbours = [word_list[i] for i in top if i != idx][:5]
        nn_results[seed_word] = neighbours
        log.info(f"NN({seed_word}): {neighbours}")

    return {"ari": ari, "nmi": nmi, "silhouette": sil, "nn_results": nn_results}


def derivational_analogy_test(embeddings: np.ndarray, nodes: pd.DataFrame, edges: pd.DataFrame, model_name: str, max_pairs: int = 500) -> dict:
    """
    Evaluates derivational analogy via 3CosAdd under a unified root-membership
    metric: a prediction is correct if it falls within the set of all known
    targets of the query source under the given relation.

    For relations where a source has exactly one target this is identical to
    exact match. For one-to-many relations (compounding, reduplication, etc.)
    it correctly accepts any valid answer.
    """
    log.info(f"Derivational analogy test: {model_name}")
    word_list = nodes["word"].tolist()
    word2idx = {w: i for i, w in enumerate(word_list)}
    normed = (embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

    rel_edges = edges[edges["relation"].isin(set(RELATION_TYPES))]
    valid_targets = {(rel, src): set(grp["target"].tolist()) for (rel, src), grp in rel_edges.groupby(["relation", "source"], sort=False)}
    results = {}
    for rel_type in RELATION_TYPES:
        pairs = [(s, t) for s, t in edges[edges["relation"] == rel_type][["source", "target"]].values.tolist() if s in word2idx and t in word2idx]
        if len(pairs) < 4:
            continue

        pairs = pairs[:int(np.ceil(np.sqrt(max_pairs))) + 1]
        n = len(pairs)
        s_words = [s for s, _ in pairs]
        s_A_idx = np.array([word2idx[s] for s, _ in pairs], dtype=np.int32)
        t_A_idx = np.array([word2idx[t] for _, t in pairs], dtype=np.int32)

        ii = np.repeat(np.arange(n), n)
        jj = np.tile(np.arange(n), n)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        if len(ii) > max_pairs:
            ii, jj = ii[:max_pairs], jj[:max_pairs]

        queries = (normed[t_A_idx[ii]] - normed[s_A_idx[ii]] + normed[s_A_idx[jj]])
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        valid_q = (norms >= 1e-9).ravel()
        queries, ii, jj = queries[valid_q], ii[valid_q], jj[valid_q]
        queries /= norms[valid_q]

        total = len(ii)
        if total == 0:
            continue

        all_sims = queries @ normed.T
        rows = np.arange(total)
        all_sims[rows, s_A_idx[ii]] = -1.0
        all_sims[rows, t_A_idx[ii]] = -1.0
        all_sims[rows, s_A_idx[jj]] = -1.0

        top10_raw = np.argpartition(all_sims, -10, axis=1)[:, -10:]
        top10_scores = all_sims[rows[:, None], top10_raw]
        order = np.argsort(-top10_scores, axis=1)
        top10_idx = top10_raw[rows[:, None], order]

        unique_jj = np.unique(jj)
        src_mask = {}
        for sb in unique_jj:
            valid = valid_targets.get((rel_type, s_words[sb]), set())
            m = np.zeros(len(word_list), dtype=bool)
            for w in valid:
                if w in word2idx:
                    m[word2idx[w]] = True
            src_mask[sb] = m

        h1 = np.zeros(total, dtype=bool)
        h5 = np.zeros(total, dtype=bool)
        h10 = np.zeros(total, dtype=bool)

        for sb in unique_jj:
            qmask = jj == sb
            vmask = src_mask[sb]
            hits = vmask[top10_idx[qmask]]
            h1[qmask] = hits[:, 0]
            h5[qmask] = hits[:, :5].any(axis=1)
            h10[qmask] = hits.any(axis=1)

        results[rel_type] = {
            "acc@1": float(h1.sum())  / total,
            "acc@5": float(h5.sum())  / total,
            "acc@10": float(h10.sum()) / total,
            "metric": "root_member",
            "n": total
        }

        log.info(
            f"[{rel_type}] acc@1={results[rel_type]['acc@1']:.3f}  "
            f"acc@5={results[rel_type]['acc@5']:.3f}  "
            f"acc@10={results[rel_type]['acc@10']:.3f}  "
            f"(n={total})"
        )

    return results


# =========================== CONFIDENCE INTERVALS ===========================
def _bootstrap_ci(arr: np.ndarray, level: float, n_iter: int = _CI_BOOTSTRAP_ITERS, rng_seed: int = 0) -> Tuple[float, float, float, float]:
    """BCa bootstrap CI. Returns (mean, std, ci_low, ci_high)."""
    from scipy.special import ndtri, ndtr

    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return mean, std, mean, mean

    rng = np.random.default_rng(rng_seed)
    boot_means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_iter)])

    prop_below = float(np.clip((boot_means < mean).mean(), 1e-6, 1 - 1e-6))
    z0 = float(ndtri(prop_below))

    jack_means = np.array([np.delete(arr, i).mean() for i in range(n)])
    jack_grand = jack_means.mean()
    jack_dev = jack_grand - jack_means
    num = float((jack_dev ** 3).sum())
    denom = float(6.0 * (jack_dev ** 2).sum() ** 1.5)
    a = num / denom if abs(denom) > 1e-12 else 0.0

    alpha = 1.0 - level

    def _adj_pct(z_alpha: float) -> float:
        inner = z0 + z_alpha
        adj = z0 + inner / (1.0 - a * inner)
        return float(np.clip(ndtr(adj) * 100.0, 0.0, 100.0))

    ci_lo = float(np.percentile(boot_means, _adj_pct(float(ndtri(alpha / 2)))))
    ci_hi = float(np.percentile(boot_means, _adj_pct(float(ndtri(1 - alpha / 2)))))
    return mean, std, ci_lo, ci_hi


def _ttest_ci(arr: np.ndarray, level: float) -> Tuple[float, float, float, float]:
    """
    t-distribution CI. Returns (mean, std, ci_low, ci_high).
    """
    from scipy import stats as scipy_stats
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return mean, std, mean, mean
    se = float(scipy_stats.sem(arr))
    h = float(scipy_stats.t.ppf(1 - (1 - level) / 2, df=n - 1) * se)
    return mean, std, mean - h, mean + h


def _compute_ci(values: list, level: float, method: str) -> dict:
    """
    Compute CI for a list of scalar metric values.
    Returns a dict with keys: mean, std, ci_low, ci_high, n, level, method.
    """
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    n   = len(arr)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "ci_low": nan, "ci_high": nan, "n": 0, "level": level, "method": method, "values": values}
    if n == 1:
        v = float(arr[0])
        return {"mean": v, "std": 0.0, "ci_low": v, "ci_high": v, "n": 1, "level": level, "method": method, "values": values}

    if method == "t":
        mean, std, lo, hi = _ttest_ci(arr, level)
    else:
        mean, std, lo, hi = _bootstrap_ci(arr, level)

    return {"mean": mean, "std": std, "ci_low": lo, "ci_high": hi, "n": n, "level": level, "method": method, "values": list(arr)}


def aggregate_seed_results(per_seed_results: List[dict], level: float = 0.95, method: str = "bootstrap") -> dict:
    """
    Given a list of all_results dicts (one per seed), compute mean ± CI
    for every numeric metric for each model and each ensemble ablation mode.

    Args:
        per_seed_results: one dict per completed seed
        level:  confidence level in (0, 1), e.g. 0.95 for 95% CI
        method: "bootstrap" (BCa, default, recommended for n<30) or "t" (t-distribution)

    Returns a summary dict.
    """
    model_names = set()
    for r in per_seed_results:
        for k in r:
            if k not in ("config", "seed", "topology", "baselines"):
                model_names.add(k)
        for bl_name in r.get("baselines", {}):
            model_names.add(f"baseline/{bl_name}")

    agg = {}
    for model in sorted(model_names):
        agg[model] = {}
        is_baseline = model.startswith("baseline/")
        bl_key = model.split("/", 1)[1] if is_baseline else None
        for metric in _CI_METRICS:
            vals = []
            for r in per_seed_results:
                entry = r.get("baselines", {}).get(bl_key, {}) if is_baseline else r.get(model, {})
                v = entry.get(metric, float("nan"))
                vals.append(float(v) if v is not None else float("nan"))
            agg[model][metric] = _compute_ci(vals, level, method)
        if not is_baseline:
            ens_modes_vals: dict = {}
            for r in per_seed_results:
                ens_res = r.get(model, {}).get("ensemble_results", {})
                for mode, mode_dict in ens_res.items():
                    if mode == "best_mode" or not isinstance(mode_dict, dict):
                        continue
                    ens_modes_vals.setdefault(mode, {})
                    for m in _ENSEMBLE_CI_METRICS:
                        ens_modes_vals[mode].setdefault(m, []).append(float(mode_dict.get(m, float("nan"))))
            if ens_modes_vals:
                agg[model]["ensemble_modes"] = {}
                for mode, metrics_dict in ens_modes_vals.items():
                    agg[model]["ensemble_modes"][mode] = {}
                    for m, vals in metrics_dict.items():
                        agg[model]["ensemble_modes"][mode][m] = _compute_ci(vals, level, method)

    return agg


def print_ci_summary(agg: dict, level: float, method: str) -> None:
    pct = int(round(level * 100))
    title = f"MULTI-SEED SUMMARY  ({pct}% CI · method={method})"
    header = f"{'Model':<28} {'Metric':<12} {'Mean':>7} {'Std':>7} {'CI-Low':>8} {'CI-High':>9} {'n':>3}"
    sep = "=" * len(header)
    log.info(sep)
    log.info(title)
    log.info(header)
    log.info("-" * len(header))

    def _row(label, metric, stats):
        mean = stats.get("mean", float("nan"))
        std = stats.get("std",  float("nan"))
        lo = stats.get("ci_low",  float("nan"))
        hi = stats.get("ci_high", float("nan"))
        n = stats.get("n", "?")
        if np.isnan(mean):
            return
        log.info(f"{label:<28} {metric:<12} {mean:>7.4f} {std:>7.4f} {lo:>8.4f} {hi:>9.4f} {n:>3}")

    for model, metrics in agg.items():
        for metric, stats in metrics.items():
            if metric == "ensemble_modes" or not isinstance(stats, dict):
                continue
            _row(model, metric, stats)
        if "ensemble_modes" in metrics:
            for mode, mode_stats in metrics["ensemble_modes"].items():
                for metric, stats in mode_stats.items():
                    _row(f"{model}[{mode}]", metric, stats)
    log.info(sep)


def _flatten_result(eval_dict: dict, history: dict = None, analogy: dict = None) -> dict:
    """
    Merge evaluate_embeddings output with link prediction metrics.
    """
    out = dict(eval_dict)
    if history is not None:
        out.update({
            "history": history,
            "auc": history["test_auc"],
            "ap": history.get("test_ap", float("nan")),
            "f1": history.get("test_f1", float("nan")),
            "precision": history.get("test_precision", float("nan")),
            "recall": history.get("test_recall", float("nan")),
            "threshold": history.get("test_threshold", float("nan")),
            "mrr": history["test_mrr"],
            "hits1":  history.get("test_hits1",  float("nan")),
            "hits5":  history.get("test_hits5",  float("nan")),
            "hits10": history.get("test_hits10", float("nan"))
        })
    else:
        out.update({
            "auc": float("nan"), "ap": float("nan"),
            "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "mrr": float("nan"),
            "hits1": float("nan"), "hits5": float("nan"), "hits10": float("nan")
        })
    if analogy is not None:
        out["analogy"] = analogy
    return out
