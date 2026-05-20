import argparse, json, yaml
from datetime import datetime
import hashlib
import networkx as nx
import numpy as np
import os
from pathlib import Path
import pandas as pd
import sys
import pickle
import torch
import torch_geometric
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import sklearn
import gensim
from gnn._base import log, DEVICE, CATEGORICAL_NODE_FEATURES, RELATION_TYPES, _ENSEMBLE_MODES, _CHECKPOINT_FILE
from gnn.features import NodeFeatureEncoder, NodeEmbedder, build_homogeneous_graph, train_baselines, train_xlmr_baseline
from gnn.models import build_gcn, build_sage, build_gat, build_rgcn, EdgeTypeHead, POSClassifierHead, RGCNLinkPredictor, EnsembleVariant
from gnn.training import train_link_predictor, run_ensemble_ablation, set_reproducible_seed
from gnn.evaluation import evaluate_embeddings, derivational_analogy_test, aggregate_seed_results, print_ci_summary, _flatten_result


# =========================== CONFIG MANAGER ===========================
def load_config(config_path: str) -> dict:
    """
    Loads configuration file.
    """
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    CLI args override YAML values when explicitly provided.
    """
    if args.seeds is not None and len(args.seeds) > 0:
        config.setdefault("global", {})["seed"]  = args.seeds[0]
        config["global"]["seeds"] = args.seeds
    return config


# =========================== DATA LOADING ===========================
def load_graph_data(nodes_path: Path, edges_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and cleans node and edge data for graph construction.
    """
    nodes = pd.read_csv(nodes_path, encoding="utf-8-sig").fillna("")
    edges = pd.read_csv(edges_path, encoding="utf-8-sig").fillna("")
    nodes = nodes.sort_values(by="word").drop_duplicates(subset=["word"], keep="first").reset_index(drop=True)
    word_set = set(nodes["word"])
    edges = edges[edges["source"].isin(word_set) | edges["target"].isin(word_set)]
    edges = edges.drop_duplicates(subset=["source", "relation", "target"]).reset_index(drop=True)

    nom_pairs = set(zip(edges[edges["relation"] == "nominalization"]["source"], edges[edges["relation"] == "nominalization"]["target"]))
    mask = ~((edges["relation"] == "derives_noun") & edges.apply(lambda r: (r["source"], r["target"]) in nom_pairs, axis=1))
    edges = edges[mask].reset_index(drop=True)

    nodes = nodes.sort_values(by="word").reset_index(drop=True)
    edges = edges.sort_values(by=["source", "relation", "target"]).reset_index(drop=True)

    all_edge_words = set(edges["source"]) | set(edges["target"])
    missing = all_edge_words - set(nodes["word"])

    if missing:
        frac = len(missing) / (len(nodes) + len(missing))
        log.info(
            f"Placeholder nodes: {len(missing)} ({frac:.1%}); added with pos='unknown', excluded from POS task only.")
        if frac > 0.10:
            log.warning("Placeholder fraction > 10% — investigate data pipeline.")
        placeholders = pd.DataFrame({"word": sorted(missing), "pos": "unknown", "definition_hy": "", "animacy": "", "declension_class": "", "verb_transitivity": "", "aktionsart": "", "scraped_at": ""})
        nodes = pd.concat([nodes, placeholders], ignore_index=True)

    connected = set(edges["source"]) | set(edges["target"])
    n_before = len(nodes)
    nodes = nodes[nodes["word"].isin(connected)].reset_index(drop=True)

    log.info(f"Loaded {len(nodes)} connected nodes ({n_before - len(nodes)} isolated dropped), {len(edges)} edges | avg degree: {len(edges) / max(len(nodes), 1):.2f}")
    return nodes, edges


def load_custom_negatives(neg_dir: Path, word2idx: dict, existing_edges: set, split: str) -> Optional[torch.Tensor]:
    """
    Loads pre-built negative edges (columns: source, target).
    """
    fname_map = {"train": "negative_samples_train.csv", "val": "negative_samples_val.csv", "test": "negative_samples_test.csv"}
    path = neg_dir / fname_map[split]
    if not path.exists():
        log.warning(f"Custom negative file not found: {path}; falling back to random sampling")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    before = len(df)
    df = df[df["source"].isin(word2idx) & df["target"].isin(word2idx)]
    df = df[~df.apply(lambda r: (word2idx[r["source"]], word2idx[r["target"]]) in existing_edges, axis=1)]
    log.info(f"Custom negatives [{split}]: {before} loaded -> {len(df)} kept after filtering")

    if len(df) == 0:
        log.warning(f"No valid custom negatives for split '{split}'; falling back to random sampling")
        return None

    src = df["source"].map(word2idx).tolist()
    tgt = df["target"].map(word2idx).tolist()
    return torch.tensor([src, tgt], dtype=torch.long)


# =========================== TOPOLOGY ANALYSIS ===========================
def analyze_topology(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """
    Computes network science metrics for the graph.
    """
    log.info("Performing topology analysis...")
    G = nx.DiGraph()
    G.add_nodes_from(nodes["word"].tolist())
    G.add_edges_from(zip(edges["source"].tolist(), edges["target"].tolist()))

    wccs = list(nx.weakly_connected_components(G))
    largest_wcc = max(wccs, key=len)

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "n_weakly_connected": len(wccs),
        "n_strongly_connected": nx.number_strongly_connected_components(G),
        "largest_wcc_size": len(largest_wcc),
        "largest_wcc_frac": len(largest_wcc) / G.number_of_nodes()
    }

    in_deg  = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]

    stats["avg_in_degree"] = float(np.mean(in_deg))
    stats["avg_out_degree"] = float(np.mean(out_deg))
    stats["max_in_degree"] = int(np.max(in_deg))
    stats["max_out_degree"] = int(np.max(out_deg))

    rel_counts = edges["relation"].value_counts().to_dict()
    stats["relation_counts"] = rel_counts
    pos_counts = nodes["pos"].value_counts().to_dict()
    stats["pos_counts"] = pos_counts

    for k, v in stats.items():
        if not isinstance(v, dict):
            log.info(f"{k}: {v}")

    for rel, cnt in rel_counts.items():
        log.info(f"Relation '{rel}': {cnt}")

    return stats


# =========================== PER-SEED RUNNER ===========================
def run_single_seed(seed: int, args: argparse.Namespace, config: dict, nodes: pd.DataFrame, edges: pd.DataFrame, encoder: NodeFeatureEncoder, data, word_list: List[str], baseline_results: dict, ft_emb, xlmr_emb, custom_negatives, run_dir: Path, ensemble_modes: tuple = ("gate",)) -> dict:
    """
    Runs the full GNN + ensemble pipeline for a single seed.
    Returns all_results dict for this seed.
    """
    log.info(f"\n{'='*60}\nRUNNING SEED {seed}\n{'='*60}")
    set_reproducible_seed(seed)

    all_results = {"config": vars(args), "seed": seed}
    all_results["baselines"] = baseline_results

    _probe_embedder = NodeEmbedder(encoder, out_dim=1)  # out_dim irrelevant for dim probe
    FEAT_DIM = (len(CATEGORICAL_NODE_FEATURES) * encoder.CAT_DIM + _probe_embedder.trigram_encoder.out_dim)
    n_pos_classes = int(encoder.vocab_sizes["pos"])

    def _build_edge_type_head(cfg):
        if not cfg.get("edge_type_aux", False):
            return None
        return EdgeTypeHead(embed_dim=cfg["embed_dim"], n_relation_types=len(RELATION_TYPES) + 1, hidden_dim=cfg.get("et_head_hidden", 128), dropout=cfg.get("et_head_dropout", 0.2))

    def _build_pos_head(cfg):
        if not cfg.get("pos_aux", False):
            return None
        return POSClassifierHead(embed_dim=cfg["embed_dim"], n_pos_classes=n_pos_classes, dropout=cfg.get("pos_head_dropout", 0.2))

    def _build_model(gnn_name, cfg, enc, feat_dim, num_relations):
        if gnn_name == "GCN":
            return build_gcn(enc, feat_dim, hidden_dim=cfg["hidden_dim"], embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])
        if gnn_name == "GraphSAGE":
            return build_sage(enc, feat_dim, hidden_dim=cfg["hidden_dim"], embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])
        if gnn_name == "GAT":
            return build_gat(enc, feat_dim, hidden_dim=cfg["hidden_dim"], embed_dim=cfg["embed_dim"], heads=cfg["heads"], dropout=cfg["dropout"])
        if gnn_name == "RGCN":
            return build_rgcn(enc, feat_dim, hidden_dim=cfg["hidden_dim"], embed_dim=cfg["embed_dim"], num_relations=num_relations, num_bases=cfg["num_bases"], dropout=cfg["dropout"])
        raise ValueError(f"Unknown model name: {gnn_name!r}")

    def _encode_model(model, split):
        mp_ei = split["mp_edge_index"].to(DEVICE)
        if isinstance(model, RGCNLinkPredictor):
            mp_et = split.get("mp_edge_type")
            if mp_et is None:
                mp_et = data.mp_edge_type
                log.warning("_encode_model: mp_edge_type missing from split; using data.mp_edge_type as fallback")
            local = Data()
            local.num_nodes = data.num_nodes
            local.cat_ids = data.cat_ids
            local.trigram_ids = data.trigram_ids
            local.offsets = data.offsets
            local.pos_labels = data.pos_labels
            local.word2idx = data.word2idx
            local.idx2word = data.idx2word
            local.mp_edge_index = mp_ei
            local.mp_edge_type = mp_et.to(DEVICE)
            z = model.encode(local, mp_ei, DEVICE).detach().cpu().numpy()
        else:
            z = model.encode(data, mp_ei, DEVICE).detach().cpu().numpy()
        return z

    GNN_REGISTRY = [
        ("GCN", "gcn", "results_gcn.json", "embeddings_gcn"),
        ("GraphSAGE", "sage", "results_sage.json", "embeddings_sage"),
        ("GAT", "gat", "results_gat.json", "embeddings_gat"),
        ("RGCN", "rgcn", "results_rgcn.json", "embeddings_rgcn")
    ]

    trained_gnns = {}
    for gnn_name, cfg_key, result_file, emb_stem in GNN_REGISTRY:
        if args.model_only and gnn_name != args.model_only:
            continue
        torch.manual_seed(seed)
        cfg = config["models"][cfg_key]
        model = _build_model(gnn_name, cfg, encoder, FEAT_DIM, data.num_relations)
        et_head = _build_edge_type_head(cfg)
        pos_head = _build_pos_head(cfg)

        if et_head is not None:
            log.info(f"[{gnn_name}] edge-type auxiliary head enabled (aux_weight={cfg['aux_weight']})")
        if pos_head is not None:
            log.info(f"[{gnn_name}] POS auxiliary head enabled (pos_weight={cfg.get('pos_weight', 0.2)})")

        model, history, split = train_link_predictor(model, data, n_epochs=cfg["epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], edge_type_head=et_head, pos_head=pos_head, pos_weight=cfg.get("pos_weight", 0.2), aux_weight=cfg.get("aux_weight", 0.3), seed=seed, name=gnn_name, device=DEVICE, custom_negatives=custom_negatives)

        z = _encode_model(model, split)
        save_embeddings(z, word_list, run_dir / f"{emb_stem}.csv")
        np.save(run_dir / f"{emb_stem}.npy", z)

        train_slice_idx = split["train_slice"].cpu().numpy()
        train_edges_df = edges.iloc[train_slice_idx].reset_index(drop=True)

        result = _flatten_result(evaluate_embeddings(z, nodes, gnn_name, seed=seed), history=history, analogy=derivational_analogy_test(z, nodes, train_edges_df, gnn_name))
        all_results[gnn_name] = result
        save_results(result, run_dir / result_file)

        trained_gnns[gnn_name] = {"model": model, "z": z, "history": history, "split": split}
        log.info(
            f"[{gnn_name}] training finished: "
            f"AUC={history['test_auc']:.4f} "
            f"AP={history.get('test_ap', float('nan')):.4f} "
            f"F1={history.get('test_f1', float('nan')):.4f} "
            f"MRR={history['test_mrr']:.4f} "
            f"Hits@1={history.get('test_hits1', float('nan')):.4f} "
            f"Hits@5={history.get('test_hits5', float('nan')):.4f} "
            f"Hits@10={history['test_hits10']:.4f}"
        )

    def gnn_score(kv):
        _, entry = kv
        return entry["history"]["test_mrr"]

    best_gnn_name, best_entry = max(trained_gnns.items(), key=gnn_score)

    best_gnn_z = best_entry["z"]
    best_gnn_split = best_entry["split"]
    log.info(f"Best GNN for ensemble: {best_gnn_name} (MRR={best_entry['history']['test_mrr']:.4f})")

    aux_emb  = None
    aux_name = "none"
    if ft_emb is not None and xlmr_emb is not None:
        ft_ari = baseline_results.get("FastText", {}).get("ari", -1.0)
        xlmr_ari = baseline_results.get("XLM-R", {}).get("ari", -1.0)
        if xlmr_ari > ft_ari:
            aux_emb, aux_name = xlmr_emb, "XLM-R"
        else:
            aux_emb, aux_name = ft_emb, "FastText"
    elif xlmr_emb is not None:
        aux_emb, aux_name = xlmr_emb, "XLM-R"
    elif ft_emb is not None:
        aux_emb, aux_name = ft_emb, "FastText"
    else:
        log.warning("No auxiliary embeddings available; skipping ensemble")

    ens_cfg = config["models"]["ensemble"]

    if aux_emb is not None:
        log.info(f"Building ensemble with {aux_name}")

        required_split_keys = ("train_pos", "test_pos", "test_neg")
        missing_keys = [k for k in required_split_keys if k not in best_gnn_split]
        if missing_keys:
            log.warning(f"best_gnn_split is missing keys {missing_keys}; recomputing split from data")
            best_gnn_split = None

        ensemble_results = run_ensemble_ablation(best_gnn_z, aux_emb, data, run_dir, device=DEVICE, precomputed_split=best_gnn_split, n_epochs=ens_cfg["epochs"], lr=ens_cfg["lr"], seed=seed, modes=ensemble_modes)

        gnn_t = torch.tensor(best_gnn_z, dtype=torch.float32).to(DEVICE)
        aux_t = torch.tensor(aux_emb, dtype=torch.float32).to(DEVICE)

        best_checkpoint = torch.load(run_dir / "ensemble_best.pt")
        best_mode = best_checkpoint["mode"]
        best_model = EnsembleVariant(best_gnn_z.shape[1], aux_emb.shape[1], best_gnn_z.shape[1], mode=best_mode).to(DEVICE)
        best_model.load_state_dict(best_checkpoint["state_dict"])
        best_model.eval()

        with torch.no_grad():
            ens_z = best_model(gnn_t, aux_t).detach().cpu().numpy()

        best_train_slice = best_entry["split"]["train_slice"].cpu().numpy()
        best_train_edges_df = edges.iloc[best_train_slice].reset_index(drop=True)
        best_mode_metrics = ensemble_results.get(best_mode, {})

        ens_eval = evaluate_embeddings(ens_z, nodes, "Ensemble", seed=seed)
        ens_analogy = derivational_analogy_test(ens_z, nodes, best_train_edges_df, "Ensemble")

        all_results["Ensemble"] = dict(ens_eval)
        all_results["Ensemble"]["analogy"] = ens_analogy
        all_results["Ensemble"].update({
            "auc": best_mode_metrics.get("auc", float("nan")),
            "ap": best_mode_metrics.get("ap", float("nan")),
            "f1": best_mode_metrics.get("f1", float("nan")),
            "precision": best_mode_metrics.get("precision", float("nan")),
            "recall": best_mode_metrics.get("recall", float("nan")),
            "mrr": best_mode_metrics.get("mrr", float("nan")),
            "hits1": best_mode_metrics.get("hits1", float("nan")),
            "hits5": best_mode_metrics.get("hits5", float("nan")),
            "hits10": best_mode_metrics.get("hits10", float("nan")),
            "ensemble_results": ensemble_results,
            "best_mode": best_mode,
            "aux_embedding": aux_name
        })
        save_results(all_results["Ensemble"], run_dir / "results_ensemble.json")

    log.info("Saving run artifacts...")
    _SPLIT_KEYS = ("mp_edge_index", "val_pos", "val_neg", "train_pos", "test_pos", "test_neg", "mp_edge_type", "train_slice", "train_edge_type")
    def _to_cpu(v):
        return v.cpu() if isinstance(v, torch.Tensor) else v

    splits_dict = {gnn_name: {k: _to_cpu(entry["split"][k]) for k in _SPLIT_KEYS if k in entry["split"]} for gnn_name, entry in trained_gnns.items()}
    model_states = {gnn_name: entry["model"].state_dict() for gnn_name, entry in trained_gnns.items()}

    data_cpu = data.clone().to("cpu")
    if hasattr(data_cpu, "mp_edge_index"):
        data_cpu.mp_edge_index = data.mp_edge_index.cpu()
    if hasattr(data_cpu, "mp_edge_type"):
        data_cpu.mp_edge_type = data.mp_edge_type.cpu()
    data_cpu.word2idx = dict(data.word2idx)
    data_cpu.idx2word = dict(data.idx2word)

    save_run_artifacts(run_dir=run_dir, args=args, data=data_cpu, splits=splits_dict, model_states=model_states, seed=seed)

    summary_data = []
    for m, r in all_results.items():
        if not isinstance(r, dict):
            continue
        if m == "baselines":
            for bl_name, bl_r in r.items():
                summary_data.append({
                    "model": bl_name,
                    "test_auc": bl_r.get("auc", float("nan")),
                    "test_ap": bl_r.get("ap", float("nan")),
                    "test_f1": bl_r.get("f1", float("nan")),
                    "test_precision": bl_r.get("precision", float("nan")),
                    "test_recall": bl_r.get("recall", float("nan")),
                    "test_mrr": bl_r.get("mrr", float("nan")),
                    "test_hits1": bl_r.get("hits1", float("nan")),
                    "test_hits5": bl_r.get("hits5", float("nan")),
                    "test_hits10": bl_r.get("hits10", float("nan")),
                    "embed_ari": bl_r.get("ari", float("nan"))
                })
        elif "ari" in r:
            summary_data.append({
                "model": m,
                "test_auc": r.get("auc", float("nan")),
                "test_ap": r.get("ap", float("nan")),
                "test_f1": r.get("f1", float("nan")),
                "test_precision": r.get("precision", float("nan")),
                "test_recall": r.get("recall", float("nan")),
                "test_threshold": r.get("threshold", float("nan")),
                "test_mrr": r.get("mrr", float("nan")),
                "test_hits1": r.get("hits1", float("nan")),
                "test_hits5": r.get("hits5", float("nan")),
                "test_hits10": r.get("hits10", float("nan")),
                "embed_ari": r.get("ari", float("nan"))
            })

    summary = pd.DataFrame(summary_data)
    summary.to_csv(run_dir / "summary.csv", index=False)
    log.info(f"\n{summary.to_string(index=False)}")
    save_results(all_results, run_dir / "all_results.json")
    log.info(f"Seed {seed} results saved to: {run_dir}")

    return all_results


# =========================== CHECKPOINTING ===========================
def load_checkpoint(out_dir: Path) -> dict:
    """Loads seed-level checkpoint from out_dir/checkpoint.json."""
    cp_path = out_dir / _CHECKPOINT_FILE
    if cp_path.exists():
        with open(cp_path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_seeds": [], "seed_run_dirs": {}}


def save_checkpoint(out_dir: Path, checkpoint: dict) -> None:
    """Persists checkpoint to out_dir/checkpoint.json."""
    cp_path = out_dir / _CHECKPOINT_FILE
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    log.info(f"Checkpoint updated: {cp_path}")


# =========================== I/O ===========================
def _stable_hash(obj):
    """
    Deterministic hash independent of PYTHONHASHSEED.
    """
    return int(hashlib.md5(obj.encode("utf-8")).hexdigest(), 16)


def save_embeddings(embeddings: np.ndarray, word_list: List[str], path: Path) -> None:
    """
    Saves embedding matrix to CSV indexed by word.
    """
    pd.DataFrame(embeddings, index=word_list).rename_axis("word").to_csv(path, encoding="utf-8-sig")
    log.info(f"Saved embeddings at {path}")


def save_results(results: dict, path: Path) -> None:
    """
    Serializes metrics/results dict to formatted JSON.
    """
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=_Encoder)


def save_run_artifacts(run_dir: Path, args: argparse.Namespace, data: Data, splits: dict, model_states: dict, seed: int) -> None:
    """
    Saves everything needed to exactly reproduce or audit a run.
    """
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    effective_config_path = artifacts_dir / "config_used.yaml"
    with open(effective_config_path, "w", encoding="utf-8") as _f:
        yaml.dump(vars(args) | {"_effective_config": True}, _f, allow_unicode=True)
    with open(artifacts_dir / "config_merged.yaml", "w", encoding="utf-8") as _f:
        yaml.dump(dict(vars(args)), _f, allow_unicode=True)

    runtime = {
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "command_line_args": vars(args),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "gensim_version": gensim.__version__,
        "networkx_version": nx.__version__,
        "platform": sys.platform
    }
    with open(run_dir / "runtime.json", "w") as f:
        json.dump(runtime, f, indent=2)

    data_to_hash = {
        'edge_index': data.edge_index,
        'mp_edge_index': data.mp_edge_index,
        'edge_type': data.edge_type,
        'mp_edge_type': data.mp_edge_type,
        'num_nodes': data.num_nodes,
        'word2idx': data.word2idx
    }
    hash_val = hashlib.sha256(pickle.dumps(data_to_hash)).hexdigest()
    with open(artifacts_dir / "data_hash.txt", "w") as f:
        f.write(hash_val)
    log.info(f"Data hash: {hash_val}")

    torch.save(splits, artifacts_dir / "splits.pt")
    log.info(f"Saved splits at {artifacts_dir / 'splits.pt'}")

    torch.save(model_states, artifacts_dir / "model_states.pt")
    log.info(f"Saved model states at {artifacts_dir / 'model_states.pt'}")

    log.info(f"All artifacts saved to {artifacts_dir}")


# =========================== PIPELINE ===========================
def main() -> None:
    """
    Runs the full training, evaluation, and artifact export pipeline.
    """
    parser = argparse.ArgumentParser(description="HyMorph GNN pipeline", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes",  type=str, required=True, help="Path to nodes.csv")
    parser.add_argument("--edges",  type=str, required=True, help="Path to edges.csv")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent.parent / "config.yaml"), help="Path to YAML config")
    parser.add_argument("--out-dir", type=str, default="./results", help="Directory to write results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], metavar="SEED", help=("One or more random seeds, e.g. --seeds 42 123 7 99 0."))
    parser.add_argument("--ci-level", type=float, default=0.95, metavar="LEVEL", help=("Confidence level for interval reporting when --seeds has >1 value"))
    parser.add_argument("--ci-method", choices=["bootstrap", "t"], default="bootstrap", help=("CI estimation method (default: bootstrap)"))
    parser.add_argument("--skip-baselines", action="store_true", help="Skip Node2Vec / FastText baseline training")
    parser.add_argument("--skip-topology", action="store_true", help="Skip topology analysis")
    parser.add_argument("--load-baselines", type=str, default=None, help="Path to existing results_baselines.json; reuse instead of retraining")
    parser.add_argument("--load-topology", type=str, default=None,  help="Path to existing topology.json; reuse instead of recomputing")
    parser.add_argument("--neg-dir", type=str, default=None, help="Directory containing negatives_train.csv, negatives_val.csv, negative_test.csv.")
    parser.add_argument("--model-only", type=str, default=None, choices=["GCN", "GraphSAGE", "GAT", "RGCN"], help="Train only a single model")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    seeds = args.seeds
    log.info(f"Seeds to run: {seeds}")
    first_seed = seeds[0]
    os.environ["PYTHONHASHSEED"] = str(first_seed)
    set_reproducible_seed(first_seed)
    log.info(f"Using device: {DEVICE}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_graph_data(Path(args.nodes), Path(args.edges))
    word_list = nodes["word"].tolist()
    shared_dir = out_dir / "shared"
    shared_dir.mkdir(exist_ok=True)

    topology_result = {}
    if args.load_topology:
        with open(args.load_topology, encoding="utf-8") as f:
            topology_result = json.load(f)
    elif not args.skip_topology:
        topology_result = analyze_topology(nodes, edges)
        save_results(topology_result, shared_dir / "topology.json")

    encoder = NodeFeatureEncoder(nodes)
    data = build_homogeneous_graph(nodes, edges, encoder)

    baseline_results = {}
    ft_emb = None
    xlmr_emb = None

    if args.load_baselines:
        with open(args.load_baselines, encoding="utf-8") as f:
            baseline_results = json.load(f)
        bl_dir = Path(args.load_baselines).parent
        if (bl_dir / "embeddings_node2vec.csv").exists():
            n2v_emb = pd.read_csv(bl_dir / "embeddings_node2vec.csv", index_col=0, encoding="utf-8-sig").reindex(
                word_list).fillna(0.0).values.astype(np.float32)
        if (bl_dir / "embeddings_fasttext.csv").exists():
            ft_emb = pd.read_csv(bl_dir / "embeddings_fasttext.csv", index_col=0, encoding="utf-8-sig").reindex(
                word_list).fillna(0.0).values.astype(np.float32)
        if (bl_dir / "embeddings_xlmr.csv").exists():
            xlmr_emb = pd.read_csv(bl_dir / "embeddings_xlmr.csv", index_col=0, encoding="utf-8-sig").reindex(
                word_list).fillna(0.0).values.astype(np.float32)
            log.info("Loaded XLM-R embeddings from cache.")
    elif not args.skip_baselines:
        n2v_emb, ft_emb, _ = train_baselines(nodes, edges, embed_dim=config["baselines"]["embed_dim"], seed=first_seed)
        save_embeddings(n2v_emb, word_list, shared_dir / "embeddings_node2vec.csv")
        np.save(shared_dir / "embeddings_node2vec.npy", n2v_emb)
        save_embeddings(ft_emb, word_list, shared_dir / "embeddings_fasttext.csv")
        np.save(shared_dir / "embeddings_fasttext.npy", ft_emb)
        for bl_name, bl_emb in [("Node2Vec", n2v_emb), ("FastText", ft_emb)]:
            baseline_results[bl_name] = _flatten_result(
                evaluate_embeddings(bl_emb, nodes, bl_name, seed=first_seed),
                history=None,
                analogy=derivational_analogy_test(bl_emb, nodes, edges, bl_name)
            )
        xlmr_model_name = config.get("baselines", {}).get("xlmr_model", "xlm-roberta-base")
        xlmr_embed_dim = config.get("baselines", {}).get("xlmr_embed_dim", 256)
        xlmr_emb, _ = train_xlmr_baseline(nodes, embed_dim=xlmr_embed_dim, model_name=xlmr_model_name, seed=first_seed, device=DEVICE)
        if xlmr_emb is not None:
            save_embeddings(xlmr_emb, word_list, shared_dir / "embeddings_xlmr.csv")
            np.save(shared_dir / "embeddings_xlmr.npy", xlmr_emb)
            baseline_results["XLM-R"] = _flatten_result(evaluate_embeddings(xlmr_emb, nodes, "XLM-R", seed=first_seed), history=None, analogy=derivational_analogy_test(xlmr_emb, nodes, edges, "XLM-R"))
        else:
            log.warning("XLM-R embeddings not available; will use FastText for ensemble.")
        save_results(baseline_results, shared_dir / "results_baselines.json")

    custom_negatives = None
    if args.neg_dir is not None:
        neg_dir = Path(args.neg_dir)
        full_existing = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        custom_negatives = {split: load_custom_negatives(neg_dir, data.word2idx, full_existing, split) for split in ("train", "val", "test")}

    checkpoint = load_checkpoint(out_dir)
    completed_seeds = checkpoint.get("completed_seeds", [])
    seed_run_dirs = checkpoint.get("seed_run_dirs", {})
    per_seed_results = []

    for done_seed in completed_seeds:
        done_dir = Path(seed_run_dirs[str(done_seed)])
        results_path = done_dir / "all_results.json"
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                per_seed_results.append(json.load(f))
            log.info(f"Loaded cached results for seed {done_seed} from {done_dir}")
        else:
            log.warning(f"Checkpoint listed seed {done_seed} as done but all_results.json missing at {done_dir}; will re-run")
            completed_seeds.remove(done_seed)

    first_seed_for_ablation = seeds[0]
    for seed in seeds:
        if seed in completed_seeds:
            log.info(f"Seed {seed} already completed (checkpoint). Skipping.")
            continue
        is_first = (seed == first_seed_for_ablation)
        if is_first:
            ensemble_modes = _ENSEMBLE_MODES
            log.info(f"Seed {seed} is the first seed — running full ensemble ablation.")
        else:
            best_mode_ckpt = checkpoint.get("best_ensemble_mode", "gate")
            ensemble_modes = (best_mode_ckpt,)
            log.info(f"Seed {seed}: single ensemble mode '{best_mode_ckpt}' (ablation already done on first seed).")

        seed_run_dir = out_dir / f"seed_{seed}"
        seed_run_dir.mkdir(parents=True, exist_ok=True)
        seed_results = run_single_seed(seed=seed, args=args, config=config, nodes=nodes, edges=edges, encoder=encoder, data=data, word_list=word_list, baseline_results=baseline_results, ft_emb=ft_emb, xlmr_emb=xlmr_emb, custom_negatives=custom_negatives, run_dir=seed_run_dir, ensemble_modes=ensemble_modes)

        if is_first:
            ens_res = seed_results.get("Ensemble", {}).get("ensemble_results", {})
            discovered_best = ens_res.get("best_mode")
            if discovered_best:
                checkpoint["best_ensemble_mode"] = discovered_best
                log.info(f"Ablation complete. Best ensemble mode: {discovered_best}. Stored in checkpoint.")

        per_seed_results.append(seed_results)
        completed_seeds.append(seed)
        seed_run_dirs[str(seed)] = str(seed_run_dir)
        checkpoint["completed_seeds"] = completed_seeds
        checkpoint["seed_run_dirs"] = seed_run_dirs
        save_checkpoint(out_dir, checkpoint)

    if len(seeds) > 1 and len(per_seed_results) > 1:
        ci_level  = args.ci_level
        ci_method = args.ci_method
        pct = int(round(ci_level * 100))
        log.info(f"Computing {pct}% CI ({ci_method}) across {len(per_seed_results)} seeds…")
        if ci_method == "bootstrap" and len(per_seed_results) < 5:
            log.warning( f"Only {len(per_seed_results)} seeds. Consider running at least 5 seeds for meaningful intervals.")
        agg = aggregate_seed_results(per_seed_results, level=ci_level, method=ci_method)
        print_ci_summary(agg, level=ci_level, method=ci_method)
        save_results(agg, out_dir / "ci_summary.json")

        ci_rows = []
        for model, metrics in agg.items():
            for metric, stats in metrics.items():
                if metric == "ensemble_modes" or not isinstance(stats, dict):
                    continue
                ci_rows.append({
                    "model": model,
                    "metric": metric,
                    "mean": stats.get("mean", float("nan")),
                    "std": stats.get("std", float("nan")),
                    "ci_low": stats.get("ci_low", float("nan")),
                    "ci_high": stats.get("ci_high", float("nan")),
                    "n_seeds": stats.get("n", 0),
                    "ci_level": ci_level,
                    "method": ci_method
                })
            if "ensemble_modes" in metrics:
                for mode, mode_stats in metrics["ensemble_modes"].items():
                    for metric, stats in mode_stats.items():
                        ci_rows.append({
                            "model": f"{model}[{mode}]",
                            "metric": metric,
                            "mean": stats.get("mean", float("nan")),
                            "std": stats.get("std", float("nan")),
                            "ci_low": stats.get("ci_low", float("nan")),
                            "ci_high": stats.get("ci_high", float("nan")),
                            "n_seeds": stats.get("n", 0),
                            "ci_level": ci_level,
                            "method": ci_method
                        })
        pd.DataFrame(ci_rows).to_csv(out_dir / "ci_summary.csv", index=False)
        log.info(f"CI summary -> {out_dir / 'ci_summary.json'} and {out_dir / 'ci_summary.csv'}")
    else:
        log.info("Single seed run; skipping confidence interval computation.")

    log.info(f"All results saved under: {out_dir}")
