import numpy as np
import os
from pathlib import Path
import random
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Tuple, Optional
from gnn._base import log, DEVICE
from gnn.models import EnsembleVariant, EdgeTypeHead, _ENSEMBLE_MODES
from gnn.evaluation import evaluate_link_ranking


def set_reproducible_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets all random seeds for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDNN_DETERMINISTIC']  = '1'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def train_link_predictor(model: nn.Module, data: Data, n_epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-5, val_ratio: float = 0.1, test_ratio: float = 0.1, edge_type_head: EdgeTypeHead = None, pos_head=None, pos_weight: float = 0.2, aux_weight: float = 0.3, seed: int = 42, name: str = "model", device=None, custom_negatives: dict = None) -> Tuple[nn.Module, dict, dict]:
    """
    Unified training loop for link prediction with optional edge-type auxiliary task.
    """
    device = device or DEVICE

    local_data = Data()
    local_data.num_nodes = data.num_nodes
    local_data.edge_index = data.edge_index
    local_data.edge_type = data.edge_type
    local_data.num_relations = data.num_relations
    local_data.pos_labels = data.pos_labels
    local_data.word2idx = data.word2idx
    local_data.idx2word = data.idx2word
    local_data.cat_ids = data.cat_ids.to(device)
    local_data.trigram_ids = data.trigram_ids.to(device)
    local_data.offsets = data.offsets.to(device)

    unknown_pos_id = data.unknown_pos_id

    split = _split_edges(local_data.edge_index, local_data.num_nodes, val_ratio=val_ratio, test_ratio=test_ratio, edge_type=data.edge_type, seed=seed)
    train_pos = split["train_pos"]
    val_pos = split["val_pos"]
    val_neg = split["val_neg"]
    test_pos = split["test_pos"]
    test_neg = split["test_neg"]
    train_edge_type = split["train_edge_type"]
    existing_edges = split["existing_edges"]

    if custom_negatives is not None:
        if custom_negatives.get("val") is not None:
            val_neg = custom_negatives["val"]
            log.info(f"[{name}] Using custom val negatives: {val_neg.size(1)} pairs")
        if custom_negatives.get("test") is not None:
            test_neg = custom_negatives["test"]
            log.info(f"[{name}] Using custom test negatives: {test_neg.size(1)} pairs")

    train_mp_edge_index = torch.cat([train_pos, train_pos.flip(0)], dim=1).to(device)
    N_BASE_RELS = local_data.num_relations // 2

    if train_edge_type is None:
        train_edge_type = torch.zeros(train_pos.size(1), dtype=torch.long)
        log.warning(f"[{name}] train_edge_type is None; defaulting all edges to relation-type 0. This should only happen for non-RGCN models.")
    local_data.mp_edge_index = train_mp_edge_index
    local_data.mp_edge_type  = torch.cat([train_edge_type, train_edge_type + N_BASE_RELS]).to(device)

    model = model.to(device)
    params = list(model.parameters())

    if edge_type_head is not None:
        edge_type_head = edge_type_head.to(device)
        params += list(edge_type_head.parameters())
        train_edge_type_gpu = train_edge_type.to(device)

    if pos_head is not None:
        pos_head = pos_head.to(device)
        params += list(pos_head.parameters())

    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "aux_loss": [], "val_auc": [], "val_ap": [], "val_f1": []}
    best_val_auc = 0.0
    best_state = None
    best_aux_state = None
    best_pos_state = None
    patience = 30
    no_improve = 0

    pos_labels_np  = local_data.pos_labels.cpu().numpy()
    pos_to_arr_raw = {}
    for node_id, pos in enumerate(pos_labels_np.tolist()):
        pos_to_arr_raw.setdefault(int(pos), []).append(node_id)
    pos_to_arr = {cls: np.sort(np.array(ids, dtype=np.int64)) for cls, ids in pos_to_arr_raw.items()}

    # ── training loop ──────────────────────────────────────────────────
    for epoch in tqdm(range(1, n_epochs + 1), desc=f"{name} training"):
        model.train()
        if edge_type_head is not None:
            edge_type_head.train()
        if pos_head is not None:
            pos_head.train()
        optimizer.zero_grad()

        z = model.encode(local_data, train_mp_edge_index, device)
        pos_ei = train_pos.to(device)
        pos_score = model.decode(z, pos_ei)

        hard_neg_ei, valid_mask = hard_negative_sampling(pos_ei, pos_to_arr, pos_labels_np, local_data.num_nodes, existing_edges, seed + epoch)
        pos_score_m = pos_score[valid_mask]
        neg_score_m = model.decode(z, hard_neg_ei)[valid_mask]

        link_loss = -F.logsigmoid(pos_score_m - neg_score_m).mean()
        loss = link_loss

        if edge_type_head is not None:
            type_logits = edge_type_head(z[pos_ei[0]], z[pos_ei[1]])
            aux_loss = F.cross_entropy(type_logits, train_edge_type_gpu)
            loss = loss + aux_weight * aux_loss
            history["aux_loss"].append(float(aux_loss.item()))
        else:
            history["aux_loss"].append(0.0)

        if pos_head is not None:
            known = (local_data.pos_labels != unknown_pos_id).to(device)
            pos_loss = F.cross_entropy(pos_head(z)[known], local_data.pos_labels.to(device)[known])
            loss = loss + pos_weight * pos_loss
            history.setdefault("pos_loss", []).append(float(pos_loss.item()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        history["train_loss"].append(float(link_loss.item()))

        if epoch % 10 == 0:
            model.eval()
            if edge_type_head is not None:
                edge_type_head.eval()
            if pos_head is not None:
                pos_head.eval()

            with torch.no_grad():
                z_val = model.encode(local_data, train_mp_edge_index, device)
                pos_s_val = torch.sigmoid(model.decode(z_val, val_pos.to(device))).cpu().numpy()
                neg_s_val = torch.sigmoid(model.decode(z_val, val_neg.to(device))).cpu().numpy()
                val_metrics = compute_link_metrics(pos_s_val, neg_s_val, threshold=None)
                val_auc = val_metrics["auc"]

            history["val_auc"].append(val_auc)
            history.setdefault("val_ap", []).append(val_metrics["ap"])
            history.setdefault("val_f1", []).append(val_metrics["f1"])

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if edge_type_head is not None:
                    best_aux_state = {k: v.clone() for k, v in edge_type_head.state_dict().items()}
                if pos_head is not None:
                    best_pos_state = {k: v.clone() for k, v in pos_head.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            log.info(
                f"Epoch {epoch:4d} | link={link_loss.item():.4f}"
                + (f" aux={aux_loss.item():.4f}" if edge_type_head is not None else "")
                + f" | val_auc={val_auc:.4f}"
                + (f" [patience {no_improve}/{patience}]" if no_improve else "")
            )
            if no_improve >= patience:
                log.info(f"Early stopping triggered at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if edge_type_head is not None and best_aux_state is not None:
        edge_type_head.load_state_dict(best_aux_state)
    if pos_head is not None and best_pos_state is not None:
        pos_head.load_state_dict(best_pos_state)

    model.eval()
    if edge_type_head is not None:
        edge_type_head.eval()

    with torch.no_grad():
        z_final = model.encode(local_data, train_mp_edge_index, device)
        pos_s_val = torch.sigmoid(model.decode(z_final, val_pos.to(device))).cpu().numpy()
        neg_s_val = torch.sigmoid(model.decode(z_final, val_neg.to(device))).cpu().numpy()
        frozen_thresh = find_best_threshold(pos_s_val, neg_s_val)
        log.info(f"[{name}] Threshold frozen from val: {frozen_thresh:.4f}")

        pos_s = torch.sigmoid(model.decode(z_final, test_pos.to(device))).cpu().numpy()
        neg_s = torch.sigmoid(model.decode(z_final, test_neg.to(device))).cpu().numpy()
        test_metrics = compute_link_metrics(pos_s, neg_s, threshold=frozen_thresh)
        auc = test_metrics["auc"]

        mrr, hits1, hits5, hits10 = evaluate_link_ranking(z_final, test_pos.to(device), local_data.num_nodes, existing_edges=existing_edges, decoder=model.decoder, seed=seed)

    log.info(
        f"[{name}] Test AUC={auc:.4f} AP={test_metrics['ap']:.4f} "
        f"F1={test_metrics['f1']:.4f} P={test_metrics['precision']:.4f} "
        f"R={test_metrics['recall']:.4f} "
        f"(thresh={test_metrics.get('threshold', 0.5):.3f}) "
        f"| MRR={mrr:.4f} | Hits@1={hits1:.4f} "
        f"| Hits@5={hits5:.4f} | Hits@10={hits10:.4f}"
    )

    history["test_auc"] = auc
    history["test_ap"] = test_metrics["ap"]
    history["test_f1"] = test_metrics["f1"]
    history["test_precision"] = test_metrics["precision"]
    history["test_recall"] = test_metrics["recall"]
    history["test_threshold"] = test_metrics.get("threshold", float("nan"))
    history["test_mrr"] = mrr
    history["test_hits1"] = hits1
    history["test_hits5"] = hits5
    history["test_hits10"] = hits10
    history["test_pos_scores"] = pos_s.tolist()
    history["test_neg_scores"] = neg_s.tolist()

    split_dict = {
        "mp_edge_index": train_mp_edge_index,
        "mp_edge_type": local_data.mp_edge_type.cpu(),
        "train_pos": train_pos,
        "train_edge_type": train_edge_type,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "train_slice": split["train_slice"]
    }

    return model, history, split_dict


def run_ensemble_ablation(gnn_emb: np.ndarray, aux_emb: np.ndarray, data: Data, out_dir: Path, device, n_epochs: int = 100, lr: float = 1e-3, precomputed_split: dict = None, seed: int = 42, modes: tuple = _ENSEMBLE_MODES) -> dict:
    """
    Trains every EnsembleVariant mode on the same split and seed.
    """
    gnn_dim, aux_dim, out_dim = gnn_emb.shape[1], aux_emb.shape[1], gnn_emb.shape[1]
    gnn_t = torch.tensor(gnn_emb, dtype=torch.float32).to(device)
    aux_t = torch.tensor(aux_emb, dtype=torch.float32).to(device)
    n_nodes = gnn_t.size(0)

    split = precomputed_split if precomputed_split is not None else _split_edges(data.edge_index, data.num_nodes, val_ratio=0.05, test_ratio=0.1, seed=seed)
    train_pos = split["train_pos"].to(device)
    test_pos = split["test_pos"].to(device)
    test_neg = split["test_neg"].to(device)

    existing_edges_set = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))

    pos_labels_np = data.pos_labels.cpu().numpy()
    pos_to_arr_raw: dict = {}
    for node_id, pos in enumerate(pos_labels_np.tolist()):
        pos_to_arr_raw.setdefault(int(pos), []).append(node_id)
    pos_to_arr = {cls: np.sort(np.array(ids, dtype=np.int64)) for cls, ids in pos_to_arr_raw.items()}

    results = {}
    best_auc, best_mode, best_state = 0.0, None, None

    for mode in modes:
        torch.manual_seed(seed)
        model = EnsembleVariant(gnn_dim, aux_dim, out_dim, mode=mode).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(1, n_epochs + 1):
            model.train()
            opt.zero_grad()
            pos_ei = train_pos
            pos_score = model.score(gnn_t, aux_t, pos_ei)
            hard_neg_ei, valid_mask = hard_negative_sampling(pos_ei, pos_to_arr, pos_labels_np, n_nodes, existing_edges_set, seed + epoch)
            neg_score = model.score(gnn_t, aux_t, hard_neg_ei)[valid_mask]
            loss = -F.logsigmoid(pos_score[valid_mask] - neg_score).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # ── evaluation — inside the mode loop ─────────────────────────
        model.eval()
        with torch.no_grad():
            val_ps = torch.sigmoid(model.score(gnn_t, aux_t, split["val_pos"].to(device))).cpu().numpy()
            val_ns = torch.sigmoid(model.score(gnn_t, aux_t, split["val_neg"].to(device))).cpu().numpy()
            frozen_thresh = find_best_threshold(val_ps, val_ns)

            ps = torch.sigmoid(model.score(gnn_t, aux_t, test_pos)).cpu().numpy()
            ns = torch.sigmoid(model.score(gnn_t, aux_t, test_neg)).cpu().numpy()
            ens_metrics = compute_link_metrics(ps, ns, threshold=frozen_thresh)
            auc = ens_metrics["auc"]

            z_for_rank = model(gnn_t, aux_t).cpu()
            mrr, hits1, hits5, hits10 = evaluate_link_ranking(z_for_rank, test_pos.cpu(), n_nodes, existing_edges=existing_edges_set, decoder = model.decoder, seed=seed)

        results[mode] = {
            "auc": auc, "ap": ens_metrics["ap"], "f1": ens_metrics["f1"],
            "precision": ens_metrics["precision"], "recall": ens_metrics["recall"],
            "mrr": mrr, "hits1": hits1, "hits5": hits5, "hits10": hits10,
        }
        log.info(f"[ensemble/{mode:10s}] AUC={auc:.4f} AP={ens_metrics['ap']:.4f} F1={ens_metrics['f1']:.4f} MRR={mrr:.4f} Hits@1={hits1:.4f} Hits@5={hits5:.4f} Hits@10={hits10:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_mode = mode
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save({"mode": best_mode, "state_dict": best_state}, out_dir / "ensemble_best.pt")
    log.info(f"Best ensemble mode: {best_mode} (AUC={best_auc:.4f})")
    results["best_mode"] = best_mode
    return results


def _split_edges(edge_index: torch.Tensor, num_nodes: int, val_ratio: float = 0.1, test_ratio: float = 0.1, edge_type: torch.Tensor = None, seed: int = 42) -> dict:
    """
    Splits graph edges into training, validation, and test sets and generates corresponding negative samples for link prediction.
    """
    perm_rng = torch.Generator()
    perm_rng.manual_seed(seed)
    perm = torch.randperm(edge_index.size(1), generator=perm_rng)
    n_test = max(1, int(edge_index.size(1) * test_ratio))
    n_val = max(1, int(edge_index.size(1) * val_ratio))

    train_slice = perm[n_test + n_val:]
    train_pos = edge_index[:, train_slice]
    val_pos = edge_index[:, perm[n_test: n_test + n_val]]
    test_pos = edge_index[:, perm[:n_test]]

    existing_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    def _sample_negatives(n: int, rng: np.random.Generator, n_nodes: int,
                          exclude: set) -> torch.Tensor:
        neg, seen = [], set()
        while len(neg) < n:
            u, v = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
            if u == v or (u, v) in exclude or (u, v) in seen:
                continue
            neg.append((u, v))
            seen.add((u, v))
        return torch.tensor(neg, dtype=torch.long).t()

    val_neg  = _sample_negatives(n_val,  np.random.default_rng(seed), num_nodes, existing_set)
    val_set  = set(zip(val_neg[0].tolist(), val_neg[1].tolist()))
    test_neg = _sample_negatives(n_test, np.random.default_rng(seed + 1), num_nodes, existing_set | val_set)

    return {
        "train_pos": train_pos,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "train_edge_type": edge_type[train_slice] if edge_type is not None else None,
        "existing_edges": existing_set,
        "train_slice": train_slice
    }


def find_best_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    Finds the F1-maximizing threshold on a validation split.
    Call this on validation scores; pass the result to compute_link_metrics.
    """
    from sklearn.metrics import precision_recall_curve
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(labels, scores)
    f1_arr = np.where((prec_arr[:-1] + rec_arr[:-1]) > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]), 0.0)
    return float(thresh_arr[int(np.argmax(f1_arr))])


def compute_link_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray, threshold: Optional[float] = None) -> dict:
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])

    if threshold is None:
        threshold = find_best_threshold(pos_scores, neg_scores)

    preds = (scores >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "threshold": threshold
    }


def hard_negative_sampling(pos_edge_index: torch.Tensor, pos_to_arr: dict, pos_labels_np: np.ndarray, num_nodes: int, existing_edges: set, seed: int = 42) -> tuple[Tensor, Tensor]:
    """
    Generates hard negative samples by replacing the destination with a node of the same
    POS class as the true destination, while ensuring (src, neg_dst) is not a real edge.
    """
    rng = np.random.default_rng(seed)
    src = pos_edge_index[0].cpu().numpy()
    dst = pos_edge_index[1].cpu().numpy()
    n_edges = len(dst)
    dst_pos = pos_labels_np[dst]
    neg_dst = np.empty(n_edges, dtype=np.int64)
    handled = np.zeros(n_edges, dtype=bool)

    _edge_dtype = np.dtype([("s", np.int64), ("t", np.int64)])
    _edge_set_arr = np.array(sorted(existing_edges), dtype=_edge_dtype) if existing_edges else np.array([], dtype=_edge_dtype)

    def _is_real_edge(srcs: np.ndarray, tgts: np.ndarray) -> np.ndarray:
        candidates = np.empty(len(srcs), dtype=_edge_dtype)
        candidates["s"] = srcs
        candidates["t"] = tgts
        return np.isin(candidates, _edge_set_arr)

    for pos_class, arr in pos_to_arr.items():
        mask = (dst_pos == pos_class)
        if not mask.any() or len(arr) <= 1:
            continue
        idxs = np.where(mask)[0]
        sampled = arr[rng.integers(0, len(arr), size=len(idxs))]

        for _ in range(20):
            bad = (sampled == dst[idxs]) | _is_real_edge(src[idxs], sampled)
            if not bad.any():
                break
            sampled[bad] = arr[rng.integers(0, len(arr), size=int(bad.sum()))]

        neg_dst[idxs] = sampled
        handled[idxs] = True

    if not handled.all():
        miss = np.where(~handled)[0]
        for _ in range(50):
            cands = rng.integers(0, num_nodes, size=len(miss))
            bad = (cands == dst[miss]) | _is_real_edge(src[miss], cands)
            accept = miss[~bad]
            neg_dst[accept] = cands[~bad]
            handled[accept] = True
            miss = miss[bad]
            if len(miss) == 0:
                break

        if len(miss) > 0:
            log.debug(f"hard_negative_sampling: {len(miss)} edges needed last-resort rejection sampling after 50 rounds.")
            for idx in miss:
                for _attempt in range(10_000):
                    c = int(rng.integers(0, num_nodes))
                    if c != int(dst[idx]) and not _is_real_edge(np.array([src[idx]], dtype=np.int64), np.array([c], dtype=np.int64))[0]:
                        neg_dst[idx] = c
                        handled[idx] = True
                        break
                else:
                    log.warning(f"hard_negative_sampling: could not find valid negative for edge ({src[idx]}, {dst[idx]}) after 10k attempts.")
                    neg_dst[idx] = int(dst[idx])

    neg_ei = torch.stack([torch.from_numpy(src), torch.from_numpy(neg_dst)]).to(pos_edge_index.device)
    valid_mask = torch.from_numpy(handled).to(pos_edge_index.device)
    return neg_ei, valid_mask
