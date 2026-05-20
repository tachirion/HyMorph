from __future__ import annotations
from collections import Counter
from typing import Tuple
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gnn._base import log, CATEGORICAL_NODE_FEATURES, DEVICE, RELATION_TYPES


# =========================== CLASSES ===========================
class NodeFeatureEncoder:
    """
    Encodes categorical features and trigrams for node representation.
    """
    TRIGRAM_DIM = 32
    CAT_DIM = 32
    MAX_TRIGRAMS = 6000

    def __init__(self, nodes: pd.DataFrame):
        self.label_encoders = {}
        self.vocab_sizes = {}

        for col in CATEGORICAL_NODE_FEATURES:
            le = LabelEncoder()
            le.fit(nodes[col].astype(str).tolist() + ["<UNK>"])
            self.label_encoders[col] = le
            self.vocab_sizes[col] = len(le.classes_)

        trigram_counts = Counter()
        for word in nodes["word"].tolist():
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                trigram_counts[padded[i:i + 3]] += 1

        top_trigrams = sorted(trigram_counts.keys(), key=lambda tg: (-trigram_counts[tg], tg))[:self.MAX_TRIGRAMS]
        self.trigram_vocab = {tg: i + 1 for i, tg in enumerate(top_trigrams)}
        self.n_trigrams = len(self.trigram_vocab) + 1

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError("feature_dim depends on CharCNN config; use NodeEmbedder.out_dim instead.")

    def encode_categorical(self, nodes: pd.DataFrame) -> torch.Tensor:
        cols = []
        for col in CATEGORICAL_NODE_FEATURES:
            le = self.label_encoders[col]
            raw = np.array(nodes[col].astype(str).tolist())
            unk_id = int(le.transform(["<UNK>"])[0])
            known = np.isin(raw, le.classes_)
            ids = np.full(len(raw), unk_id, dtype=np.int64)
            ids[known] = le.transform(raw[known]).astype(np.int64)
            cols.append(torch.tensor(ids, dtype=torch.long))
        return torch.stack(cols, dim=1)

    def encode_trigrams(self, nodes: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        all_ids = []
        offsets = []
        for word in nodes["word"].tolist():
            offsets.append(len(all_ids))
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                all_ids.append(self.trigram_vocab.get(padded[i:i + 3], 0))

        if not all_ids:
            all_ids = [0]

        return torch.tensor(all_ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class CharCNNEncoder(nn.Module):
    """
    Lightweight character-level CNN.
    Input:  variable-length sequence of trigram IDs per word (padded).
    Output: fixed-size vector per word.
    Architecture: two parallel conv banks (widths 3 and 5) → max-pool → concat.
    """

    def __init__(self, vocab_size: int, emb_dim: int = 32, out_channels: int = 32, max_len: int = 32):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(emb_dim, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, out_channels, kernel_size=5, padding=2)
        self.out_dim = out_channels * 2   # concat of two banks

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        n_words = offsets.size(0)
        device = ids.device
        n_ids = ids.size(0)

        ends = torch.cat([offsets[1:], ids.new_tensor([n_ids])])  # (n_words,)
        lengths = (ends - offsets).clamp(min=0)  # actual trigrams per word

        padded = torch.zeros(n_words, self.max_len, dtype=torch.long, device=device)

        if n_ids > 0:
            # word_of_id[i] = which word owns ids[i]
            word_of_id = torch.repeat_interleave(torch.arange(n_words, device=device), lengths)
            # pos_in_word[i] = position of ids[i] within its word
            pos_in_word = torch.arange(n_ids, device=device) - offsets[word_of_id]
            valid = pos_in_word < self.max_len
            padded[word_of_id[valid], pos_in_word[valid]] = ids[valid]

        x = self.embedding(padded).permute(0, 2, 1)  # (B, emb_dim, L)
        h = torch.cat([
            self.conv3(x).relu().max(dim=-1).values,  # (B, out_channels)
            self.conv5(x).relu().max(dim=-1).values,
        ], dim=1)  # (B, out_dim)
        return h


class NodeEmbedder(nn.Module):
    def __init__(self, encoder: NodeFeatureEncoder, out_dim: int, use_char_cnn: bool = True):
        super().__init__()
        self.encoder = encoder
        self.out_dim  = out_dim
        self.cat_embeds = nn.ModuleList([nn.Embedding(encoder.vocab_sizes[col], encoder.CAT_DIM) for col in CATEGORICAL_NODE_FEATURES])

        if use_char_cnn:
            self.trigram_encoder = CharCNNEncoder(vocab_size=encoder.n_trigrams, emb_dim=encoder.TRIGRAM_DIM, out_channels=32)
            trig_dim = self.trigram_encoder.out_dim   # ← derived, not hardcoded
        else:
            self.trigram_encoder = nn.EmbeddingBag(encoder.n_trigrams, encoder.TRIGRAM_DIM, mode="mean", padding_idx=0)
            trig_dim = encoder.TRIGRAM_DIM

        raw_dim = len(CATEGORICAL_NODE_FEATURES) * encoder.CAT_DIM + trig_dim
        self.proj = nn.Sequential(nn.Linear(raw_dim, out_dim), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, cat_ids, trigram_ids, offsets):
        cat_vecs = [emb(cat_ids[:, i]) for i, emb in enumerate(self.cat_embeds)]
        cat_out = torch.cat(cat_vecs, dim=1)
        trigram_out = self.trigram_encoder(trigram_ids, offsets)
        return self.proj(torch.cat([cat_out, trigram_out], dim=1))


# =========================== FUNCTIONS ===========================
def build_homogeneous_graph(nodes: pd.DataFrame, edges: pd.DataFrame, encoder: NodeFeatureEncoder) -> Data:
    """
    Constructs a PyG Data object from node and edge frames.
    """
    word2idx = {w: i for i, w in enumerate(nodes["word"].tolist())}
    src_idx = edges["source"].map(word2idx).tolist()
    tgt_idx = edges["target"].map(word2idx).tolist()

    edge_index_directed = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_index_bidir = torch.cat([edge_index_directed, torch.tensor([tgt_idx, src_idx], dtype=torch.long)], dim=1)

    rel2idx = {r: i for i, r in enumerate(RELATION_TYPES)}
    UNK_REL = len(RELATION_TYPES)
    N_BASE_RELS = len(RELATION_TYPES) + 1

    edge_type_directed = torch.tensor([rel2idx.get(r, UNK_REL) for r in edges["relation"].tolist()], dtype=torch.long)
    edge_type_bidir = torch.cat([edge_type_directed, edge_type_directed + N_BASE_RELS])

    cat_ids = encoder.encode_categorical(nodes)
    trigram_ids, offsets = encoder.encode_trigrams(nodes)

    pos_le = encoder.label_encoders["pos"]
    raw = nodes["pos"].astype(str).tolist()
    pos_ids = [int(pos_le.transform([v])[0]) if v in pos_le.classes_ else int(pos_le.transform(["<UNK>"])[0]) for v in raw]

    data = Data(edge_index=edge_index_directed)
    data.mp_edge_index = edge_index_bidir
    data.edge_type = edge_type_directed
    data.mp_edge_type = edge_type_bidir
    data.num_relations = 2 * N_BASE_RELS
    data.cat_ids = cat_ids
    data.trigram_ids = trigram_ids
    data.offsets = offsets
    data.num_nodes = len(nodes)
    data.word2idx = word2idx
    data.idx2word = {v: k for k, v in word2idx.items()}
    data.pos_labels = torch.tensor(pos_ids, dtype=torch.long)

    data.unknown_pos_id = (int(pos_le.transform(["unknown"])[0]) if "unknown" in pos_le.classes_ else -1)

    return data


def train_baselines(nodes: pd.DataFrame, edges: pd.DataFrame, embed_dim: int = 128, seed: int = 42, n_walks: int = 10, walk_length: int = 40, workers: int = 4, node2vec_p: float = 1.0, node2vec_q: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Trains Node2Vec and FastText baselines over the morphological graph.
    """
    word_list = nodes["word"].tolist()
    word2idx = {w: i for i, w in enumerate(word_list)}
    n_nodes = len(word_list)

    src_idx, tgt_idx = [], []
    for _, row in edges.iterrows():
        s, t = row["source"], row["target"]
        if s in word2idx and t in word2idx:
            si, ti = word2idx[s], word2idx[t]
            src_idx += [si, ti]
            tgt_idx += [ti, si]
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)

    torch.manual_seed(seed)
    n2v_model = Node2Vec(edge_index, embedding_dim=embed_dim, walk_length=walk_length, context_size=5, walks_per_node=n_walks, p=node2vec_p, q=node2vec_q, num_nodes=n_nodes, sparse=True).to(DEVICE)

    n2v_loader = n2v_model.loader(batch_size=128, shuffle=True, num_workers=0)
    n2v_opt = torch.optim.SparseAdam(list(n2v_model.parameters()), lr=0.01)

    n2v_model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for pos_rw, neg_rw in n2v_loader:
            n2v_opt.zero_grad()
            loss = n2v_model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
            loss.backward()
            n2v_opt.step()
            total_loss += float(loss)
        log.info(f"Node2Vec epoch {epoch}/5 loss={total_loss / max(len(n2v_loader), 1):.4f}")

    n2v_model.eval()
    with torch.no_grad():
        n2v_emb = n2v_model.embedding.weight.detach().cpu().numpy().astype(np.float32)
    log.info(f"Node2Vec embeddings: {n2v_emb.shape}")

    rng = np.random.default_rng(seed)
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for s, t in zip(src_idx, tgt_idx):
        adj[s].append(t)

    sentences: list[list[str]] = []
    for _ in range(n_walks):
        order = list(range(n_nodes))
        rng.shuffle(order)
        for start in order:
            walk    = [start]
            current = start
            for _ in range(walk_length - 1):
                nbrs = adj[current]
                if not nbrs:
                    break
                current = nbrs[int(rng.integers(0, len(nbrs)))]
                walk.append(current)
            sentences.append([word_list[i] for i in walk])

    log.info(f"Generated {len(sentences)} FastText walks (n_walks={n_walks}, length={walk_length})")
    ft = FastText(sentences=sentences, vector_size=embed_dim, window=5, min_count=1, workers=workers, seed=seed, epochs=10)
    ft_emb = np.array([ft.wv[w] if w in ft.wv else np.zeros(embed_dim, dtype=np.float32) for w in word_list], dtype=np.float32)
    log.info(f"FastText embeddings:  {ft_emb.shape}")

    return n2v_emb, ft_emb, word_list


def train_xlmr_baseline(nodes: pd.DataFrame, embed_dim: int = 256, model_name: str = "xlm-roberta-base", batch_size: int = 64, seed: int = 42, device=None) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts contextualized embeddings for each word.
    """
    device = device or DEVICE
    word_list = nodes["word"].tolist()
    log.info(f"Extracting XLM-R embeddings with '{model_name}' (batch_size={batch_size}, embed_dim={embed_dim})")

    def_col = "definition_hy" if "definition_hy" in nodes.columns else None
    if def_col is not None:
        defs = nodes[def_col].fillna("").astype(str).tolist()
    else:
        defs = [""] * len(word_list)

    def _make_sentence(word: str, definition: str) -> str:
        definition = definition.strip()
        if definition:
            if len(definition) > 120:
                definition = definition[:120].rsplit(" ", 1)[0] + "…"
            return f"Բառը՝ {word}. Սահմանումը՝ {definition}"
        return f"Բառը՝ {word}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    xlmr_model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_vecs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(word_list), batch_size), desc="XLM-R encoding"):
            batch_words = word_list[start: start + batch_size]
            batch_defs = defs[start: start + batch_size]
            framed = [_make_sentence(w, d) for w, d in zip(batch_words, batch_defs)]
            enc = tokenizer(framed, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = xlmr_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_vecs.append(vecs.cpu().float().numpy())

    raw_matrix = np.concatenate(all_vecs, axis=0)  # (n_words, hidden_size)
    log.info(f"Raw XLM-R embeddings shape: {raw_matrix.shape}")

    n_components = min(embed_dim, raw_matrix.shape[0], raw_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(raw_matrix).astype(np.float32)
    log.info(f"XLM-R PCA: kept {n_components} components, explained variance = {pca.explained_variance_ratio_.sum():.3f}")

    if n_components < embed_dim:
        projected = np.pad(projected, ((0, 0), (0, embed_dim - n_components)))

    log.info(f"XLM-R embeddings projected to shape: {projected.shape}")
    return projected, word_list
