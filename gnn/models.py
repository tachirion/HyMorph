import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, RGCNConv
from gnn._base import _ENSEMBLE_MODES
from gnn.features import NodeFeatureEncoder, NodeEmbedder


# =========================== ASYMMETRIC DECODER ===========================
class AsymmetricDecoder(nn.Module):
    """
    Learns separate source and target projections so score(A -> B) != score(B -> A).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_src = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_tgt = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = self.W_src(z[edge_index[0]])
        tgt = self.W_tgt(z[edge_index[1]])
        return (src * tgt).sum(dim=-1)


# =========================== BASE GNN CLASSES ===========================
class BaseGNNEncoder(nn.Module):
    """
    Shared encoder for GCN, SAGE and GAT.
    """
    def __init__(self, node_embedder: NodeEmbedder, conv1: nn.Module, conv2: nn.Module, dropout: float = 0.5, activation = F.relu):
        super().__init__()
        self.node_embedder = node_embedder
        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = dropout
        self.activation = activation

    def embed_nodes(self, data, device) -> torch.Tensor:
        return self.node_embedder(data.cat_ids.to(device), data.trigram_ids.to(device), data.offsets.to(device))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x0 = x
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.layer_norm(x, x.shape[1:])
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(torch.cat([x, x0], dim=1), edge_index)
        return x


class BaseLinkPredictor(nn.Module):
    """
    Shared link predictor for GCN, SAGE and GAT.
    """
    def __init__(self, encoder: BaseGNNEncoder, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = AsymmetricDecoder(embed_dim)

    def encode(self, data, edge_index: torch.Tensor, device) -> torch.Tensor:
        x = self.encoder.embed_nodes(data, device)
        z = self.encoder(x, edge_index)
        return F.normalize(z, p=2, dim=-1)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_index)


# =========================== RGCN MODEL ===========================
class RGCNEncoder(BaseGNNEncoder):
    """
    RGCN forward needs edge_type passed to both conv layers.
    """
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x0 = x
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.layer_norm(x, x.shape[1:])
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(torch.cat([x, x0], dim=1), edge_index, edge_type)
        return x


class RGCNLinkPredictor(nn.Module):
    def __init__(self, encoder: RGCNEncoder, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = AsymmetricDecoder(embed_dim)

    def encode(self, data, edge_index: torch.Tensor, device) -> torch.Tensor:
        x = self.encoder.embed_nodes(data, device)
        edge_type = data.mp_edge_type.to(device)
        z = self.encoder(x, edge_index, edge_type)
        return F.normalize(z, p=2, dim=-1)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_index)


# =========================== ENSEMBLE ===========================
class EnsembleVariant(nn.Module):
    """
    Unified ensemble head covering all ablation modes.
    """
    def __init__(self, gnn_dim: int, aux_dim: int, out_dim: int, mode: str = "gate"):
        super().__init__()
        assert mode in _ENSEMBLE_MODES
        self.mode = mode
        self.decoder = AsymmetricDecoder(out_dim)

        if mode == "gate":
            self.gate = nn.Sequential(nn.Linear(gnn_dim + aux_dim, out_dim), nn.Sigmoid())
            self.proj_gnn = nn.Linear(gnn_dim, out_dim, bias=False)
            self.proj_aux = nn.Linear(aux_dim, out_dim, bias=False)
        elif mode == "concat":
            self.proj = nn.Linear(gnn_dim + aux_dim, out_dim)
        elif mode == "weighted":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.proj_gnn = nn.Linear(gnn_dim, out_dim, bias=False)
            self.proj_aux = nn.Linear(aux_dim, out_dim, bias=False)
        elif mode == "sum":
            self.proj_gnn = nn.Linear(gnn_dim, out_dim, bias=False)
            self.proj_aux = nn.Linear(aux_dim, out_dim, bias=False)
        elif mode == "gnn_only":
            self.proj_gnn = nn.Linear(gnn_dim, out_dim, bias=False)
        elif mode == "aux_only":
            self.proj_aux = nn.Linear(aux_dim, out_dim, bias=False)

    def forward(self, gnn_emb: torch.Tensor, aux_emb: torch.Tensor) -> torch.Tensor:
        if self.mode == "gate":
            g = self.gate(torch.cat([gnn_emb, aux_emb], dim=1))
            return g * self.proj_gnn(gnn_emb) + (1 - g) * self.proj_aux(aux_emb)
        if self.mode == "concat":
            return self.proj(torch.cat([gnn_emb, aux_emb], dim=1))
        if self.mode == "weighted":
            a = torch.sigmoid(self.alpha)
            return a * self.proj_gnn(gnn_emb) + (1 - a) * self.proj_aux(aux_emb)
        if self.mode == "sum":
            return self.proj_gnn(gnn_emb) + self.proj_aux(aux_emb)
        if self.mode == "gnn_only":
            return self.proj_gnn(gnn_emb)
        return self.proj_aux(aux_emb)

    def score(self, gnn_emb: torch.Tensor, aux_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.forward(gnn_emb, aux_emb)
        return self.decoder(z, edge_index)


class EdgeTypeHead(nn.Module):
    """
    Auxiliary multi-class head: (z_src ‖ z_tgt ‖ z_src |.| z_tgt) -> relation type logits.
    Trained jointly with the link predictor; set aux_weight=0 to disable.
    """
    def __init__(self, embed_dim: int, n_relation_types: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_relation_types))

    def forward(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_src, z_tgt, z_src * z_tgt], dim=1))


# =========================== POS CLASSIFIER ===========================
class POSClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, n_pos_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_pos_classes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =========================== HELPERS ===========================
def build_gcn(feature_encoder: NodeFeatureEncoder, feat_dim: int, hidden_dim: int, embed_dim: int, dropout: float = 0.5) -> BaseLinkPredictor:
    embedder = NodeEmbedder(feature_encoder, feat_dim)
    conv1 = GCNConv(feat_dim, hidden_dim)
    conv2 = GCNConv(hidden_dim + feat_dim, embed_dim)
    encoder = BaseGNNEncoder(embedder, conv1, conv2, dropout=dropout, activation=F.relu)
    return BaseLinkPredictor(encoder, embed_dim)


def build_sage(feature_encoder: NodeFeatureEncoder, feat_dim: int, hidden_dim: int, embed_dim: int, dropout: float = 0.5, aggr: str = "mean") -> BaseLinkPredictor:
    embedder = NodeEmbedder(feature_encoder, feat_dim)
    conv1 = SAGEConv(feat_dim, hidden_dim, aggr=aggr)
    conv2 = SAGEConv(hidden_dim + feat_dim, embed_dim, aggr=aggr)
    encoder = BaseGNNEncoder(embedder, conv1, conv2, dropout=dropout, activation=F.relu)
    return BaseLinkPredictor(encoder, embed_dim)


def build_gat(feature_encoder: NodeFeatureEncoder, feat_dim: int, hidden_dim: int, embed_dim: int, heads: int = 4, dropout: float = 0.5) -> BaseLinkPredictor:
    embedder = NodeEmbedder(feature_encoder, feat_dim)
    conv1 = GATv2Conv(feat_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
    conv2 = GATv2Conv(hidden_dim + feat_dim, embed_dim, heads=1, dropout=dropout, concat=False)
    encoder = BaseGNNEncoder(embedder, conv1, conv2, dropout=dropout, activation=F.relu)
    return BaseLinkPredictor(encoder, embed_dim)


def build_rgcn(feature_encoder: NodeFeatureEncoder, feat_dim: int, hidden_dim: int, embed_dim: int, num_relations: int, num_bases: int = 16, dropout: float = 0.5) -> RGCNLinkPredictor:
    embedder = NodeEmbedder(feature_encoder, feat_dim)
    num_bases = max(4, min(num_bases, num_relations))
    conv1 = RGCNConv(feat_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
    conv2 = RGCNConv(hidden_dim + feat_dim, embed_dim, num_relations=num_relations, num_bases=num_bases)
    encoder = RGCNEncoder(embedder, conv1, conv2, dropout=dropout)
    return RGCNLinkPredictor(encoder, embed_dim)
