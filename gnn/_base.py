from __future__ import annotations
import logging
import torch


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORICAL_NODE_FEATURES = ["pos", "animacy", "declension_class", "verb_transitivity", "aktionsart"]

RELATION_TYPES = [
    # suffixation family
    "nominalization", "adjectivalization", "verbalization", "other_suffixation", "demonym",
    # prefixation family
    "negation_prefix", "intensifying_prefix", "directional_prefix", "locative_prefix", "temporal_prefix",
    # compounding family
    "root_compound", "synthetic_compound", "adjective_compound", "compound_component",
    # derivation
    "causative", "detransitive", "diminutive", "reduplication",
    # inverse suffixation
    "denominalization", "deadjectivalization", "deverbalization", "other_suffixation_inv", "demonym_inv",
    # inverse prefixation
    "negation_prefix_inv", "intensifying_prefix_inv", "directional_prefix_inv", "locative_prefix_inv", "temporal_prefix_inv",
    # inverse compounding
    "root_compound_inv", "synthetic_compound_inv", "adjective_compound_inv", "compound_component_inv",
    # inverse derivation
    "causative_inv", "detransitive_inv", "diminutive_inv", "reduplication_inv"
]

COMPOUND_RELATIONS = {"root_compound", "synthetic_compound", "adjective_compound", "compound_component"}

_ENSEMBLE_MODES = ("gate", "concat", "weighted", "sum", "gnn_only", "aux_only")

_CI_METRICS = ["auc", "ap", "f1", "precision", "recall", "mrr", "hits1", "hits5", "hits10", "ari"]

_ENSEMBLE_CI_METRICS = ["auc", "ap", "f1", "precision", "recall", "mrr", "hits1", "hits5", "hits10"]

_CI_BOOTSTRAP_ITERS = 9999

_CHECKPOINT_FILE = "checkpoint.json"