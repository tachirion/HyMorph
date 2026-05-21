import argparse
import shutil
from pathlib import Path
import pandas as pd


# ===================== RENAME MAP =====================
RENAME = {
    # suffixation family — old subtype_suffixation pattern
    "denominal_suffixation": "nominalization",
    "deadjectival_suffixation": "adjectivalization",
    "deverbal_suffixation": "verbalization",
    "demonym_suffixation": "demonym",
    "rare_suffixation": "other_suffixation",
    "suffixation": "other_suffixation",
    # old derives_noun (noun->verb direction) = nominalization
    "derives_noun": "nominalization",
    # old demonym_suffix
    "demonym_suffix": "demonym",
    "locational_prefix": "locative_prefix",
    "locational_prefix_base": "locative_prefix_inv",
    "wiki_compound_member": "compound_component",
    "compound_from_wiki": "compound_component",
    "etymological_component": "compound_component",
    # old inverse suffixation labels
    "denominalization": "denominalization",
    "deadjectivalization": "deadjectivalization",
    "deverbalization": "deverbalization",
    "back_formation": "other_suffixation_inv",
    "anticausative_base": "causative_inv",
    "causative_base": "detransitive_inv",
    "denominal_base": "denominalization",
    "deverbal_base": "deverbalization",
    "deadjectival_base": "deadjectivalization",
    "demonym_base": "demonym_inv",
    # old _base suffix on prefix inverses
    "negation_prefix_base": "negation_prefix_inv",
    "intensifying_prefix_base": "intensifying_prefix_inv",
    "directional_prefix_base": "directional_prefix_inv",
    "locative_prefix_base": "locative_prefix_inv",
    "temporal_prefix_base": "temporal_prefix_inv",
    # old compound_component_inv if it existed
    "compound_component_inv": "compound_component_inv"
}

# ===================== RELATION HIERARCHY =====================
RELATION_HIERARCHY = {
    "nominalization": "suffixation",
    "adjectivalization": "suffixation",
    "verbalization": "suffixation",
    "other_suffixation": "suffixation",
    "demonym": "suffixation",
    "negation_prefix": "prefixation",
    "intensifying_prefix": "prefixation",
    "directional_prefix": "prefixation",
    "locative_prefix": "prefixation",
    "temporal_prefix": "prefixation",
    "root_compound": "compounding",
    "synthetic_compound": "compounding",
    "adjective_compound": "compounding",
    "compound_component": "compounding",
    "causative": "derivation",
    "detransitive": "derivation",
    "diminutive": "derivation",
    "reduplication": "derivation",
    "denominalization": "inverse_suffixation",
    "deadjectivalization": "inverse_suffixation",
    "deverbalization": "inverse_suffixation",
    "other_suffixation_inv": "inverse_suffixation",
    "demonym_inv": "inverse_suffixation",
    "negation_prefix_inv": "inverse_prefixation",
    "intensifying_prefix_inv": "inverse_prefixation",
    "directional_prefix_inv": "inverse_prefixation",
    "locative_prefix_inv": "inverse_prefixation",
    "temporal_prefix_inv": "inverse_prefixation",
    "root_compound_inv": "inverse_compounding",
    "synthetic_compound_inv": "inverse_compounding",
    "adjective_compound_inv": "inverse_compounding",
    "compound_component_inv": "inverse_compounding",
    "causative_inv": "inverse_derivation",
    "detransitive_inv": "inverse_derivation",
    "diminutive_inv": "inverse_derivation",
    "reduplication_inv": "inverse_derivation"
}

KNOWN_RELATIONS = set(RELATION_HIERARCHY.keys())


def reclassify_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["relation"] = df["relation"].map(lambda r: RENAME.get(r, r))
    if "relation_class" in df.columns:
        df["relation_class"] = df["relation"].map(lambda r: RELATION_HIERARCHY.get(r, "other"))
    unknown = sorted(r for r in df["relation"].unique() if r not in KNOWN_RELATIONS)
    return df, unknown


def reclassify_edges(edges_path: Path, nodes_path: Path, negatives_path: Path = None) -> None:
    edges_df = pd.read_csv(edges_path, dtype=str).fillna("")
    nodes_df = pd.read_csv(nodes_path, dtype=str).fillna("")

    shutil.copy(edges_path, edges_path.with_suffix(".bak"))
    shutil.copy(nodes_path, nodes_path.with_suffix(".bak"))
    print(f"Backed up to {edges_path.with_suffix('.bak')} and {nodes_path.with_suffix('.bak')}")

    before_counts = edges_df["relation"].value_counts().to_dict()
    before_len = len(edges_df)
    edges_df, unknown = reclassify_df(edges_df, "edges")
    edges_df = edges_df.drop_duplicates(subset=["source", "relation", "target"])
    dropped_dups = before_len - len(edges_df)
    after_counts = edges_df["relation"].value_counts().to_dict()

    print(f"\nEdge relation changes:")
    for old, new in sorted(RENAME.items()):
        if old == new:
            continue
        n = before_counts.get(old, 0)
        if n:
            print(f"{old!r} -> {new!r} ({n} edges)")
    if dropped_dups:
        print(f"Dropped {dropped_dups} duplicate edges after rename.")
    if unknown:
        print(f"\nUnrecognized relation types in edges (not in RENAME or RELATION_HIERARCHY):")
        for r in unknown:
            print(f"{r!r} ({after_counts.get(r, 0)} edges)")
    else:
        print("All relation types recognized.")

    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(edges_df)} edges to {edges_path}")

    if "relation_class" in nodes_df.columns:
        nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(nodes_df)} nodes to {nodes_path}")
    else:
        print("nodes.csv has no relation_class column — left untouched.")

    if negatives_path is not None:
        shutil.copy(negatives_path, negatives_path.with_suffix(".bak"))
        print(f"\nBacked up negatives to {negatives_path.with_suffix('.bak')}")
        neg_df = pd.read_csv(negatives_path, dtype=str).fillna("")
        neg_before = neg_df["relation"].value_counts().to_dict()
        neg_df, neg_unknown = reclassify_df(neg_df, "negatives")
        print(f"\nNegatives relation changes:")
        for old, new in sorted(RENAME.items()):
            if old == new:
                continue
            n = neg_before.get(old, 0)
            if n:
                print(f"{old!r} -> {new!r} ({n} rows)")
        if neg_unknown:
            print(f"Unrecognized relation types in negatives:")
            for r in neg_unknown:
                print(f"{r!r}")
        else:
            print("All relation types recognized.")
        neg_df.to_csv(negatives_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(neg_df)} rows to {negatives_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify edge types in-place.")
    parser.add_argument("--edges", type=Path, default=Path("edges.csv"))
    parser.add_argument("--nodes", type=Path, default=Path("nodes.csv"))
    parser.add_argument("--negatives", type=Path, default=None, help="Optional negatives or combined CSV to reclassify in-place.")
    args = parser.parse_args()
    if not args.edges.exists():
        raise FileNotFoundError(args.edges)
    if not args.nodes.exists():
        raise FileNotFoundError(args.nodes)
    if args.negatives is not None and not args.negatives.exists():
        raise FileNotFoundError(args.negatives)
    reclassify_edges(args.edges, args.nodes, args.negatives)


if __name__ == "__main__":
    main()
