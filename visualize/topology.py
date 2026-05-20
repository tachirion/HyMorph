import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans', 'Arial Unicode MS']

try:
    import matplotlib.font_manager as fm

    armenian_font_path = None
    for font in fm.findSystemFonts():
        if 'NotoSerifArmenian' in font or 'NotoSansArmenian' in font:
            armenian_font_path = font
            break

    if armenian_font_path:
        armenian_font_prop = fm.FontProperties(fname=armenian_font_path)
        print(f"Using Armenian font: {armenian_font_path}")
    else:
        armenian_font_prop = None
        print("Note: Armenian font not found, using default")
except:
    armenian_font_prop = None
    print("Note: Could not load Armenian font, using default")


def clean_word(word: str) -> str:
    if not isinstance(word, str):
        return ""

    match = re.search(r'[Ա-Ֆա-ֆ]+', word)
    if match:
        cleaned = match.group(0)
    else:
        cleaned = re.sub(r'^[\W\d_]+', '', word)
        cleaned = re.sub(r'[\W\d_]+$', '', cleaned)

    cleaned = re.sub(r'^[\*†‡\[\]\(\)\.\s]+', '', cleaned)

    return cleaned if cleaned else word


def load_full_graph(nodes_path: Path, edges_path: Path):
    nodes = pd.read_csv(nodes_path, encoding="utf-8-sig").fillna("")
    edges = pd.read_csv(edges_path, encoding="utf-8-sig").fillna("")

    nodes["word"] = nodes["word"].apply(clean_word)
    edges["source"] = edges["source"].apply(clean_word)
    edges["target"] = edges["target"].apply(clean_word)

    nodes = nodes[nodes["word"].str.len() > 0]
    edges = edges[edges["source"].str.len() > 0]
    edges = edges[edges["target"].str.len() > 0]

    nodes = nodes.sort_values(by="word").drop_duplicates(subset=["word"], keep="first").reset_index(drop=True)
    edges = edges.drop_duplicates(subset=["source", "relation", "target"]).reset_index(drop=True)

    word_set = set(nodes["word"])
    edges = edges[edges["source"].isin(word_set) | edges["target"].isin(word_set)]

    nom_pairs = set(zip(edges[edges["relation"] == "nominalization"]["source"], edges[edges["relation"] == "nominalization"]["target"]))
    mask = ~((edges["relation"] == "derives_noun") & edges.apply(lambda r: (r["source"], r["target"]) in nom_pairs, axis=1))
    edges = edges[mask].reset_index(drop=True)

    all_edge_words = set(edges["source"]) | set(edges["target"])
    missing = all_edge_words - set(nodes["word"])
    if missing:
        placeholders = pd.DataFrame({
            "word": sorted(missing), "pos": "unknown",
            "definition_hy": "", "animacy": "", "declension_class": "",
            "verb_transitivity": "", "aktionsart": "", "scraped_at": ""
        })
        nodes = pd.concat([nodes, placeholders], ignore_index=True)

    connected = set(edges["source"]) | set(edges["target"])
    nodes = nodes[nodes["word"].isin(connected)].reset_index(drop=True)

    G = nx.DiGraph()
    G.add_nodes_from(nodes["word"].tolist())
    G.add_edges_from(zip(edges["source"].tolist(), edges["target"].tolist()))

    print(f"Loaded full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    return G, nodes, edges


def find_top_roots_by_outdegree(G: nx.DiGraph, top_k: int = 7) -> list:
    out_degrees = [(node, G.out_degree(node)) for node in G.nodes()]
    out_degrees.sort(key=lambda x: x[1], reverse=True)
    top_roots = out_degrees[:top_k]

    print("\nTop roots by out-degree:")
    for root, deg in top_roots:
        print(f"  {clean_word(root)}: {deg}")

    return top_roots


def get_family_members(G: nx.DiGraph, root: str, max_members: int = 12) -> list:
    distances = {}
    for node in G.nodes():
        if node != root:
            try:
                dist = nx.shortest_path_length(G, root, node)
                distances[node] = dist
            except:
                pass

    sorted_members = sorted(distances.items(), key=lambda x: x[1])
    members = [m for m, d in sorted_members[:max_members]]

    return members


def build_family_subgraph(G: nx.DiGraph, root: str, members: list) -> nx.DiGraph:
    family_set = set(members) | {root}
    subG = G.subgraph(family_set).copy()
    return subG


def compact_radial_layout(G: nx.DiGraph, root: str, radius_increment: float = 0.6):
    distances = {}
    for node in G.nodes():
        if node == root:
            distances[node] = 0
        else:
            try:
                distances[node] = nx.shortest_path_length(G, root, node)
            except:
                distances[node] = 999

    nodes_by_dist = {}
    max_dist = 0
    for node, dist in distances.items():
        if dist < 999:
            if dist not in nodes_by_dist:
                nodes_by_dist[dist] = []
            nodes_by_dist[dist].append(node)
            max_dist = max(max_dist, dist)

    pos = {}
    pos[root] = (0, 0)

    for dist in range(1, max_dist + 1):
        if dist not in nodes_by_dist:
            continue

        nodes = nodes_by_dist[dist]
        num_nodes = len(nodes)
        radius = dist * radius_increment

        for i, node in enumerate(nodes):
            angle = (2 * np.pi * i / num_nodes) - np.pi / 2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[node] = (x, y)

    return pos


def draw_family_on_axis(ax, G: nx.DiGraph, root: str, color: str, node_circle_radius=0.35, padding=0.4, max_members=8, root_fontsize=14, child_fontsize=11):
    members = get_family_members(G, root, max_members=max_members)
    subG = build_family_subgraph(G, root, members)

    if len(subG.nodes()) <= 1:
        ax.text(0, 0, clean_word(root), ha='center', va='center', fontsize=root_fontsize, color='white', bbox=dict(boxstyle='circle', facecolor=color, edgecolor='#333', linewidth=2))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        return

    pos = compact_radial_layout(subG, root, radius_increment=0.7)

    for u, v in subG.edges():
        if u in pos and v in pos:
            rad = 0.1 if u == root else 0.3
            ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle='->', color='#888', lw=1.2, alpha=0.7, connectionstyle=f'arc3,rad={rad}'))

    root_x, root_y = pos[root]
    root_circle = plt.Circle((root_x, root_y), node_circle_radius, color=color, ec='#333', lw=2, zorder=3)
    ax.add_patch(root_circle)

    clean_root = clean_word(root)
    if armenian_font_prop:
        ax.text(root_x, root_y, clean_root, fontsize=root_fontsize, ha='center', va='center', fontproperties=armenian_font_prop, color='white', zorder=4)
    else:
        ax.text(root_x, root_y, clean_root, fontsize=root_fontsize, ha='center', va='center', color='white', zorder=4)

    for node in subG.nodes():
        if node == root:
            continue
        x, y = pos[node]
        clean_child = clean_word(node)
        bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCC', alpha=0.9)
        if armenian_font_prop:
            ax.text(x, y, clean_child, fontsize=child_fontsize, ha='center', va='center', fontproperties=armenian_font_prop, color='#333', bbox=bbox_props)
        else:
            ax.text(x, y, clean_child, fontsize=child_fontsize, ha='center', va='center', color='#333', bbox=bbox_props)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_aspect('equal')
    ax.axis('off')


def visualize_single_root(G: nx.DiGraph, root: str, out_deg: int, rank: int, output_dir: Path, color: str):
    print(f"\nGenerating plot {rank}: {root} (out-degree: {out_deg})...")

    members = get_family_members(G, root, max_members=12)
    subG = build_family_subgraph(G, root, members)

    num_nodes = len(subG.nodes())
    fig_size = max(8, min(14, num_nodes * 0.7))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if len(subG.nodes()) <= 1:
        ax.text(0.5, 0.5, f"{clean_word(root)}\n(out-degree: {out_deg})\nNo family members", ha='center', va='center', fontsize=12)
        ax.set_title(f"Root: {clean_word(root)}", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        output_path = output_dir / f"{rank:02d}_{clean_word(root)}_family.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return

    pos = compact_radial_layout(subG, root, radius_increment=0.6)

    for u, v in subG.edges():
        if u in pos and v in pos:
            if u == root:
                rad = 0.1
            else:
                rad = 0.3
            ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle='->', color='#888', lw=1.0, alpha=0.7, connectionstyle=f'arc3,rad={rad}'))

    root_x, root_y = pos[root]
    parent_circle = plt.Circle((root_x, root_y), 0.28, color=color, ec='#333', lw=2, zorder=3)
    ax.add_patch(parent_circle)

    clean_root = clean_word(root)
    if armenian_font_prop:
        ax.text(root_x, root_y, clean_root, fontsize=10, ha='center', va='center', fontproperties=armenian_font_prop, color='white', fontweight='normal', zorder=4)
    else:
        ax.text(root_x, root_y, clean_root, fontsize=10, ha='center', va='center', color='white', fontweight='normal', zorder=4)

    for node in subG.nodes():
        if node == root:
            continue
        x, y = pos[node]
        clean_child = clean_word(node)
        if armenian_font_prop:
            ax.text(x, y, clean_child, fontsize=9, ha='center', va='center', fontproperties=armenian_font_prop, color='#333', fontweight='normal', zorder=3, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCC', alpha=0.9))
        else:
            ax.text(x, y, clean_child, fontsize=9, ha='center', va='center', color='#333', fontweight='normal', zorder=3, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCC', alpha=0.9))

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    padding = 0.8
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)

    ax.set_title(f"{rank}. {clean_root}\nOut-degree: {out_deg} | Family: {len(subG.nodes()) - 1} members", fontsize=12, fontweight='normal', pad=15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()

    safe_root_name = re.sub(r"[^\w\-_]", '_', clean_root)
    output_path = output_dir / f"{rank:02d}_{safe_root_name}_family.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved to {output_path}")
    clean_members = [clean_word(m) for m in members[:6]]
    members_str = ", ".join(clean_members)
    if len(members) > 6:
        members_str += f" (+{len(members) - 6} more)"
    print(f"Members: {members_str}")


def visualize_combined_top3(G: nx.DiGraph, top_roots: list, output_dir: Path, colors: list, fmt='png'):
    print("\nGenerating combined image for top 3 roots (tight vertical stack for one column)...")

    fig_width = 3.5
    fig_height = 10.0

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(3, 1, hspace=0.01)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    root1, deg1 = top_roots[0]
    root2, deg2 = top_roots[1]
    root3, deg3 = top_roots[2]

    draw_family_on_axis(ax1, G, root1, colors[0], node_circle_radius=0.25, padding=0.15, max_members=8, root_fontsize=9, child_fontsize=7)
    draw_family_on_axis(ax2, G, root2, colors[1], node_circle_radius=0.25, padding=0.15, max_members=8, root_fontsize=9, child_fontsize=7)
    draw_family_on_axis(ax3, G, root3, colors[2], node_circle_radius=0.25, padding=0.15, max_members=8, root_fontsize=9, child_fontsize=7)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.tight_layout(pad=0.05, h_pad=0.05)

    ext = fmt.lower()
    output_path = output_dir / f"combined_top3_families.{ext}"
    if ext == 'pdf':
        plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    else:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.close()
    print(f"Combined image saved to {output_path}")


def visualize_all_roots(G: nx.DiGraph, top_roots: list, output_dir: Path, fmt='png'):
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ['#E63946', '#457B9D', '#F4A261', '#2A9D8F', '#E9C46A', '#6A4C93', '#80B918']

    for idx, (root, out_deg) in enumerate(top_roots, 1):
        visualize_single_root(G, root, out_deg, idx, output_dir, colors[idx - 1])

    print(f"\nAll {len(top_roots)} individual visualizations saved to: {output_dir}")

    if len(top_roots) >= 3:
        visualize_combined_top3(G, top_roots[:3], output_dir, colors[:3], fmt=fmt)


def main():
    parser = argparse.ArgumentParser(description="Visualize root families from FULL GNN graph")
    parser.add_argument("--nodes", required=True, help="Path to nodes.csv")
    parser.add_argument("--edges", required=True, help="Path to edges.csv")
    parser.add_argument("--out-dir", default="root_family_plots", help="Output directory")
    parser.add_argument("--top-k", type=int, default=7, help="Number of top roots")
    parser.add_argument("--format", default="png", choices=["png", "pdf"], help="Output format for combined image (PNG or PDF)")

    args = parser.parse_args()

    G, nodes, edges = load_full_graph(Path(args.nodes), Path(args.edges))
    top_roots = find_top_roots_by_outdegree(G, top_k=args.top_k)

    visualize_all_roots(G, top_roots, Path(args.out_dir), fmt=args.format)


if __name__ == "__main__":
    main()
