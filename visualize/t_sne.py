import argparse
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_single_model_2d(nodes_csv, model_configs, sample_size=2000, random_state=42, output_prefix="pos_clustering"):
    df_nodes = pd.read_csv(nodes_csv)
    if 'word' not in df_nodes.columns or 'pos' not in df_nodes.columns:
        raise ValueError("nodes.csv must contain 'word' and 'pos' columns.")
    word_to_pos = dict(zip(df_nodes['word'], df_nodes['pos']))

    all_pos_tags = sorted(df_nodes['pos'].unique())
    cmap = plt.colormaps.get_cmap('tab10')
    pos_color_map = {pos: cmap(i / len(all_pos_tags)) for i, pos in enumerate(all_pos_tags)}

    all_words = None
    embeddings_dict = {}

    for cfg in model_configs:
        name = cfg['name']
        df_embed = pd.read_csv(cfg['embed_csv'])

        meta_cols = ['word']
        embed_cols = [c for c in df_embed.columns if c not in meta_cols and np.issubdtype(df_embed[c].dtype, np.number)]
        if not embed_cols:
            raise ValueError(f"No numeric embedding columns found in {cfg['embed_csv']}")

        words_with_pos = [w for w in df_embed['word'] if w in word_to_pos]
        df_filtered = df_embed[df_embed['word'].isin(words_with_pos)].copy()
        if df_filtered.empty:
            raise ValueError(f"No words in {cfg['embed_csv']} have POS labels in {nodes_csv}")

        word_to_vec = dict(zip(df_filtered['word'], df_filtered[embed_cols].values))
        embeddings_dict[name] = word_to_vec

        current_words = set(word_to_vec.keys())
        if all_words is None:
            all_words = current_words
        else:
            all_words = all_words.intersection(current_words)

    if not all_words:
        raise ValueError("No common words with POS labels found across all models.")

    rng = np.random.RandomState(random_state)
    sampled_words = rng.choice(list(all_words), size=min(sample_size, len(all_words)), replace=False)

    for cfg in model_configs:
        name = cfg['name']
        word_to_vec = embeddings_dict[name]

        X = []
        y_pos = []
        for w in sampled_words:
            if w in word_to_vec and w in word_to_pos:
                X.append(word_to_vec[w])
                y_pos.append(word_to_pos[w])
        X = np.array(X)
        y_pos = np.array(y_pos)

        if len(X) == 0:
            print(f"Warning: No sampled words found in {name} embeddings. Skipping.")
            continue

        print(f"t‑SNE: {name} ({X.shape[0]} points)")
        tsne = TSNE(n_components=2, perplexity=30, random_state=random_state, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X)
        _plot_2d(X_tsne, y_pos, name, "tsne", output_prefix, pos_color_map)


def _plot_2d(X_2d, y_pos, model_name, reducer, output_prefix, pos_color_map):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', s=10, alpha=0.3, rasterized=True)

    unique_pos = np.unique(y_pos)
    for pos in unique_pos:
        mask = y_pos == pos
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=50, alpha=0.9, edgecolors='black', linewidth=0.4, color=pos_color_map[pos], label=pos.capitalize())

    plt.axis('off')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Part-of-Speech", markerscale=1.2)
    plt.tight_layout()

    output_file = f"{output_prefix}_{model_name}_{reducer}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot t-SNE projections of word embeddings colored by POS")
    parser.add_argument("--nodes", required=True, help="Path to nodes.csv with 'word' and 'pos' columns")
    parser.add_argument("--model", action="append", required=True, help="Model configuration in format name:embedding_csv (can be used multiple times)")
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of words to sample for t-SNE")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-prefix", default="pos_clustering", help="Prefix for output PNG files")

    args = parser.parse_args()

    model_configs = []
    for model_arg in args.model:
        if ':' not in model_arg:
            raise ValueError(f"Invalid model format: {model_arg}. Expected name:path")
        name, path = model_arg.split(':', 1)
        model_configs.append({"name": name, "embed_csv": path})

    plot_single_model_2d(nodes_csv=args.nodes, model_configs=model_configs, sample_size=args.sample_size, random_state=args.random_state, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
    