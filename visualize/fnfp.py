import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import re
from pathlib import Path


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans', 'Arial Unicode MS']

armenian_font_prop = None
try:
    import matplotlib.font_manager as fm

    for font in fm.findSystemFonts():
        if 'NotoSerifArmenian' in font or 'NotoSansArmenian' in font:
            armenian_font_prop = fm.FontProperties(fname=font)
            print(f"Using Armenian font: {font}")
            break
    if armenian_font_prop is None:
        print("Note: Armenian font not found, using default")
except:
    print("Note: Could not load Armenian font, using default")


def clean_word(word: str) -> str:
    """Keep only Armenian characters."""
    if not isinstance(word, str):
        return ""
    match = re.search(r'[Ա-Ֆա-ֆ]+', word)
    if match:
        return match.group(0)
    return word


def draw_edge_pair(src, tgt, out_path, arrow_color='#2A9D8F', linestyle='solid', relation=None, score=None, shrinkA=40, shrinkB=40):
    """
    Draw two nodes as rounded boxes with a directed arrow.
    """
    src_clean = clean_word(src)
    tgt_clean = clean_word(tgt)
    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    src_pos = (0, 0)
    tgt_pos = (2, 0)

    node_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333', alpha=0.9, linewidth=1.5)

    # Source node
    if armenian_font_prop:
        ax.text(src_pos[0], src_pos[1], src_clean, fontsize=10, ha='center', va='center', fontproperties=armenian_font_prop, bbox=node_bbox)
    else:
        ax.text(src_pos[0], src_pos[1], src_clean, fontsize=10, ha='center', va='center', bbox=node_bbox)

    if armenian_font_prop:
        ax.text(tgt_pos[0], tgt_pos[1], tgt_clean, fontsize=10, ha='center', va='center', fontproperties=armenian_font_prop, bbox=node_bbox)
    else:
        ax.text(tgt_pos[0], tgt_pos[1], tgt_clean, fontsize=10, ha='center', va='center', bbox=node_bbox)

    arrow = FancyArrowPatch(src_pos, tgt_pos, arrowstyle='->', mutation_scale=20, color=arrow_color, linestyle=linestyle, linewidth=2.0, shrinkA=shrinkA, shrinkB=shrinkB)
    ax.add_patch(arrow)

    if relation:
        mid_x = (src_pos[0] + tgt_pos[0]) / 2
        mid_y = (src_pos[1] + tgt_pos[1]) / 2 + 0.12
        ax.text(mid_x, mid_y, relation, fontsize=8, ha='center', color=arrow_color, backgroundcolor='none')

    if score is not None:
        ax.text(src_pos[0], src_pos[1] - 0.35, f"score={score:.2f}", ha='center', fontsize=7, style='italic', color='#555')

    max_len = max(len(src_clean), len(tgt_clean))
    xlim_pad = 0.6 if max_len < 8 else 0.9
    ax.set_xlim(-xlim_pad, 2 + xlim_pad)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# known examples
output_dir = Path("error_analysis_figures")
output_dir.mkdir(exist_ok=True)

fn_examples = [("բազմանուն", "բազմանունություն", "adjectivalization", 0.0121, 40, 50), ("նախաճաշ", "նախաճաշել", "nominalization", 0.2368, 10, 40)]

fp_examples = [("ցեղաբանական", "կապոտվել", None, 1.0000, 20, 30), ("լուսանկարչորեն", "փշալար", None, 1.0000, 20, 25)]


for i, (src, tgt, rel, score, shrinkA, shrinkB) in enumerate(fn_examples, 1):
    draw_edge_pair(src, tgt, output_dir / f"false_negative_{i}.png", arrow_color='#2A9D8F', linestyle='solid', relation=rel, score=score, shrinkA=shrinkA, shrinkB=shrinkB)

for i, (src, tgt, rel, score, shrinkA, shrinkB) in enumerate(fp_examples, 1):
    draw_edge_pair(src, tgt, output_dir / f"false_positive_{i}.png", arrow_color='#E63946', linestyle='dashed', relation=rel, score=score, shrinkA=shrinkA, shrinkB=shrinkB)


print("\nDone. Files saved in:", output_dir)
