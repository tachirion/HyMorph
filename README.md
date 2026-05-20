# HyMorph — Armenian Morphological Derivation Graph & GNN Pipeline

HyMorph builds a derivational morphology graph for Modern Eastern Armenian by scraping Armenian Wiktionary, then trains Graph Neural Network models to learn word embeddings that capture morphological relationships.

---

## Overview

The pipeline has two main stages:

**1. Scraping** (`scraper/`)
Crawls Armenian Wiktionary (`hy.wiktionary.org`) to extract derivational relationships — suffixation, prefixation, compounding, causativization, etc. —  and builds a morphological graph saved as `nodes.csv` and `edges.csv` under a timestamped run directory.

**2. GNN Training** (`gnn/`)
Loads the graph, trains GCN / GraphSAGE / GAT / R-GCN models for link prediction, evaluates baselines (Node2Vec, FastText, XLM-R) and a gated ensemble, then saves embeddings, metrics, and artifacts per seed.

---

## Project Structure

```
HyMorph/
├── data/
│   ├── armenian_eastern.dic
│   └── ...
│
├── scraper/
│   ├── __init__.py
│   ├── constants.py
│   ├── morphology.py
│   ├── pipeline.py
│   └── sampling.py
│
├── gnn/
│   ├── __init__.py
│   ├── features.py 
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── runs/
│   └── scrape_run_YYYYMMDD_HHMM/
│       ├── output/
│       │   ├── nodes.csv
│       │   ├── edges.csv
│       │   └── negative_samples.csv
│       └── logs/
│           ├── scrape_log.jsonl
│           └── checkpoint.txt
│
├── results/
│   ├── shared/
│   │   ├── topology.json
│   │   ├── embeddings_node2vec.csv / .npy
│   │   ├── embeddings_fasttext.csv / .npy
│   │   ├── embeddings_xlmr.csv / .npy
│   │   └── results_baselines.json
│   ├── seed_42/
│   │   ├── artifacts/
│   │   │   ├── splits.pt
│   │   │   ├── model_states.pt
│   │   │   ├── data_hash.txt
│   │   │   ├── config_used.yaml
│   │   │   └── config_merged.yaml
│   │   ├── embeddings_gcn.csv / .npy
│   │   ├── embeddings_sage.csv / .npy
│   │   ├── embeddings_gat.csv / .npy
│   │   ├── embeddings_rgcn.csv / .npy
│   │   ├── results_gcn.json
│   │   ├── results_sage.json
│   │   ├── results_gat.json
│   │   ├── results_rgcn.json
│   │   ├── results_ensemble.json
│   │   ├── ensemble_best.pt
│   │   ├── summary.csv
│   │   ├── all_results.json
│   │   └── runtime.json
│   └── seed_.../
│
├── config.yaml
├── one_line.py
└── requirements.txt
```


---

## Requirements

- Python **3.12**
- CPU
- ~4 GB disk space for a full scrape + results

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-root>

# create
python3.12 -m venv .venv

# activate — Linux / macOS
source .venv/bin/activate

# activate — Windows
.venv\Scripts\activate
```

### 2. Create and activate a virtual environment

```bash
# create
python3.12 -m venv .venv

# activate — Linux / macOS
source .venv/bin/activate

# activate — Windows
.venv\Scripts\activate
```

### 3. Install PyTorch

CPU-Only for compatibility.
```bash
# CPU only
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install PyTorch Geometric

```bash
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```