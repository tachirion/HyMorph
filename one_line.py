import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent  # HyMorph/


def _find_latest_run() -> Path:
    runs = sorted([p for p in (ROOT / "runs").glob("scrape_run_*") if p.is_dir()], key=lambda p: p.name, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No scrape run found under {ROOT / 'runs'}")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="HyMorph", formatter_class=argparse.RawDescriptionHelpFormatter)
    s = parser.add_argument_group("scraper")
    s.add_argument("--seed-file", type=str, default=None, help="Path to .dic seed file (default: data/armenian_eastern.dic)")
    s.add_argument("--resume", action="store_true", help="Resume last scrape run")
    s.add_argument("--delay", type=float, default=0.5, help="Request delay in seconds")
    s.add_argument("--max-queue", type=int, default=100_000, help="Max words in scrape queue")
    s.add_argument("--neg-sample", action="store_true", help="Generate negative samples after scraping")
    s.add_argument("--neg-ratio", type=int, default=1, help="Negatives per positive edge")
    s.add_argument("--neg-strategy", choices=["random", "typed", "relcorr"], default="typed")
    s.add_argument("--gnn-split", action="store_true", help="Split negatives into train/val/test CSVs")

    g = parser.add_argument_group("gnn")
    g.add_argument("--seeds", type=int, nargs="+", default=[42], metavar="SEED")
    g.add_argument("--ci-level", type=float, default=0.95)
    g.add_argument("--ci-method", choices=["bootstrap", "t"], default="bootstrap")
    g.add_argument("--skip-baselines", action="store_true")
    g.add_argument("--skip-topology", action="store_true")
    g.add_argument("--load-baselines", type=str, default=None, help="Path to existing results_baselines.json")
    g.add_argument("--load-topology", type=str, default=None, help="Path to existing topology.json")
    g.add_argument("--model-only", type=str, default=None, choices=["GCN", "GraphSAGE", "GAT", "RGCN"])

    args = parser.parse_args()

    # SCRAPER
    scraper_argv = []
    if args.seed_file:
        scraper_argv += ["--seed-file", args.seed_file]
    if args.resume:
        scraper_argv += ["--resume"]
    if args.delay != 0.5:
        scraper_argv += ["--delay", str(args.delay)]
    if args.max_queue != 100_000:
        scraper_argv += ["--max-queue", str(args.max_queue)]
    if args.neg_sample:
        scraper_argv += [
            "--neg-sample",
            "--neg-ratio", str(args.neg_ratio),
            "--neg-strategy", args.neg_strategy
        ]
    if args.gnn_split:
        scraper_argv += ["--gnn-split"]

    sys.argv = ["scraper.pipeline"] + scraper_argv
    from scraper.pipeline import main as scraper_main
    scraper_main()

    # WIRE PATHS
    run_dir = _find_latest_run()  # HyMorph/runs/scrape_run_YYYYMMDD_HHMM/
    nodes_path = run_dir / "output" / "nodes.csv"
    edges_path = run_dir / "output" / "edges.csv"
    neg_dir = run_dir / "output"

    # GNN
    gnn_argv = [
        "--nodes", str(nodes_path),  # HyMorph/runs/.../output/nodes.csv
        "--edges", str(edges_path),  # HyMorph/runs/.../output/edges.csv
        "--config", str(ROOT / "config.yaml"),  # HyMorph/config.yaml
        "--out-dir", str(ROOT / "results"),  # HyMorph/results/
        "--seeds", *[str(s) for s in args.seeds],
        "--ci-level", str(args.ci_level),
        "--ci-method", args.ci_method
    ]
    if args.skip_baselines:
        gnn_argv += ["--skip-baselines"]
    if args.skip_topology:
        gnn_argv += ["--skip-topology"]
    if args.load_baselines:
        gnn_argv += ["--load-baselines", args.load_baselines]
    if args.load_topology:
        gnn_argv += ["--load-topology", args.load_topology]
    if args.model_only:
        gnn_argv += ["--model-only", args.model_only]
    if (neg_dir / "negative_samples_train.csv").exists():
        gnn_argv += ["--neg-dir", str(neg_dir)]

    sys.argv = ["gnn.pipeline"] + gnn_argv
    from gnn.pipeline import main as gnn_main
    gnn_main()


if __name__ == "__main__":
    main()