import csv
import json
import time
import argparse
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Dict, List
import requests
import pandas as pd
from scraper._base import *
from scraper.morphology import extract_compound_section_words, extract_etymology_section_words, _edge, ArmenianWordAnalyzer, _inverse_edge, is_proper_noun, seed_pos_map, infer_animacy, infer_aktionsart, infer_transitivity, infer_declension, get_primary_pos, extract_first_definition, resolve_pos
from scraper.sampling import NegativeSampler, NegativeStrategy, split_for_gnn
from tqdm import tqdm


EDGES_FILE, NODES_FILE, LOG_FILE, CHECKPOINT_FILE = None, None, None, None


# =========================== MEDIAWIKI API ===========================
def api_get(params: dict, session: requests.Session) -> dict:
    """
    Call MediaWiki API with retry and backoff on transient errors.
    """
    params.setdefault("format", "json")
    params.setdefault("utf8", 1)
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(API_URL, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return {}


def get_wikitext_batch(words: List[str], session: requests.Session) -> dict:
    """
    Fetch raw Wiktionary wikitext for a batch of page titles.
    """
    if not words:
        return {}
    data = api_get({"action": "query", "titles": "|".join(words), "prop": "revisions", "rvprop": "content", "rvslots": "main"}, session)
    result = {w: None for w in words}
    normalized = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
    for page in data.get("query", {}).get("pages", {}).values():
        if page.get("ns") != 0 or "missing" in page:
            continue
        title = page.get("title", "")
        try:
            wt = page["revisions"][0]["slots"]["main"]["*"]
        except (KeyError, IndexError):
            continue
        original = normalized.get(title, title)
        if original in result:
            result[original] = wt
        elif title in result:
            result[title] = wt
    return result


# =========================== WORD SCRAPING ===========================
def scrape_word(word: str, wikitext: str, pos: str, pos_map: Dict[str, str], analyzer: ArmenianWordAnalyzer, queue: deque, queued: set, processed: set) -> List[dict]:
        """
        Extract and infer derivational edges for one dictionary word.
        Emits both forward and inverse (bidirectional) edges.
        Multiple parallel edges (e.g. prefixation + suffixation) are all kept.
        """
        edges = []
        seen = set()
        new_words_to_queue = set()

        def _add_edge(e) -> None:
            key = (e["source"], e["relation"], e["target"])
            if key not in seen:
                seen.add(key)
                edges.append(e)
            # also emit inverse if defined
            inv = _inverse_edge(e)
            if inv is not None:
                inv_key = (inv["source"], inv["relation"], inv["target"])
                if inv_key not in seen:
                    seen.add(inv_key)
                    edges.append(inv)

        try:
            for compound in extract_compound_section_words(wikitext, word):
                if is_proper_noun(compound):
                    continue
                compound_pos = pos_map.get(compound, "unknown")
                classified = analyzer.analyze_word(compound, compound_pos)
                if classified:
                    for e in classified:
                        _add_edge(e)
                # no fallback: unclassifiable compounds are dropped
                if compound not in pos_map and compound not in queued:
                    new_words_to_queue.add(compound)
        except Exception as exc:
            log.warning(f"Error extracting compounds for {word}: {exc}")

        try:
            for edge in analyzer.analyze_word(word, pos):
                _add_edge(edge)
                src = edge["source"]
                if src not in pos_map and src not in queued and not is_proper_noun(src) and src not in ALL_SUFFIXES and src not in PREFIXES:
                    new_words_to_queue.add(src)
        except Exception as exc:
            log.warning(f"Error in linguistic analysis for {word}: {exc}")

        if not edges:
            try:
                for component in extract_etymology_section_words(wikitext, word):
                    comp_pos = pos_map.get(component, "unknown")
                    classified = analyzer.find_compound(word, pos)
                    if classified:
                        for e in classified:
                            _add_edge(e)
                    elif component in pos_map or component in COMMON_COMPOUND_ROOTS:
                        _add_edge(_edge(source=component, relation="compound_component", target=word, target_pos=pos, source_pos=comp_pos))
                    if component not in pos_map and component not in queued:
                        new_words_to_queue.add(component)
            except Exception as exc:
                log.warning(f"Error extracting etymology for {word}: {exc}")

        for new_word in sorted(new_words_to_queue):
            if new_word not in queued and new_word not in processed:
                queue.append(new_word)
                queued.add(new_word)
                log.debug(f"Queued new word: {new_word} (from {word})")

        return edges



def run_full_reanalysis_pass(edges_file: Optional[Path] = None, nodes_file: Optional[Path] = None) -> List[dict]:
    """
    Reanalyze all nodes after the main scrape pass using the complete pos_map.
    Finds edges that were missed during scraping because pos_map was incomplete
    at the time a word was processed.
    """
    ef = edges_file or EDGES_FILE
    nf = nodes_file or NODES_FILE

    if not ef.exists() or not nf.exists():
        log.warning("No output files found for reanalysis")
        return []

    edges_df = pd.read_csv(ef, encoding="utf-8-sig")
    nodes_df = pd.read_csv(nf, encoding="utf-8-sig")

    pos_map = dict(zip(nodes_df["word"], nodes_df["pos"]))
    analyzer = ArmenianWordAnalyzer(pos_map)

    existing = set(zip(edges_df["source"], edges_df["relation"], edges_df["target"]))

    new_edges = []
    stats = {}

    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Full reanalysis"):
        word = row["word"]
        pos = row["pos"]
        for edge in analyzer.analyze_word(word, pos):
            key = (edge["source"], edge["relation"], edge["target"])
            if key not in existing:
                new_edges.append(edge)
                existing.add(key)
                stats[edge["relation"]] = stats.get(edge["relation"], 0) + 1
            inv = _inverse_edge(edge)
            if inv is not None:
                inv_key = (inv["source"], inv["relation"], inv["target"])
                if inv_key not in existing:
                    new_edges.append(inv)
                    existing.add(inv_key)
                    stats[inv["relation"]] = stats.get(inv["relation"], 0) + 1

    if new_edges:
        # ensure relation_class is filled on all new edges
        for e in new_edges:
            if not e.get("relation_class"):
                e["relation_class"] = RELATION_HIERARCHY.get(e.get("relation", ""), "other")
        pd.DataFrame(new_edges).to_csv(ef, mode="a", header=False, index=False, encoding="utf-8-sig")
        log.info(f"Full reanalysis added {len(new_edges)} new edges")
        for rel, count in sorted(stats.items(), key=lambda x: -x[1]):
            log.info(f"{rel}: {count}")

    return new_edges


def backfill_edge_pos(edges_file: Path = None, nodes_file: Path = None) -> None:
    """
    Fill in unknown source_pos and target_pos in edges.csv using the
    completed nodes.csv. Run once after main scraping is done.
    Words with no Wiktionary page stay "unknown" — those are the
    placeholder nodes and their unknown POS is correct/expected.
    """
    ef = edges_file or EDGES_FILE
    nf = nodes_file or NODES_FILE

    edges_df = pd.read_csv(ef, encoding="utf-8-sig")
    nodes_df = pd.read_csv(nf, encoding="utf-8-sig")[["word", "pos"]]
    word_to_pos = dict(zip(nodes_df["word"], nodes_df["pos"]))

    before_src = (edges_df["source_pos"] == "unknown").sum()
    before_tgt = (edges_df["target_pos"] == "unknown").sum()

    src_mask = edges_df["source_pos"] == "unknown"
    tgt_mask = edges_df["target_pos"] == "unknown"

    edges_df.loc[src_mask, "source_pos"] = (edges_df.loc[src_mask, "source"].map(word_to_pos).fillna("unknown"))
    edges_df.loc[tgt_mask, "target_pos"] = (edges_df.loc[tgt_mask, "target"].map(word_to_pos).fillna("unknown"))

    after_src = (edges_df["source_pos"] == "unknown").sum()
    after_tgt = (edges_df["target_pos"] == "unknown").sum()

    edges_df.to_csv(ef, index=False, encoding="utf-8-sig")

    log.info(f"POS backfill | source_pos: {before_src} -> {after_src} unknown")
    log.info(f"POS backfill | target_pos: {before_tgt} -> {after_tgt} unknown")


def reclassify_wiki_compound_edges(edges_file: Path = None, nodes_file: Path = None) -> None:
    """
    Attempt full reclassification of any remaining wiki_compound_member edges
    via analyze_word(). Unclassifiable edges are dropped — no wiki_compound_member
    rows survive into the final output.
    """
    ef = edges_file or EDGES_FILE
    nf = nodes_file or NODES_FILE
    edges_df = pd.read_csv(ef, encoding="utf-8-sig", dtype=str).fillna("")
    nodes_df = pd.read_csv(nf, encoding="utf-8-sig", dtype=str).fillna("")
    pos_map = dict(zip(nodes_df["word"], nodes_df["pos"]))
    analyzer = ArmenianWordAnalyzer(pos_map)
    wiki_mask = edges_df["relation"] == "wiki_compound_member"
    wiki_edges = edges_df[wiki_mask]
    clean_edges = edges_df[~wiki_mask]
    classified_rows: List[dict] = []
    reclassified = 0
    for _, row in tqdm(wiki_edges.iterrows(), total=len(wiki_edges), desc="Reclassifying wiki_compound_member"):
        compound = row["target"]
        compound_pos = pos_map.get(compound, "unknown")
        new_edges = analyzer.analyze_word(compound, compound_pos)
        if new_edges:
            for e in new_edges:
                e["relation_class"] = RELATION_HIERARCHY.get(e["relation"], "other")
                classified_rows.append(e)
                inv = _inverse_edge(e)
                if inv:
                    classified_rows.append(inv)
            reclassified += 1
    new_df = pd.DataFrame(classified_rows)
    result = (
        pd.concat([clean_edges, new_df], ignore_index=True)
        .drop_duplicates(subset=["source", "relation", "target"])
    )
    result.to_csv(ef, index=False, encoding="utf-8-sig")
    dropped = len(wiki_edges) - reclassified
    log.info(
        f"wiki_compound_member reclassification | "
        f"total={len(wiki_edges)} | reclassified={reclassified} | dropped={dropped}"
    )


# =========================== I/O ===========================
def init_csv_files(resume: bool) -> None:
    """
    Initialize output CSV files and write headers when needed.
    """
    if resume:
        if not EDGES_FILE.exists() or not NODES_FILE.exists():
            resume = False
    if not resume:
        for path, fields in [(EDGES_FILE, EDGE_FIELDS), (NODES_FILE, NODE_FIELDS)]:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fields).writeheader()


def _is_valid_source(word: str) -> bool:
    """
    Validate that a token is eligible as a graph node/source.
    """
    if not word or len(word) < 2:
        return False
    if word in ALL_SUFFIXES or word in PREFIXES:
        return False
    if is_proper_noun(word):
        return False
    return True


def append_edges(edges: List[dict], pos_map: dict) -> None:
    """
    Append validated edges to CSV with POS backfilling.
    Backfills relation_class if not already set.
    """
    if not edges:
        return
    with open(EDGES_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, EDGE_FIELDS)
        for e in edges:
            if not _is_valid_source(e["source"]):
                continue
            # backfill relation_class
            if not e.get("relation_class"):
                e["relation_class"] = RELATION_HIERARCHY.get(e.get("relation", ""), "other")
            if e["source_pos"] == "unknown":
                e["source_pos"] = pos_map.get(e["source"], "unknown")
            if e["target_pos"] == "unknown":
                inferred = False
                target = e["target"]
                for pos_type in ("noun", "adjective", "verb", "adverb"):
                    for suffix in SUFFIXES[pos_type]:
                        if target.endswith(suffix):
                            e["target_pos"] = pos_type
                            inferred = True
                            break
                    if inferred:
                        break
            w.writerow({k: e.get(k, "") for k in EDGE_FIELDS})


def append_node(word: str, pos: str, definition: str) -> None:
    """
    Append a node row with inferred linguistic feature columns.
    """
    if not _is_valid_source(word):
        return
    if pos == "unknown":
        inferred = False
        for pos_type in ("noun", "adjective", "verb", "adverb"):
            for suffix in SUFFIXES[pos_type]:
                if word.endswith(suffix):
                    pos = pos_type
                    inferred = True
                    break
            if inferred:
                break
    with open(NODES_FILE, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, NODE_FIELDS).writerow(
            {
                "word": word,
                "pos": pos,
                "definition_hy": definition,
                "animacy": infer_animacy(word, pos),
                "declension_class": infer_declension(word, pos),
                "verb_transitivity": infer_transitivity(word, pos),
                "aktionsart": infer_aktionsart(word, pos),
                "scraped_at": datetime.now(timezone.utc).isoformat()
            }
        )


def log_word(word: str, status: str, n_edges: int) -> None:
    """
    Write one JSONL processing log entry for a scraped word.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"word": word, "status": status, "n_edges": n_edges, "ts": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")


def load_checkpoint():
    """
    Load processed and queued word sets from checkpoint file.
    """
    if not CHECKPOINT_FILE.exists():
        return set(), set()
    data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return set(data.get("processed", [])), set(data.get("queued", []))


def clean_outputs() -> None:
    """
    Filter invalid/proper/suffix nodes and duplicate edge rows.
    Backfills relation_class column for edges written before this field was added.
    """
    if NODES_FILE.exists():
        df = pd.read_csv(NODES_FILE, encoding="utf-8-sig", dtype=str).fillna("")
        before = len(df)
        df = df[~df["word"].apply(is_proper_noun)]
        df = df[~df["word"].isin(ALL_SUFFIXES)]
        df = df[~df["word"].isin(PREFIXES)]
        df = df[df["word"].str.len() >= 2]
        df.to_csv(NODES_FILE, index=False, encoding="utf-8-sig")
        log.info(f"Cleaned nodes: {before} -> {len(df)} rows")

    if EDGES_FILE.exists():
        df = pd.read_csv(EDGES_FILE, encoding="utf-8-sig", dtype=str).fillna("")
        before = len(df)
        df = df[~df["source"].apply(is_proper_noun)]
        df = df[~df["target"].apply(is_proper_noun)]
        df = df[~df["source"].isin(ALL_SUFFIXES)]
        df = df[~df["source"].isin(PREFIXES)]
        df = df[df["source"].str.len() >= 2]
        # backfill relation_class for older rows
        if "relation_class" not in df.columns:
            df["relation_class"] = df["relation"].map(lambda r: RELATION_HIERARCHY.get(r, "other"))
        else:
            mask = df["relation_class"].isin(["", "nan"])
            df.loc[mask, "relation_class"] = df.loc[mask, "relation"].map(lambda r: RELATION_HIERARCHY.get(r, "other"))
        df = df.drop_duplicates(subset=["source", "relation", "target"])
        df.to_csv(EDGES_FILE, index=False, encoding="utf-8-sig")
        log.info(f"Cleaned edges: {before} -> {len(df)} rows")


def save_checkpoint(processed: set, queued: set) -> None:
    """
    Persist processed and queued word sets to checkpoint file.
    """
    CHECKPOINT_FILE.write_text(json.dumps({"processed": sorted(processed), "queued": sorted(queued)}, ensure_ascii=False), encoding="utf-8")


def resolve_out_dir(path_str: Optional[str] = None, resume: bool = False) -> None:
    """
    Resolve run output paths and create output/log directories.
    """
    global OUT_DIR, EDGES_FILE, NODES_FILE, LOG_FILE, CHECKPOINT_FILE
    base_path = Path(path_str).expanduser().resolve() if path_str else Path(__file__).parent.parent
    runs_root = base_path / "runs"
    if resume:
        existing_runs = sorted([p for p in runs_root.glob("scrape_run_*") if p.is_dir()], key=lambda p: p.name, reverse=True)
        if existing_runs:
            OUT_DIR = existing_runs[0]
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            OUT_DIR = runs_root / f"scrape_run_{ts}"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        OUT_DIR = runs_root / f"scrape_run_{ts}"
    (OUT_DIR / "output").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    EDGES_FILE = OUT_DIR / "output" / "edges.csv"
    NODES_FILE = OUT_DIR / "output" / "nodes.csv"
    LOG_FILE = OUT_DIR / "logs" / "scrape_log.jsonl"
    CHECKPOINT_FILE = OUT_DIR / "logs" / "checkpoint.txt"

# =========================== MAIN ===========================
def main() -> None:
    """
    Run end-to-end scraping, graph extraction, and post-processing.
    """
    global REQUEST_DELAY

    parser = argparse.ArgumentParser(description="Armenian Wiktionary Etymology Graph Scraper", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-file", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--max-queue", type=int, default=100000)
    parser.add_argument("--neg-sample", action="store_true", help="Generate negative samples for GNN training after scrape")
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--neg-strategy", choices=["random", "typed", "relcorr"], default="typed")
    parser.add_argument("--gnn-split", action="store_true", help="After negative sampling, split into train/val/test CSVs")
    args = parser.parse_args()

    resolve_out_dir(args.out_dir, resume=args.resume)
    REQUEST_DELAY = args.delay

    if args.clean:
        set_log_format(clean=True)
        clean_outputs()
        return

    set_log_format(clean=False)

    session = requests.Session()
    pos_map = {}
    seed_pos_map(pos_map)

    if args.resume:
        processed, queued = load_checkpoint()
        queue = deque(w for w in queued if w not in processed)
        log.info(f"Resuming: {len(processed)} pages done, {len(queue)} in queue")
    else:
        processed, queued = set(), set()
        queue = deque()

    init_csv_files(resume=args.resume)

    seed_words = []
    dic_path = Path(args.seed_file).expanduser().resolve() if args.seed_file else DIC_FILE
    if dic_path and dic_path.exists():
        with open(dic_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip().split("/")[0].strip()
                if (word and len(word) >= 2
                    and not is_proper_noun(word)
                    and word not in ALL_SUFFIXES
                    and word not in PREFIXES
                    and word not in FOREIGN_BORROWINGS):
                    seed_words.append(word)
        log.info(f"Loaded {len(seed_words)} words from {dic_path.name}")

    if not seed_words:
        seed_words = [w for w in DEFAULT_SEED_WORDS if not is_proper_noun(w) and w not in ALL_SUFFIXES and w not in PREFIXES]
        log.info(f"Using {len(seed_words)} built-in seed words.")

    for w in seed_words:
        if w not in queued:
            queue.append(w)
            queued.add(w)

    BATCH = 50

    log.info(f"Starting | words={len(queue)} | batch={BATCH} pages/request")

    total_edges = 0
    pages_visited = len(processed)
    analyzer = ArmenianWordAnalyzer(pos_map)
    pbar = tqdm(desc="Words scraped", unit=" words", initial=pages_visited)

    while queue:
        batch_words = [queue.popleft() for _ in range(min(BATCH, len(queue)))]
        wikitexts = get_wikitext_batch(batch_words, session)

        for word in batch_words:
            if word in processed:
                pbar.update(1)
                continue
            if is_proper_noun(word) or word in ALL_SUFFIXES or word in PREFIXES or word in FOREIGN_BORROWINGS:
                processed.add(word)
                pbar.update(1)
                continue

            wikitext = wikitexts.get(word)
            if not wikitext:
                processed.add(word)
                pbar.update(1)
                continue

            pos = get_primary_pos(wikitext)
            if pos == "unknown":
                pos = resolve_pos(word, pos_map)
            pos_map[word] = pos
            definition = extract_first_definition(wikitext)

            edges = scrape_word(word, wikitext, pos, pos_map, analyzer, queue, queued, processed)
            append_edges(edges, pos_map)
            append_node(word, pos, definition)
            log_word(word, "ok", len(edges))

            total_edges += len(edges)
            pages_visited += 1
            processed.add(word)

            pbar.update(1)
            pbar.set_postfix(edges=total_edges, queue=len(queue))

            if len(queue) > args.max_queue:
                log.warning(f"Queue size exceeded limit ({args.max_queue}), stopping...")
                queue.clear()
                break

        if pages_visited % 500 == 0 and pages_visited > 0:
            save_checkpoint(processed, set(queue))
            log.info(f"Checkpoint | pages={pages_visited} | edges={total_edges} | queue={len(queue)}")

        time.sleep(REQUEST_DELAY)

    pbar.close()
    save_checkpoint(processed, set(queue))

    log.info("Queue exhausted. Deduplicating...")
    try:
        e_df = pd.read_csv(EDGES_FILE, encoding="utf-8-sig")
        e_df = e_df[~e_df["source"].isin(ALL_SUFFIXES)]
        e_df = e_df[~e_df["source"].apply(is_proper_noun)]
        e_df = e_df[e_df["source"].str.len() >= 2]
        e_df.drop_duplicates(subset=["source", "relation", "target"]).to_csv(EDGES_FILE, index=False, encoding="utf-8-sig")

        n_df = pd.read_csv(NODES_FILE, encoding="utf-8-sig")
        n_df = n_df[~n_df["word"].isin(ALL_SUFFIXES)]
        n_df = n_df[~n_df["word"].isin(PREFIXES)]
        n_df = n_df[~n_df["word"].apply(is_proper_noun)]
        n_df = n_df[n_df["word"].str.len() >= 2]
        n_df.drop_duplicates(subset=["word"], keep="last").to_csv(NODES_FILE, index=False, encoding="utf-8-sig")

        log.info("Backfilling...")
        backfill_edge_pos(EDGES_FILE, NODES_FILE)

        log.info("Running reanalysis pass on all words...")
        run_full_reanalysis_pass(edges_file=EDGES_FILE, nodes_file=NODES_FILE)

        log.info("Reclassifying wiki_compound_member edges...")
        reclassify_wiki_compound_edges(edges_file=EDGES_FILE, nodes_file=NODES_FILE)

        log.info(
            f"\nDone.\nPages visited: {pages_visited}"
            f"\nUnique edges: {len(e_df)}"
            f"\nUnique nodes: {len(n_df)}"
            f"\nOutput dir: {OUT_DIR}"
        )
    except Exception as e:
        log.warning(f"Dedup failed: {e}")

    if args.neg_sample:
        neg_file = OUT_DIR / "output" / "negative_samples.csv"
        sampler = NegativeSampler(ratio=args.neg_ratio, strategy=NegativeStrategy(args.neg_strategy))
        sampler.generate(EDGES_FILE, NODES_FILE, neg_file)

    if args.gnn_split:
        neg_file_combined = OUT_DIR / "output" / "negative_samples.csv"
        split_for_gnn(neg_file_combined)
