from scraper._base import *
from typing import Optional, Dict, List, Tuple
import re


# ===================== SUFFIXATION =====================
def lookup_suffix(word: str, source_pos: Optional[str] = None) -> Optional[Tuple[str, List[Tuple[str, str, str]]]]:
    """
    Returns (matched_suffix, rules) using greedy longest-first matching via ALL_SUFFIXES.
    """
    for suffix in ALL_SUFFIXES:
        if word.endswith(suffix):
            rules = SUFFIX_INDEX[suffix]
            if source_pos is not None:
                filtered = [r for r in rules if r[0] == source_pos]
                return suffix, filtered if filtered else rules
            return suffix, rules
    return None


def resolve_suffix_rules(suffix: str, source_pos: str, *, fallback_to_all: bool = True) -> List[Tuple[str, str, str]]:
    """
    Returns SUFFIX_INDEX rules for suffix compatible with source_pos.
    """
    rules = SUFFIX_INDEX.get(suffix, [])
    if not rules:
        return []
    matched = [r for r in rules if r[0] == source_pos]
    if matched:
        return matched
    if source_pos == "unknown" and fallback_to_all:
        return rules
    return []


# ===================== PROPER NOUN FILTER =====================
def is_proper_noun(word: str) -> bool:
    """
    Heuristically flag proper nouns and invalid lexical entries.
    """
    if not word or len(word) < 2:
        return True
    if any(c.isdigit() for c in word):
        return True
    first_char = word[0]
    if ARM_UPPER_RANGE[0] <= first_char <= ARM_UPPER_RANGE[1]:
        return True
    return False


# ===================== NODE FEATURE INFERENCE HELPERS =====================
def infer_animacy(word: str, pos: str) -> str:
    """
    Heuristic animacy for nouns (Dum-Tragut §2.1.1.1).
    Returns '+human', '-human', or '' (unknown / non-noun).
    """
    if pos != "noun":
        return ""
    if word in HUMAN_NOUN_STEMS:
        return "+human"
    for suffix in HUMAN_NOUN_SUFFIXES:
        if word.endswith(suffix):
            return "+human"
    return "-human"


def infer_declension(word: str, pos: str) -> str:
    """
    Infer declension class from surface form (Dum-Tragut §2.1.2).
    Returns a class label or empty string.
    """
    if pos != "noun":
        return ""
    for suffix, cls in DECLENSION_HINTS:
        if word.endswith(suffix):
            return cls
    return "consonant-decl"  # default: consonant stem declensions §2.1.2.6-7


def infer_transitivity(word: str, pos: str) -> str:
    """
    Infer verb transitivity (Dum-Tragut §2.5.1.3).
    Returns 'transitive', 'intransitive', 'causative', or ''.
    """
    if pos != "verb":
        return ""
    for suf in CAUSATIVE_SUFFIXES:
        if word.endswith(suf):
            return "causative"
    for suf in INTRANSITIVE_SUFFIXES:
        if word.endswith(suf):
            return "intransitive"
    return "transitive"


def infer_aktionsart(word: str, pos: str) -> str:
    """
    Infer aktionsart from verb morphology (Dum-Tragut §2.5.4).
    Returns an aktionsart label or ''.
    """
    if pos != "verb":
        return ""
    for suf, label in AKTIONSART_MAP:
        if word.endswith(suf):
            return label
    return ""


# ===================== WORD ANALYZER =====================
def _edge(source: str, relation: str, target: str, target_pos: str, source_pos: str = "unknown") -> dict:
    """
    Direction always from base to derived.
    Fields include 'relation_class' from RELATION_HIERARCHY for GNN edge-type schemas.
    """
    return {
        "source": source.strip(),
        "source_pos": source_pos,
        "relation": relation,
        "relation_class": RELATION_HIERARCHY.get(relation, "other"),
        "target": target.strip(),
        "target_pos": target_pos
    }


def _inverse_edge(forward_edge: dict) -> Optional[dict]:
    """
    Emit the inverse (bidirectional) edge for a given forward edge.
    Returns None if no inverse relation is defined for this relation type.
    The inverse swaps source/target so that the derived form points back to its base.
    """
    inv_rel = INVERSE_RELATIONS.get(forward_edge["relation"])
    if inv_rel is None:
        return None
    return {
        "source": forward_edge["target"],
        "source_pos": forward_edge["target_pos"],
        "relation": inv_rel,
        "relation_class": RELATION_HIERARCHY.get(inv_rel, "inverse_derivation"),
        "target": forward_edge["source"],
        "target_pos": forward_edge["source_pos"]
    }


def get_all_root_variants(word: str) -> List[str]:
    variants = [word]
    if word in ROOT_TRANSFORMATIONS:
        variants.extend(ROOT_TRANSFORMATIONS[word])
    for original in REVERSE_TRANSFORMATIONS.get(word, []):
        if original != word:
            variants.append(original)
            if original in ROOT_TRANSFORMATIONS:
                variants.extend(ROOT_TRANSFORMATIONS[original])
    return list(set(variants))


class ArmenianWordAnalyzer:
    """
    Analyzes Armenian words into derivational graph edges.

    Edge direction: source = base/underived form, target = derived/complex form.

    Relation types produced:
        denominal/deadjectival/deverbal_suffixation  - §4.1.2
        negation/intensifying/directional/locational/temporal_prefix  - §4.1.1
        root_compound, synthetic_compound, adjective_compound  - §4.2
        detransitive   - §3.1.2.2 multifunctional -վ-
        causative      - §3.1.2.1 -ցնել/-եցնել/-ացնել
        nominalization - deverbal noun formation
        derives_noun   - denominal verb -> base noun direction
        diminutive     - §4.3.3
        demonym_suffix - §4.1.2.1.1 place -> person
        reduplication  - §4.3.1
        compound_component - etymology section fallback
    """

    MIN_ROOT_LEN = 2

    def __init__(self, pos_map: Dict[str, str]):
        self.pos_map = pos_map

    def is_known_word(self, word: str) -> bool:
        if not word or len(word) < self.MIN_ROOT_LEN:
            return False
        if word in FOREIGN_BORROWINGS:
            return False
        if word in ALL_SUFFIXES:
            return False
        if word in PREFIXES:
            return False
        return word in self.pos_map or word in COMMON_COMPOUND_ROOTS

    def is_known_word_for_compound(self, word: str) -> bool:
        """
        Requires the word to be in pos_map with a not 'unknown' POS or in COMMON_COMPOUND_ROOTS.
        Short words (< 4 chars) additionally require a pos_map entry.
        """
        if not word or len(word) < self.MIN_ROOT_LEN:
            return False
        if word in FOREIGN_BORROWINGS or word in ALL_SUFFIXES or word in PREFIXES:
            return False
        if word in COMMON_COMPOUND_ROOTS:
            return True
        pos = self.pos_map.get(word)
        if pos and pos != "unknown":
            return True
        # short words with unknown pos are too noisy as compound anchors
        return False

    def _strip_bridge_vowel(self, stem: str) -> Optional[str]:
        for vowel in COMPOUND_BRIDGE_VOWELS:
            if stem.endswith(vowel) and len(stem) > len(vowel) + self.MIN_ROOT_LEN:
                return stem[: -len(vowel)]
        return None

    def find_compound(self, word: str, word_pos: str) -> List[dict]:
        """
        Classify compound type.
        - root_compound §4.2.1.1 (verbless)
        - synthetic_compound §4.2.1.2 (verbal root as second element)
        - adjective_compound §4.2.2
        """
        if len(word) < 6:
            return []

        def _part_anchored(surface_part: str, candidate: str) -> bool:
            if candidate in PREFIXES:
                return False
            if candidate == surface_part:
                return True
            # allow standard stem alternations only (final 1 char may differ)
            if (len(candidate) >= MIN_COMPOUND_PART_LEN
                    and len(candidate) >= len(surface_part) - 1
                    and surface_part.startswith(candidate[:-1])):
                return True
            return False

        def make_compound_edges(p1, p2, surface_p1, ctype):
            p1_pos = self.pos_map.get(p1, "unknown")
            p2_pos = self.pos_map.get(p2, "unknown")
            return [
                _edge(p1, ctype, word, word_pos, source_pos=p1_pos),
                _edge(p2, ctype, word, word_pos, source_pos=p2_pos),
            ]

        def classify(p1, p2):
            p1_pos = self.pos_map.get(p1, "unknown")
            p2_pos = self.pos_map.get(p2, "unknown")
            if p2 in SYNTHETIC_VERB_ROOTS or p2_pos == "verb":
                return "synthetic_compound"
            if p1_pos == "adjective" or p2_pos == "adjective":
                return "adjective_compound"
            return "root_compound"

        def candidate_ok(candidate: str) -> bool:
            """Word must be known (with real POS) AND not a prefix."""
            if candidate in PREFIXES:
                return False
            if not self.is_known_word_for_compound(candidate):
                return False
            return True

        # ── with bridge vowel (հոդակապ §4.2.1.1) ──
        # Iterate longest-first-part so we don't under-split at «կան»
        for i in range(len(word) - MIN_COMPOUND_PART_LEN - 1, MIN_COMPOUND_PART_LEN - 1, -1):
            surface_p1 = word[:i]
            remaining = word[i:]
            for vowel in COMPOUND_BRIDGE_VOWELS:
                if not remaining.startswith(vowel):
                    continue
                surface_p2 = remaining[len(vowel):]
                if len(surface_p1) < MIN_COMPOUND_PART_LEN or len(surface_p2) < MIN_COMPOUND_PART_LEN:
                    continue
                if surface_p2 in ALL_SUFFIXES and not self.is_known_word_for_compound(surface_p2):
                    continue
                for p1 in get_all_root_variants(surface_p1):
                    if not candidate_ok(p1):
                        continue
                    if not _part_anchored(surface_p1, p1):
                        continue
                    if len(p1) == 2 and p1 not in self.pos_map:
                        continue
                    for p2 in get_all_root_variants(surface_p2):
                        if not candidate_ok(p2):
                            continue
                        if not _part_anchored(surface_p2, p2):
                            continue
                        if len(p2) == 2 and p2 not in self.pos_map:
                            continue
                        ctype = classify(p1, p2)
                        return make_compound_edges(p1, p2, surface_p1, ctype)

        # ── direct juxtaposition §4.2.1.1 ──
        for i in range(len(word) - MIN_COMPOUND_PART_LEN, MIN_COMPOUND_PART_LEN - 1, -1):
            surface_p1 = word[:i]
            surface_p2 = word[i:]
            if len(surface_p1) < MIN_COMPOUND_PART_LEN or len(surface_p2) < MIN_COMPOUND_PART_LEN:
                continue
            if surface_p2 in ALL_SUFFIXES and not self.is_known_word_for_compound(surface_p2):
                continue
            for p1 in get_all_root_variants(surface_p1):
                if not candidate_ok(p1):
                    continue
                if not _part_anchored(surface_p1, p1):
                    continue
                for p2 in get_all_root_variants(surface_p2):
                    if not candidate_ok(p2):
                        continue
                    if not _part_anchored(surface_p2, p2):
                        continue
                    ctype = classify(p1, p2)
                    return make_compound_edges(p1, p2, surface_p1, ctype)

        return []

    def find_suffixation(self, word: str, word_pos: str) -> List[dict]:
        """
        Use SUFFIX_INDEX to emit typed derivation edges.
        Returns ALL valid base->derived edges (multiple derivation paths allowed,
        e.g. "գրություն" from both "գիր" and "գրել").
        (Dum-Tragut §4.1.2)
        """
        raw_candidates = []

        for suffix in ALL_SUFFIXES:
            if not word.endswith(suffix):
                continue
            if len(word) <= len(suffix) + self.MIN_ROOT_LEN:
                continue

            raw_stem = word[: -len(suffix)]
            stems_to_try = [raw_stem]
            clean = self._strip_bridge_vowel(raw_stem)
            if clean:
                stems_to_try.append(clean)
            for variant in get_all_root_variants(raw_stem):
                if variant not in stems_to_try:
                    stems_to_try.append(variant)

            _verb_endings = ("ել", "ալ", "իլ")
            for _s in list(stems_to_try):
                for _ve in _verb_endings:
                    _candidate = _s + _ve
                    if _candidate not in stems_to_try and self.is_known_word(_candidate):
                        stems_to_try.append(_candidate)

            for stem in stems_to_try:
                if not self.is_known_word(stem):
                    continue
                actual_src_pos = self.pos_map.get(stem, "unknown")
                typed_entries = resolve_suffix_rules(suffix, actual_src_pos)

                if typed_entries:
                    for src_pos, tgt_pos, subtype in typed_entries:
                        relation = f"{subtype}_suffixation"
                        raw_candidates.append((len(stem), stem, suffix, relation, actual_src_pos, tgt_pos))
                else:
                    raw_candidates.append((len(stem), stem, suffix, "suffixation", actual_src_pos, word_pos))

        if not raw_candidates:
            return []

        seen_paths = set()
        deduped = []
        for cand in sorted(raw_candidates, key=lambda t: (-t[0], len(t[2]))):
            path_key = (cand[1], cand[3])
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                deduped.append(cand)

        edges = []
        for _, best_stem, _, relation, src_pos, tgt_pos in deduped:
            edges.append(_edge(best_stem, relation, word, tgt_pos, source_pos=src_pos))
        return edges

    def find_prefixation(self, word: str, word_pos: str) -> List[dict]:
        """
        Emit typed prefix relation labels (Dum-Tragut §4.1.1)
        """
        for prefix in sorted(PREFIX_TABLE.keys(), key=len, reverse=True):
            if not word.startswith(prefix) or len(word) <= len(prefix) + self.MIN_ROOT_LEN:
                continue
            relation_label, _ = PREFIX_TABLE[prefix]
            stem = word[len(prefix):]
            candidates = list(dict.fromkeys([stem]
                + [stem[len(v):] for v in COMPOUND_BRIDGE_VOWELS if stem.startswith(v)]
                + get_all_root_variants(stem)
            ))
            for candidate in candidates:
                if self.is_known_word(candidate):
                    return [_edge(candidate, relation_label, word, word_pos, source_pos=self.pos_map.get(candidate, "unknown"))]
        return []

    def find_detransitivisation(self, word: str, word_pos: str = "") -> List[dict]:
        """
        Model the multifunctional -վ- suffix (Dum-Tragut §3.1.2.2).
        Produces a 'detransitive' edge. The relation string intentionally omits
        diathesis subtype (see below) as no other relation uses this label,
        so there is no ambiguity in the current schema.

        Dum-Tragut §3.1.2.2 identifies four diathesis types sharing this suffix:
        (1) passive (§3.1.2.2.1),
        (2) anticausative (§3.1.2.2.2),
        (3) reflexive (§3.1.2.2.3),
        (4) reciprocal (§3.1.2.2.4)
        All four are morphologically identical on the surface; true disambiguation
        requires syntactic context (argument structure, verb semantics, discourse)
        that is not available during single-word scraping. Anticausative is used
        as the sole label here because it is the most frequent function of -վ- in
        Modern Eastern Armenian (Dum-Tragut §3.1.2.2.2) and the safest default
        when context is absent.
        """
        if word_pos not in ("verb", "unknown", ""):
            return []
        if not word.endswith("վել"):
            return []
        stem = word[:-3]
        for verb_ending in ("ել", "ալ", "իլ"):
            base = stem + verb_ending
            if self.is_known_word(base) and self.pos_map.get(base, "") == "verb":
                return [_edge(base, "detransitive", word, "verb", source_pos="verb")]
            for variant in get_all_root_variants(stem):
                base_v = variant + verb_ending
                if self.is_known_word(base_v) and self.pos_map.get(base_v, "") == "verb":
                    return [_edge(base_v, "detransitive", word, "verb", source_pos="verb")]
        return []

    def find_causative(self, word: str, word_pos: str = "") -> List[dict]:
        """
        Model the causative suffix -ցնել/-եցնել/-ացնել as a 'causative' verb -> verb edge. (Dum-Tragut §3.1.2.1)
        """
        if word_pos not in ("verb", "unknown", ""):
            return []
        causative_suffixes = [("ացնել", ["ել", "ալ", "անալ"]), ("եցնել", ["ել", "ալ"]), ("ցնել", ["ել", "ալ"])]
        for caus_suf, base_endings in causative_suffixes:
            if not word.endswith(caus_suf):
                continue
            stem = word[: -len(caus_suf)]
            for base_end in base_endings:
                base = stem + base_end
                if self.is_known_word(base) and self.pos_map.get(base, "") == "verb":
                    return [_edge(base, "causative", word, "verb", source_pos="verb")]
                
                for vowel in COMPOUND_BRIDGE_VOWELS:
                    if stem.endswith(vowel):
                        base2 = stem[: -len(vowel)] + base_end
                        if self.is_known_word(base2) and self.pos_map.get(base2, "") == "verb":
                            return [_edge(base2, "causative", word, "verb", source_pos="verb")]
        return []

    def find_reduplication(self, word: str, word_pos: str) -> List[dict]:
        """
        MEA reduplication patterns (Dum-Tragut §4.3.1).
        Patterns:
          X-X (hyphenated: կամաց-կամաց ([kamats-kamats], bit by bit), մի-մի ([mi-mi]))
          X X (space-separated: handled if caller passes normalized input)
        """
        if "-" in word:
            parts = word.split("-", 1)
            if len(parts) == 2 and parts[0] == parts[1] and self.is_known_word(parts[0]):
                return [_edge(parts[0], "reduplication", word, word_pos, source_pos=self.pos_map.get(parts[0], "unknown"))]
        return []

    def find_diminutive(self, word: str, word_pos: str) -> List[dict]:
        diminutive_suffixes = ["իկ", "ակ", "ուկ", "ույկ"]
        candidates = []
        for suffix in diminutive_suffixes:
            if not word.endswith(suffix) or len(word) <= len(suffix) + self.MIN_ROOT_LEN:
                continue
            stem = word[: -len(suffix)]
            if stem.endswith("ն"):
                stem = stem[:-1]
            for variant in get_all_root_variants(stem):
                if self.is_known_word(variant):
                    candidates.append((len(variant), variant))
                    break
        if not candidates:
            return []
        _, best_stem = max(candidates, key=lambda t: t[0])
        return [_edge(best_stem, "diminutive", word, word_pos, source_pos=self.pos_map.get(best_stem, "unknown"))]

    def find_nominalization(self, word: str, word_pos: str) -> List[dict]:
        nom_map = [("ացում", ["ացնել", "անալ"]), ("եցում", ["եցնել"]), ("ում", ["ել", "ալ", "վել"])]
        for noun_suf, verb_endings in nom_map:
            if word.endswith(noun_suf) and len(word) > len(noun_suf) + 2:
                stem = word[: -len(noun_suf)]
                for verb_end in verb_endings:
                    candidate = stem + verb_end
                    if self.is_known_word(candidate) and self.pos_map.get(candidate, "") == "verb":
                        return [_edge(candidate, "nominalization", word, word_pos, source_pos="verb")]
        return []

    def find_verb_to_noun(self, word: str, word_pos: str) -> List[dict]:
        """
        Emits a denominal base edge: noun (source) -> denominal verb (target).
        Called when the current word is a verb derived from a noun stem.
        Handles direct stems, bridge-vowel stems, and compound nouns
        whose root variant equals the bare verb stem.
        """
        if word_pos != "verb":
            return []
        for suffix in ["ել", "ալ", "իլ", "անալ", "եցնել", "ացնել", "վել"]:
            if not word.endswith(suffix) or len(word) <= len(suffix) + 2:
                continue
            stem = word[: -len(suffix)]
            # direct noun match
            if self.is_known_word(stem) and self.pos_map.get(stem, "") == "noun":
                return [_edge(stem, "derives_noun", word, word_pos, source_pos=self.pos_map.get(stem, "noun"))]
            # bridge-vowel stripped match
            for vowel in COMPOUND_BRIDGE_VOWELS:
                if stem.endswith(vowel):
                    clean = stem[: -len(vowel)]
                    if self.is_known_word(clean) and self.pos_map.get(clean, "") == "noun":
                        return [_edge(clean, "derives_noun", word, word_pos, source_pos=self.pos_map.get(clean, "noun"))]
            # ց-restored match
            candidate = stem + "ց"
            if self.is_known_word(candidate) and self.pos_map.get(candidate, "") == "noun":
                return [_edge(candidate, "derives_noun", word, word_pos, source_pos=self.pos_map.get(candidate, "noun"))]
            # compound noun whose standard root variant equals stem
            for noun, pos in self.pos_map.items():
                if pos != "noun":
                    continue
                for variant in get_all_root_variants(noun):
                    if variant == stem:
                        return [_edge(noun, "derives_noun", word, word_pos, source_pos="noun")]
        return []

    def find_place_person_suffix(self, word: str, word_pos: str) -> List[dict]:
        """
        Demonym: place (source) -> person (target). (§4.1.2.1.1)
        """
        place_suffixes = ["ցի", "ացի", "եցի"]
        for suffix in place_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stem = word[: -len(suffix)]
                if self.is_known_word(stem):
                    return [_edge(stem, "demonym_suffix", word, word_pos, source_pos=self.pos_map.get(stem, "unknown"))]
        return []

    def analyze_word(self, word: str, word_pos: str) -> List[dict]:
        """
        Return ALL applicable derivational edges for a word.

        Priority groups (from most specific to most general):
          GROUP 1 – unambiguous morphological processes: reduplication, diminutive,
                    detransitivisation, causative, place-person suffix, nominalization,
                    verb-to-noun.  All non-empty results from this group are kept.
          GROUP 2 – typed suffixation (may produce multiple edges for multiple bases).
          GROUP 3 – prefixation (at most one prefix per word in MEA).
          GROUP 4 – compounding (structural).
        A word may have edges from multiple groups (e.g. prefixation + suffixation).
        """
        if word in FOREIGN_BORROWINGS:
            return []

        all_edges = []
        seen_keys = set()

        def _collect(new_edges: List[dict]) -> None:
            for e in new_edges:
                key = (e["source"], e["relation"], e["target"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_edges.append(e)

        # Group 1: specific processes; always run all of them
        for finder in [
            self.find_reduplication,
            self.find_place_person_suffix,
            self.find_diminutive,
            self.find_detransitivisation,
            self.find_causative,
            self.find_nominalization,
            self.find_verb_to_noun,
        ]:
            _collect(finder(word, word_pos))

        # Group 2: typed suffixation (multi-derivation paths)
        _collect(self.find_suffixation(word, word_pos))

        # Group 3: prefixation (independent of suffix findings)
        _collect(self.find_prefixation(word, word_pos))

        # Group 4: compounding; run independently; compound edges complement
        suffix_fired = any(
            e["relation"].endswith("_suffixation") or
            e["relation"] in ("nominalization", "causative", "detransitive")
            for e in all_edges
        )
        has_compound_edge = any(
            e["relation"] in ("root_compound", "synthetic_compound", "adjective_compound")
            for e in all_edges
        )
        if not suffix_fired and not has_compound_edge:
            _collect(self.find_compound(word, word_pos))

        return all_edges


# ===================== WIKITEXT EXTRACTION HELPERS =====================
def extract_compound_section_words(wikitext: str, title: str) -> List[str]:
    """
    Extract compound words from the Բաղադրյալ բառեր ([Baghadryal barer], Compound words) section.

    hy.wiktionary.org uses three encoding styles:
      1. Plain/piped wikilinks under the section heading:
             * [[word]]   or   [[word|display text]]
      2. Template: {{բաղ|word1|word2|...}}
      3. {{der-top}} / {{der-bottom}} blocks with * [[word]] bullets.
    Heading depth may be ==, ===, or ====.
    """
    seen = set()
    compounds = []

    def _add(word: str) -> None:
        """
        Accept only clean single-word Armenian strings.
        """
        word = word.strip()
        if not word or len(word) < 2 or word == title:
            return
        if "Կատեգորիա" in word:  # [Kategoria], Category
            return
        # reject multiword phrases and apostrophe cross-references
        if any(c in word for c in (" ", chr(0x2019), chr(0x27))):
            return
        # every character must be Armenian (U+0531-U+058F) or a hyphen
        if not all((0x0531 <= ord(c) <= 0x058F) or c == "-" for c in word):
            return
        if word not in seen:
            seen.add(word)
            compounds.append(word)

    WL_RE = re.compile('\\[\\[([^\\]|#:]+)(?:\\|[^\\]]*)?\\]\\]')

    # named section heading
    if "Բաղադրյալ բառեր" in wikitext:
        sec_re = r"={2,4}\s*" + re.escape("Բաղադրյալ բառեր") + r"\s*={2,4}\s*(.*?)(?=\n={2,4}|\Z)"
        m = re.search(sec_re, wikitext, re.DOTALL)
        if m:
            for lm in WL_RE.finditer(m.group(1)):
                _add(lm.group(1))
            bagh_re = r"\{\{" + re.escape("բաղ") + r"\|([^}]+)\}\}"
            for lm in re.finditer(bagh_re, m.group(1)):
                for part in lm.group(1).split("|"):
                    if "=" not in part and not part.strip().isdigit():
                        _add(part)

    # {{der-top}} / {{der-bottom}} blocks
    for der_m in re.finditer(r"\{\{der-top\}\}(.*?)\{\{der-bottom\}\}", wikitext, re.DOTALL):
        for lm in WL_RE.finditer(der_m.group(1)):
            _add(lm.group(1))
    return compounds


def extract_etymology_section_words(wikitext: str, title: str) -> List[str]:
    """
    Extract etymology components from Ստուգաբանություն ([Stugabanutyun], Etymology) section as fallback.
    """
    if "Ստուգաբանություն" not in wikitext:
        return []
    pattern = r"==\s*Ստուգաբանություն\s*==\s*(.*?)(?=\n=|\Z)"
    match = re.search(pattern, wikitext, re.DOTALL)
    if not match:
        return []
    etymology_section = match.group(1)
    components = []
    for pat in [
        r"\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]\s*\+\s*\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]",
        r"\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]-\s*\+\s*\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]",
        r"\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]\s*\+\s*-\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]"
    ]:
        for m in re.findall(pat, etymology_section):
            for token in m:
                if token and len(token) > 1 and token != title and "Կատեգորիա" not in token:
                    components.append(token.strip())
    return list(set(components))


def get_primary_pos(wikitext: str) -> str:
    """
    Infer primary POS from Armenian section templates or headers.
    """
    for tmpl, pos in POS_TEMPLATES.items():
        if tmpl in wikitext:
            return pos
    for header, pos in POS_HEADERS.items():
        if re.search(r"={2,4}\s*" + re.escape(header) + r"\s*={2,4}", wikitext):
            return pos
    return "unknown"


def extract_first_definition(wikitext: str) -> str:
    """
    Extract and normalize the first dictionary definition line.
    """
    for line in wikitext.splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("#:") and not line.startswith("#*"):
            clean = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", line[2:])
            clean = re.sub(r"\{\{[^}]+\}\}", "", clean)
            clean = re.sub(r"'{2,3}", "", clean).strip()
            return clean[:300]
    return ""


# =========================== SEEDS ===========================
def seed_pos_map(pos_map: Dict[str, str]) -> None:
    """
    Populate POS map with curated seed POS hints.
    """
    for word, pos in SEED_WORD_POS_HINTS.items():
        if word not in pos_map:
            pos_map[word] = pos


def resolve_pos(word: str, pos_map: Dict[str, str]) -> str:
    """
    Resolve POS from known mappings or suffix-based fallback.
    """
    if word in pos_map:
        return pos_map[word]
    for pos_type in ("noun", "adjective", "verb", "adverb"):
        for suffix in SUFFIXES[pos_type]:
            if word.endswith(suffix):
                log.debug(f"POS fallback for '{word}': '{pos_type}' from suffix '{suffix}'")
                return pos_type
    return "unknown"
