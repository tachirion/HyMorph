import logging
from pathlib import Path
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
_tqdm_handler = TqdmLoggingHandler()
_tqdm_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_tqdm_handler)
log.propagate = False

def set_log_format(clean: bool = False) -> None:
    fmt = "%(message)s" if clean else "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = None if clean else "%H:%M:%S"
    for handler in log.handlers:
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))


API_URL = "https://hy.wiktionary.org/w/api.php"
MAX_WIKI_COMPOUNDS = 10
HEADERS = {"User-Agent": "AUA NLP Research; contact: tatev_stepanyan@edu.aua.am"}
REQUEST_DELAY = 0.5
RETRY_DELAY = 5.0
MAX_RETRIES = 3
OUT_DIR = Path(__file__).parent.parent
DIC_FILE = OUT_DIR / "data" / "armenian_eastern.dic"

ARM_UPPER_RANGE = (chr(0x0531), chr(0x0556))

# ===================== POS =====================
POS_HEADERS = {
    "Գոյական": "noun",  # [Goyakan]
    "Բայ": "verb",  # [Bay]
    "Ածական": "adjective",  # [Atsakan]
    "Մակբայ": "adverb",  # [Makbay]
    "Դերանուն": "pronoun",  # [Deranun]
    "Թվական": "numeral",  # [Tvakan]
    "Շաղկապ": "conjunction",  # [Shaghkap]
    "Վերջաբան": "particle",  # [Verjaban]
    "Ձայնարկություն": "interjection",  # [Dzaynarkutyun]
    "Վերաբերական": "modal"  # [Veraberakan]
}

POS_TEMPLATES = {
    "{{-hy-գո-}}": "noun",  # shortened forms of the ones above
    "{{-hy-բայ-}}": "verb",
    "{{-hy-ած-}}": "adjective",
    "{{-hy-մակ-}}": "adverb",
    "{{-hy-դեր-}}": "pronoun",
    "{{-hy-կապ-}}": "preposition",
    "{{-hy-միջ-}}": "interjection",
    "{{-hy-թվ-}}": "numeral",
    "{{-hy-շաղ-}}": "conjunction",
    "{{-hy-հ-}}": "particle",
    "{{-hy-եղբ-}}": "modal",
    "{{ած}}": "adjective",
    "{{գոյ}}": "noun",
    "{{բայ}}": "verb",
    "{{մակ}}": "adverb"
}

# ===================== LINGUISTIC PATTERNS =====================
MIN_COMPOUND_PART_LEN = 3
MAX_COMPOUND_WORD_LEN = 32
CONTENT_POS = {"noun", "verb", "adjective", "adverb"}
COMPOUND_BRIDGE_VOWELS = ["ա", "ե"]  # [a, e]

# ===================== RELATION HIERARCHY =====================
RELATION_HIERARCHY = {
    # suffixation
    "denominal_suffixation": "suffixation",
    "deadjectival_suffixation": "suffixation",
    "deverbal_suffixation": "suffixation",
    "demonym_suffixation": "suffixation",
    "rare_suffixation": "suffixation",
    "other_suffixation": "suffixation",
    "suffixation": "suffixation",
    # prefixation
    "negation_prefix": "prefixation",
    "intensifying_prefix": "prefixation",
    "directional_prefix": "prefixation",
    "locational_prefix": "prefixation",
    "temporal_prefix": "prefixation",
    # compounding
    "root_compound": "compounding",
    "synthetic_compound": "compounding",
    "adjective_compound": "compounding",
    "wiki_compound_member": "compounding",
    "compound_component": "compounding",
    # derivation specials
    "causative": "derivation",
    "detransitive": "derivation",
    "nominalization": "derivation",
    "diminutive": "derivation",
    "demonym_suffix": "derivation",
    "derives_noun": "derivation",
    "reduplication": "derivation",
    # inverse relations
    "denominalization": "inverse_derivation",
    "deadjectivalization": "inverse_derivation",
    "deverbalization": "inverse_derivation",
    "back_formation": "inverse_derivation",
    "anticausative_base": "inverse_derivation",
    "causative_base": "inverse_derivation",
    "denominal_base": "inverse_derivation",
    "deverbal_base": "inverse_derivation",
    "deadjectival_base": "inverse_derivation"
}

# ===================== INVERSE RELATION MAP =====================
# defines bidirectionality: forward_relation -> (inverse_relation, swap_direction)
# swap_direction=True means the inverse edge has (original_target -> original_source)
INVERSE_RELATIONS = {
    # suffixation
    "denominal_suffixation": "denominalization",
    "deadjectival_suffixation": "deadjectivalization",
    "deverbal_suffixation": "deverbalization",
    "suffixation": "back_formation",
    "nominalization": "denominalization",
    "causative": "anticausative_base",
    "detransitive": "causative_base",
    "derives_noun": "deverbal_base",
    "demonym_suffix": "demonym_base",
    # prefixation
    "negation_prefix": "negation_prefix_base",
    "intensifying_prefix": "intensifying_prefix_base",
    "directional_prefix": "directional_prefix_base",
    "locational_prefix": "locational_prefix_base",
    "temporal_prefix": "temporal_prefix_base"
}

# ===================== SUFFIXES =====================
# suffix, source_pos, target_pos, subtype (Dum-Tragut §4.1.2)
TYPED_SUFFIXES = [
    # §4.1.2.1.1 denominal noun suffixes
    ("ություն", "noun", "noun", "denominal"),  # abstract state
    ("անություն", "noun", "noun", "denominal"),
    ("կանություն", "noun", "noun", "denominal"),
    ("ուհի", "noun", "noun", "denominal"),  # feminine agent
    ("արան", "noun", "noun", "denominal"),  # place/container
    ("ոց", "noun", "noun", "denominal"),  # place
    ("ստան", "noun", "noun", "denominal"),  # country/region
    ("ցի", "noun", "noun", "demonym"),  # person from place
    ("ացի", "noun", "noun", "demonym"),
    ("եցի", "noun", "noun", "demonym"),
    ("ուն", "noun", "noun", "denominal"),
    ("ույթ", "noun", "noun", "denominal"),
    ("անք", "noun", "noun", "denominal"),
    # §4.1.2.1.2 deadjectival noun suffixes
    ("ություն", "adjective", "noun", "deadjectival"),  # duplicated suffix, resolved by source_pos
    ("ություն", "verb", "noun", "deverbal"),
    # §4.1.2.1.3 deverbal noun suffixes
    ("ում", "verb", "noun", "deverbal"),  # processual noun
    ("ացում", "verb", "noun", "deverbal"),
    ("եցում", "verb", "noun", "deverbal"),
    ("ուած", "verb", "noun", "deverbal"),
    ("ված", "verb", "noun", "deverbal"),
    ("իչ", "verb", "noun", "deverbal"),  # agent noun
    ("ակ", "verb", "noun", "deverbal"),  # agent/instrument
    ("ուկ", "verb", "noun", "deverbal"),
    ("արկու", "verb", "noun", "deverbal"),  # §4.1.2.1.3
    # §4.1.2.2 unproductive / rare noun suffixes
    ("ոն", "noun", "noun", "rare"),
    ("ոնք", "noun", "noun", "rare"),
    # §4.1.2.3.1 denominal adjective suffixes
    ("ական", "noun", "adjective", "denominal"),
    ("ային", "noun", "adjective", "denominal"),
    ("ավոր", "noun", "adjective", "denominal"),
    ("ոտ", "noun", "adjective", "denominal"),
    ("անի", "noun", "adjective", "denominal"),
    ("ենի", "noun", "adjective", "denominal"),
    ("եղեն", "noun", "adjective", "denominal"),
    ("ավուն", "noun", "adjective", "denominal"),
    ("ագ", "noun", "adjective", "denominal"),
    # §4.1.2.3.2 deadjectival adjective suffixes
    ("ավուն", "adjective", "adjective", "deadjectival"),
    ("ացիկ", "adjective", "adjective", "deadjectival"),
    ("եցիկ", "adjective", "adjective", "deadjectival"),
    # §4.1.2.3.3 deverbal adjective suffixes
    ("ելի", "verb", "adjective", "deverbal"),
    ("ալի", "verb", "adjective", "deverbal"),
    ("ողական", "verb", "adjective", "deverbal"),
    ("չական", "verb", "adjective", "deverbal"),
    ("ար", "verb", "adjective", "deverbal"),
    ("իկ", "verb", "adjective", "deverbal"),
    # §4.1.2.3.4 other adjective suffixes
    ("յա", "noun", "adjective", "other"),
    # §4.1.2.4 adverb suffixes
    ("որեն", "adjective", "adverb", "deadjectival"),
    ("ապես", "adjective", "adverb", "deadjectival"),
    ("աբար", "adjective", "adverb", "deadjectival"),
    ("գույն", "adjective", "adverb", "deadjectival"),
    # §4.1.2.5.1-5.2 verb suffixes
    ("ացնել", "noun", "verb", "denominal"),  # causative/factitive
    ("եցնել", "noun", "verb", "denominal"),
    ("ացնել", "adjective", "verb", "deadjectival"),
    ("անալ", "noun", "verb", "denominal"),  # inchoative
    ("անալ", "adjective", "verb", "deadjectival"),
    ("ել", "noun", "verb", "denominal"),
    ("ալ", "noun", "verb", "denominal"),
    # ("վել", "verb", "verb", "deverbal")  # handled separately
]

ALL_SUFFIXES = sorted(set(entry[0] for entry in TYPED_SUFFIXES), key=len, reverse=True)

# index: suffix -> list of (source_pos, target_pos, subtype) tuples
# when source_pos is unknown we iterate and pick the best match via pos_map
SUFFIX_INDEX = {}
for _suf, _src, _tgt, _sub in TYPED_SUFFIXES:
    SUFFIX_INDEX.setdefault(_suf, []).append((_src, _tgt, _sub))

SUFFIXES = {pos: sorted(set(e[0] for e in TYPED_SUFFIXES if e[2] == pos), key=len, reverse=True) for pos in ("noun", "adjective", "verb", "adverb")}

# ===================== PREFIXES =====================
# relation_label, semantic_class
PREFIX_TABLE = {
    # negation prefixes (§4.1.1, §3.4.2.4)
    "ան": ("negation_prefix", "negation"),
    "ապ": ("negation_prefix", "negation"),
    "դժ": ("negation_prefix", "negation"),
    # temporal/sequential prefix (§4.1.1)
    "ապա": ("temporal_prefix", "temporal"),
    # intensifying/degree prefixes (§4.1.1)
    "գեր": ("intensifying_prefix", "intensifying"),
    # directional/oppositional (§4.1.1)
    "հակա": ("directional_prefix", "directional"),
    "վեր": ("directional_prefix", "directional"),
    # locational/positional (§4.1.1)
    "ստոր": ("locational_prefix", "locational"),
    "ներ": ("locational_prefix", "locational"),
    # temporal (§4.1.1)
    "նախա": ("temporal_prefix", "temporal"),
    "նախ": ("temporal_prefix", "temporal")
}

PREFIXES = set(PREFIX_TABLE.keys())

# ===================== ROOT COLLECTION =====================
SYNTHETIC_VERB_ROOTS = {
    "գործ", "կիր", "կար", "բեր", "տուր", "տար",
    "ասաց", "կաց", "գնաց", "եկ", "լսեց", "կարդաց",
    "գրեց", "խոս", "պահ", "ձայն", "իմաց", "ուս",
    "հաս", "ճան", "մտ", "դուրս", "ներս"
}

ROOT_TRANSFORMATIONS = {
    "գիր": ["գր"],
    "ձեռք": ["ձեռ", "ձեռն"],
    "սիրտ": ["սրտ"],
    "տուն": ["տն"],
    "լույս": ["լուս"],
    "ջուր": ["ջր"],
    "մայր": ["մոր"],
    "հայր": ["հոր"],
    "երկիր": ["երկ", "երկր"],
    "օրենք": ["օրին", "օրեն"],
    "ձուկ": ["ձկն", "ձկան"],
    "գառ": ["գառն"],
    "դուռ": ["դռն"],
    "թռչուն": ["թռչն"],
    "հարս": ["հարսն"],
    "կես": ["կիս"],
    "մանուկ": ["մանկ"],
    "զենք": ["զին"],
    "ծաղիկ": ["ծաղկ"],
    "գույն": ["գուն"],
    "սեր": ["սիր"],
    "բույս": ["բուս"],
    "խիստ": ["խստ"],
    "կրկին": ["կրկն"],
    "դեմ": ["դիմ"],
    "պատիվ": ["պատվ"],
    "ազնիվ": ["ազնվ"],
    "ձյուն": ["ձն"],
    "բուժ": ["բույժ"],
    "թուղթ": ["թղթ"],
    "ուսում": ["ուսմ"],
    "ածուխ": ["ածխ"],
    "ծնել": ["ծն"],
    "սերել": ["սեռ"],
    "սերունդ": ["սեռ"],
    "հենք": ["հեն"],
    "հեռու": ["հեռ"],
    "սունկ": ["սնկ"],
    "խումբ": ["խմբ"],
    "կտուց": ["կտց"],
    "հունչ": ["հնչ"],
    "շունչ": ["շնչ"],
    "ցուրտ": ["ցրտ"],
    "թիվ":  ["թվ"],
    "հանուր": ["հանր"]
}

REVERSE_TRANSFORMATIONS = {}
for _original, _variants in ROOT_TRANSFORMATIONS.items():
    for _variant in _variants:
        REVERSE_TRANSFORMATIONS.setdefault(_variant, []).append(_original)

COMMON_COMPOUND_ROOTS = {
    "մոլ", "պահ", "չափ", "գործ", "տեր", "կիր", "գիր", "նկար", "կապ", "հոգի",
    "միտ", "սեր", "կամ", "տուր", "զոր", "մաս", "գունդ", "հարս", "ծով", "հող",
    "տախտակ", "կտուց", "մանուկ", "մեծ", "փոքր", "բարի", "չար", "քաղցր", "դառ",
    "կարմիր", "սև", "սպիտակ", "կանաչ", "դեղին", "խոս", "պաշտ", "զարմ", "տոհմ",
    "զարդ", "նավ", "նետ", "պատ", "փոշի", "արկ", "որս", "որդ", "ծախ",
    "բազ", "բազմ", "ձայն", "իմաց", "ուս", "պետ"
}

FOREIGN_BORROWINGS = {
    "ֆուտբոլ", "բասկետբոլ", "վոլեյբոլ", "ինտերնետ", "մոնիտոր", "պրոցեսոր",
    "բիզնես", "մարքեթինգ", "մենեջեր", "դեմոկրատիա", "ռեսպուբլիկա", "ֆեդերացիա",
    "ռեստորան", "կոնյակ", "շամպայն", "պիցցա", "ակադեմիա", "ունիվերսիտետ",
    "ֆակուլտետ", "սիստեմա", "ալկոհոլ"
}

# =========================== SEEDS ===========================
DEFAULT_SEED_WORDS = [
    "աղ", "հաց", "ջուր", "արև", "նվեր", "սեր",
    "գիր", "գրիչ", "գրություն", "գրագետ", "գրադարան", "սիրել",
    "սիրուն", "սիրահար", "լույս", "լուսին", "լուսավոր", "լուսաբաց",
    "բառ", "բառարան", "բառապաշար", "մայր", "հայր", "եղբայր",
    "քույր", "տուն", "տնային", "տնտեսություն", "գիտություն", "գիտնական",
    "գիտակ", "դպրոց", "ուսուցիչ", "ուսանող", "ուսում", "երկիր",
    "անկախ", "բնություն", "հայերեն", "օրենք", "օրինակ", "ամպ",
    "քար", "օդ", "շնորհ", "գոհ", "հեռու", "բարի",
    "որոշ", "հաճ", "մոլ"
]

SEED_WORD_POS_HINTS = {
    "աղ": "noun", "հաց": "noun", "ջուր": "noun", "արև": "noun", "նվեր": "noun", "սեր": "noun",
    "գիր": "noun", "գրիչ": "noun", "գրություն": "noun", "գրագետ": "adjective", "գրադարան": "noun",
    "սիրել": "verb", "սիրուն": "adjective", "սիրահար": "adjective", "լույս": "noun", "լուսին": "noun",
    "լուսավոր": "adjective", "լուսաբաց": "noun", "բառ": "noun", "բառարան": "noun", "բառապաշար": "noun",
    "մայր": "noun", "հայր": "noun", "եղբայր": "noun", "քույր": "noun", "տուն": "noun",
    "տնային": "adjective", "տնտեսություն": "noun", "գիտություն": "noun", "գիտնական": "noun",
    "գիտակ": "noun", "դպրոց": "noun", "ուսուցիչ": "noun", "ուսանող": "noun", "ուսում": "noun",
    "երկիր": "noun", "անկախ": "adjective", "բնություն": "noun", "հայերեն": "noun", "օրենք": "noun",
    "օրինակ": "noun", "ամպ": "noun", "քար": "noun", "օդ": "noun", "շնորհ": "noun",
    "գոհ": "adjective", "հեռու": "adverb", "բարի": "adjective", "որոշ": "adjective", "հաճ": "noun",
    "մոլ": "noun"
}

# ===================== NODE FEATURE INFERENCE =====================
# humanness/animacy: suffixes that reliably produce +human nouns (§2.1.1.1)
HUMAN_NOUN_SUFFIXES = {"իչ", "ուհի", "ցի", "ացի", "եցի", "արկու", "ուն"}

HUMAN_NOUN_STEMS = {"մարդ", "կին", "տղա", "աղջիկ", "հայ", "ուսուցիչ", "բժիշկ", "գիտնական", "ուսանող", "դատավոր", "զինվոր"}

DECLENSION_HINTS = [
    ("ություն", "i-decl"),
    ("ում", "i-decl"),
    ("ոց", "i-decl"),
    ("ույթ", "i-decl"),
    ("ու", "u-decl"),
    ("ան", "an-decl"),
    ("վա", "va-decl"),
    ("ոջ", "oj-decl")
]

# aktionsart inference from verb suffix (§2.5.4); maps suffix to aktionsart label
AKTIONSART_MAP = [
    ("անալ", "inchoative"),
    ("ացնել", "causative"),
    ("եցնել", "causative"),
    ("ցնել", "causative"),
    ("վել", "anticausative"),
    ("ել", "processual"),
    ("ալ", "processual")
]

# transitivity inference (§2.5.1.3)
INTRANSITIVE_SUFFIXES = {"անալ", "վել"}
CAUSATIVE_SUFFIXES = {"ացնել", "եցնել", "ցնել"}

EDGE_FIELDS = ["source", "source_pos", "relation", "relation_class", "target", "target_pos"]
NODE_FIELDS = ["word", "pos", "definition_hy", "animacy", "declension_class", "verb_transitivity", "aktionsart", "scraped_at"]
