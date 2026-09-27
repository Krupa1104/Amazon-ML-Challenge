"""Name / address normalization.

Design rules:
  * Country-agnostic: no branch on the country label. Rules cover US, India and common
    French forms (SARL, bd, av...) but everything falls back to generic char/token logic,
    so an unseen country still gets sensible output.
  * Same function for every source, so both sides of a pair are normalized identically.
  * Produces several "views" per record; later stages pick what they need.

Output columns (all strings):
  name_norm   full normalized name
  name_core   name without legal suffixes / 'the' / DBA part    -> main matching field
  name_key    name_core tokens sorted (word-order invariant)     -> exact-match / blocking key
  name_dba    text after 'dba' / 'doing business as' (often empty)
  addr_norm   normalized address, token order preserved
  addr_bag    unique address tokens sorted (component-order invariant)
  addr_nums   unique digit groups sorted, e.g. '11 16 23 37' (house nos, units, PIN/ZIP)
"""
import re
import unicodedata

import pandas as pd

try:                                   # ISC-licensed transliteration: देवनागरी / বাংলা -> Latin
    from anyascii import anyascii as _anyascii
except ImportError:                    # pragma: no cover
    _anyascii = None

NORM_VERSION = "v2"                    # bump when rules change -> new cache files

# ---------------------------------------------------------------- basic cleaning
_COMBINING = re.compile(r"[\u0300-\u036f]")
_APOS = re.compile(r"[\'\u2019\u2018`\u00b4]")
_NONWORD = re.compile(r"[\W_]+")          # anything not a letter/digit (unicode-aware)
_DIGITS = re.compile(r"\d+")
_DBA = re.compile(r"\b(?:d\s*/?\s*b\s*/?\s*a|doing business as|trading as|t\s*/\s*a|a\s*/\s*k\s*/\s*a)\b")
_BEHIND = re.compile(r"\bb\s*/\s*h\b")   # 'B/H' = behind (India)
_CARE_OF = re.compile(r"\bc\s*/\s*o\b")


def fold(s: str) -> str:
    """lowercase, strip accents (é->e), transliterate non-Latin scripts to Latin."""
    if not s.isascii():
        s = _COMBINING.sub("", unicodedata.normalize("NFKD", s))
        if _anyascii is not None and not s.isascii():
            s = _anyascii(s)
    return s.lower()


# ---------------------------------------------------------------- names
_NAME_MAP = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd",
    "private": "pvt", "pvtltd": "pvt ltd", "brothers": "bros", "international": "intl",
    "manufacturing": "mfg", "centre": "center", "ctr": "center", "associates": "assoc",
    "enterprises": "enterprise", "services": "service", "svcs": "service",
    "technologies": "tech", "technology": "tech", "saint": "st",
    "et": "and", "societe": "ste",
}
# dropped from name_core (legal forms in US / India / France and a few common others)
_LEGAL = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "pc", "plc", "pllc", "opc",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ste", "gmbh", "ag", "bv", "nv",
    "the",
}


def _name_tokens(raw: str):
    s = fold(raw)
    s = _APOS.sub("", s)                         # orelee's -> orelees
    s = s.replace(".", "")                       # l.l.c. -> llc, pvt. -> pvt
    s = s.replace("&", " and ").replace("+", " plus ")
    s = _DBA.sub(" dba ", s)
    toks = []
    for t in _NONWORD.sub(" ", s).split():
        for u in _NAME_MAP.get(t, t).split():
            if not toks or toks[-1] != u:        # 'export export' -> 'export'
                toks.append(u)
    return toks


def normalize_name(raw: str):
    toks = _name_tokens(raw if isinstance(raw, str) else "")
    if "dba" in toks:
        i = toks.index("dba")
        main, dba = toks[:i], toks[i + 1:]
    else:
        main, dba = toks, []
    core = [t for t in main if t not in _LEGAL]
    while core and core[-1] == "and":
        core.pop()
    while core and core[0] == "and":
        core.pop(0)
    if not core:                                  # name made only of legal words: keep it
        core = main
    name_norm = " ".join(t for t in toks if t != "dba")
    return name_norm, " ".join(core), " ".join(sorted(core)), " ".join(dba)


# ---------------------------------------------------------------- addresses
_ADDR_MAP = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "aven": "ave",
    "drive": "dr", "drv": "dr", "boulevard": "blvd", "bd": "blvd", "lane": "ln",
    "circle": "cir", "court": "ct", "place": "pl", "highway": "hwy", "parkway": "pkwy",
    "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg",
    "floor": "fl", "flr": "fl", "number": "no", "num": "no",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "nr": "near", "opposite": "opp", "opp": "opp", "bh": "behind",
    "sector": "sec", "nagar": "ngr", "ngr": "ngr",
}
_ADDR_DROP = {"no", "door", "hno", "dno"}          # 'Door No 22' / 'H.No 22' / '#22' -> '22'
_HOUSE_NO = re.compile(r"\b(?:h|house|d|door|flat|plot)\s*no\b")
_PO_BOX = re.compile(r"\bp\s*o\s*box\b")
_PLACE_PHRASES = {
    # US states (full name -> USPS code)
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    # Indian states / UTs (common forms -> short code)
    "west bengal": "wb", "paschimbanga": "wb", "pscimbng": "wb", "maharashtra": "mh",
    "karnataka": "ka", "tamil nadu": "tn", "telangana": "tg", "andhra pradesh": "ap",
    "uttar pradesh": "up", "madhya pradesh": "mp", "gujarat": "gj", "rajasthan": "rj",
    "delhi": "dl", "new delhi": "dl", "kerala": "kl", "punjab": "pb", "haryana": "hr",
    "bihar": "br", "odisha": "od", "orissa": "od", "jharkhand": "jh", "uttarakhand": "uk",
    "chhattisgarh": "cg", "goa": "ga",
    # city aliases
    "calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai", "bangalore": "bengaluru",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
}
_PLACE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _PLACE_PHRASES), key=len, reverse=True)) + r")\b")
LANDMARK_WORDS = {"near", "opp", "behind", "beside", "next", "adjacent", "pres"}


def normalize_address(raw: str):
    s = fold(raw if isinstance(raw, str) else "")
    s = _APOS.sub("", s)
    s = _BEHIND.sub(" behind ", s)
    s = _CARE_OF.sub(" careof ", s)
    s = _PO_BOX.sub(" pobox ", s)
    s = _NONWORD.sub(" ", s)                      # dots/commas/slashes/# -> space
    s = _HOUSE_NO.sub(lambda m: m.group(0).split()[0] if m.group(0).split()[0] in ("flat", "plot") else " ", s)
    s = _PLACE_RE.sub(lambda m: _PLACE_PHRASES[m.group(1)], s)
    toks, prev = [], None
    for t in s.split():
        t = _ADDR_MAP.get(t, t)
        if t in _ADDR_DROP:
            continue
        if t != prev:                             # 'unit unit 2' -> 'unit 2'
            toks.append(t)
        prev = t
    nums = sorted(set(_DIGITS.findall(s)), key=lambda x: (len(x), x))
    return " ".join(toks), " ".join(sorted(set(toks))), " ".join(nums)


# ---------------------------------------------------------------- frame-level
NORM_COLS = ["name_norm", "name_core", "name_key", "name_dba", "addr_norm", "addr_bag", "addr_nums"]


def normalize_frame(df: pd.DataFrame, chunk: int = 500_000, verbose: bool = True) -> pd.DataFrame:
    """df has raw columns [entity_id, business_name, business_address, country]."""
    parts = []
    for start in range(0, len(df), chunk):
        sub = df.iloc[start:start + chunk]
        names = [normalize_name(x) for x in sub["business_name"].tolist()]
        addrs = [normalize_address(x) for x in sub["business_address"].tolist()]
        out = pd.DataFrame(names, columns=NORM_COLS[:4])
        out[NORM_COLS[4:]] = pd.DataFrame(addrs)
        out.insert(0, "entity_id", sub["entity_id"].to_numpy())
        out.insert(1, "country", sub["country"].to_numpy())
        parts.append(out.astype("string[pyarrow]"))
        if verbose:
            print(f"  normalized {min(start + chunk, len(df)):,}/{len(df):,}", end="\r")
    if verbose:
        print()
    return pd.concat(parts, ignore_index=True)


def load_normalized(split: str, source: str, columns=None) -> pd.DataFrame:
    """Cached normalized view of a source. First call takes ~1-2 min per million rows."""
    from .config import CACHE_DIR
    from .io_utils import load_source
    pq = CACHE_DIR / f"norm_{NORM_VERSION}_{split}_{source}.parquet"
    if not pq.exists():
        print(f"normalizing {split}/{source} ...")
        normalize_frame(load_source(split, source)).to_parquet(pq, index=False)
    return pd.read_parquet(pq, columns=columns, dtype_backend="pyarrow")


# ---------------------------------------------------------------- phonetic skeleton (v3)
# Handles phonetic re-spellings seen in the data: inphra~infra, praivet~private, gujrat~gujarat,
# ist~east, saut~south. Drop vowels after unifying similar sounds; repeated letters collapse.
_SK_SUBS = [("ph", "f"), ("ck", "k"), ("sh", "s"), ("th", "t"), ("kh", "k"), ("gh", "g"),
            ("bh", "b"), ("dh", "d"), ("ch", "k"), ("q", "k"), ("c", "k"), ("z", "s"),
            ("w", "v"), ("x", "ks"), ("h", "")]
_VOWELS = str.maketrans("", "", "aeiouy")


def skeleton(tok: str) -> str:
    if not tok or tok.isdigit():
        return tok
    s = tok
    for a, b in _SK_SUBS:
        s = s.replace(a, b)
    s = s.translate(_VOWELS)
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out) or tok


_LEGAL_SKEL = {skeleton(t) for t in _LEGAL | {"private", "limited", "incorporated", "corporation", "company"}
               if len(skeleton(t)) >= 3} | {"prvt", "prv", "prvr", "lmtd", "lmrd", "lmtt", "lmt"}


def name_skeleton(name_core: str) -> str:
    return " ".join(k for k in (skeleton(t) for t in name_core.split()) if k and k not in _LEGAL_SKEL)


def load_skeleton(split: str, source: str) -> pd.Series:
    """name_skel column for a source (cached)."""
    from .config import CACHE_DIR
    pq = CACHE_DIR / f"skel_{NORM_VERSION}_{split}_{source}.parquet"
    if not pq.exists():
        core = load_normalized(split, source, columns=["name_core"])["name_core"].tolist()
        pd.DataFrame({"name_skel": [name_skeleton(x) for x in core]}).astype("string[pyarrow]").to_parquet(pq, index=False)
    return pd.read_parquet(pq, dtype_backend="pyarrow")["name_skel"]


# ---------------------------------------------------------------- glued / domain-style names (v4)
# 'myagrocom' ~ 'my agro', '0rthopedicsafecentercom' ~ 'orthopedic safe center', 'bi0' ~ 'bio'
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def name_concat(name_core: str) -> str:
    toks = []
    for t in name_core.split():
        if t.isdigit():
            if len(t) >= 7:            # phone numbers / registration ids appended to names
                continue
        elif any(ch.isdigit() for ch in t):
            t = t.translate(_LEET)       # letters+digits mixed -> look-alike letters
        toks.append(t)
    s = "".join(toks)
    if s.startswith("www") and len(s) > 6:
        s = s[3:]
    if s.endswith("com") and len(s) > 6:
        s = s[:-3]
    return s


def load_concat(split: str, source: str) -> pd.Series:
    """name_concat column for a source (cached)."""
    from .config import CACHE_DIR
    pq = CACHE_DIR / f"concat_{NORM_VERSION}_{split}_{source}.parquet"
    if not pq.exists():
        core = load_normalized(split, source, columns=["name_core"])["name_core"].tolist()
        pd.DataFrame({"name_concat": [name_concat(x) for x in core]}).astype("string[pyarrow]").to_parquet(pq, index=False)
    return pd.read_parquet(pq, dtype_backend="pyarrow")["name_concat"]


# ---------------------------------------------------------------- address views (v9)
def address_views(addr_norm: str):
    """(phonetic skeleton of the address, address with landmark phrases removed).
    Landmark phrases ('near fortis hospital', 'opp rta office') differ between sources, so the
    word after a landmark marker and the next one are dropped too."""
    toks = addr_norm.split()
    skel = " ".join(skeleton(t) for t in toks)
    core, skip = [], 0
    for t in toks:
        if t in LANDMARK_WORDS:
            skip = 2
            continue
        if skip:
            skip -= 1
            continue
        core.append(t)
    return skel, " ".join(core)


def load_addr_views(split: str, source: str) -> pd.DataFrame:
    """addr_skel / addr_core columns for a source (cached)."""
    from .config import CACHE_DIR
    pq = CACHE_DIR / f"addrviews_{NORM_VERSION}_{split}_{source}.parquet"
    if not pq.exists():
        addrs = load_normalized(split, source, columns=["addr_norm"])["addr_norm"].tolist()
        pd.DataFrame([address_views(a) for a in addrs], columns=["addr_skel", "addr_core"]) \
          .astype("string[pyarrow]").to_parquet(pq, index=False)
    return pd.read_parquet(pq, dtype_backend="pyarrow")


# ---------------------------------------------------------------- v9: French text fixes
# French elision: "L'Atelier" -> "Atelier", "rue de l'Eglise" -> "rue de Eglise". Only before a
# vowel or h, so names like "D'Souza" / "D'Costa" (consonant after the apostrophe) are untouched.
_ELISION = re.compile(r"\b[lLdD]['\u2019`]\s*(?=[aeiouhyAEIOUHY\u00C0-\u00FF])")
_FR_ADDR = {"chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte", "quai": "qu",
            "saint": "st", "sainte": "ste", "cedex": "", "bis": ""}


def elide(s: str) -> str:
    return _ELISION.sub("", s) if isinstance(s, str) else ""


def normalize_address_fr(raw: str):
    a_norm, _, nums = normalize_address(elide(raw))
    toks = [t for t in (_FR_ADDR.get(t, t) for t in a_norm.split()) if t]
    return " ".join(toks), " ".join(sorted(set(toks))), nums


def load_fr_fixed(split: str, source: str) -> pd.DataFrame:
    """name_core / name_key / addr_norm / addr_bag recomputed with the v9 French fixes (cached).
    Used for pair FEATURES only; blocking keeps its cached inputs."""
    from .config import CACHE_DIR
    from .io_utils import load_source
    pq = CACHE_DIR / f"frfix_v9_{split}_{source}.parquet"
    if not pq.exists():
        raw = load_source(split, source, columns=["business_name", "business_address"])
        names = [normalize_name(elide(x)) for x in raw["business_name"].tolist()]
        addrs = [normalize_address_fr(x) for x in raw["business_address"].tolist()]
        pd.DataFrame({"name_core": [n[1] for n in names], "name_key": [n[2] for n in names],
                      "addr_norm": [a[0] for a in addrs], "addr_bag": [a[1] for a in addrs]}
                     ).astype("string[pyarrow]").to_parquet(pq, index=False)
    return pd.read_parquet(pq, dtype_backend="pyarrow")