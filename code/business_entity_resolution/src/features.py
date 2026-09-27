"""Pair features for the matching model. All string similarities use rapidfuzz.process.cpdist
(C++ and multithreaded), computed in chunks so memory stays bounded.

Feature groups
  name:     ratio / token-sort / token-set / partial / Jaro-Winkler on the core name,
            token-set on the full name, DBA cross-match, exact-key flag
  address:  token-set / partial on normalized address, ratio on the order-free token bag
  numbers:  shared digit groups, Jaccard, conflict flag, suffix match (handles 1025 vs 025)
  meta:     country equal, candidate source, name lengths, blocking keys shared
  context:  rank / gap of this pair among all S1s competing for the same record and among all
            records proposed for the same S1; how ambiguous the name is among Source 1
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .blocking import arrow, rank_desc, take_list

TEXT_COLS = ["country", "name_core", "name_norm", "name_dba", "name_key", "addr_norm", "addr_bag", "addr_nums", "name_skel", "name_concat"]

FEATURES = [
    "name_ratio", "name_tsort", "name_tset", "name_partial", "name_jw", "name_full_tset",
    "dba_x", "name_exact", "addr_tset", "addr_partial", "addr_bag_ratio",
    "num_shared", "num_jacc", "num_conflict", "num_suffix", "num_n_s1", "num_n_c",
    "country_eq", "is_s3", "len_s1", "len_c", "ntok_diff",
    "nkeys", "cheap", "s1_key_freq", "c_key_freq",
    "rank_in_c", "gap_in_c", "n_s1_for_c", "rank_in_s1", "gap_in_s1", "n_c_for_s1",
    "gap_name_in_c", "gap_addr_in_c",
    # v3
    "skel_ratio", "skel_tset", "first_tok_ratio",
    "hn_first", "hn_first_ed", "hn_long", "hn_long_ed", "num_only_s1", "num_only_c",
    "addr_exact", "s1_addr_freq", "c_addr_freq", "ntok_s1", "ntok_c",
    # v4
    "glue_ratio", "glue_partial", "addr_len_s1", "addr_len_c",
    # v6: similarity search
    "knn_sim", "knn_rank", "knnn_sim", "knnn_rank",
    # v9: rarity-weighted (IDF) token overlap
    "name_idf_jacc", "name_idf_shared_max", "name_idf_miss_s1", "name_idf_miss_c",
    "addr_idf_jacc", "addr_idf_miss_s1", "addr_idf_miss_c",
    # v5: sibling corroboration (other candidates proposed for the same S1)
    "sib_n", "sib_addr_eq", "sib_name_eq", "sib_best_sim", "sib_best_sim_x_cheap", "sib_best_addr",
]


# Features the model uses. Sibling-count features are computed but EXCLUDED: test has ~23% more
# records per S1 than train, so those counts shift and made v5 over-merge on the leaderboard.
MODEL_FEATURES = [f for f in FEATURES if not f.startswith("sib_")]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def _num_feats(a_list, b_list):
    out = np.zeros((len(a_list), 6), np.float32)
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            out[k] = (0, 0, 0, 0, len(sa), len(sb))
            continue
        shared = len(sa & sb)
        suffix = shared > 0 or any((x.endswith(y) or y.endswith(x)) and min(len(x), len(y)) >= 2
                                   for x in sa for y in sb)
        out[k] = (shared, shared / len(sa | sb), float(not suffix), float(suffix), len(sa), len(sb))
    return out


def _compat(x, y):
    return x == y or ((x.endswith(y) or y.endswith(x)) and min(len(x), len(y)) >= 2)


def _hn_code(x, y):
    """house-number comparison: 2 exact, 1 truncated (1025 vs 025), 0 conflict, -1 missing."""
    if not x or not y:
        return -1.0
    if x == y:
        return 2.0
    return 1.0 if _compat(x, y) else 0.0


_FIRST_NUM = re.compile(r"\d+")


def _house_feats(addr1, addr2, nums1, nums2):
    """First number in address order, and longest number; plus unmatched-number counts."""
    out = np.zeros((len(addr1), 6), np.float32)
    for k, (a1, a2, n1, n2) in enumerate(zip(addr1, addr2, nums1, nums2)):
        m1, m2 = _FIRST_NUM.search(a1), _FIRST_NUM.search(a2)
        f1, f2 = (m1.group(0) if m1 else ""), (m2.group(0) if m2 else "")
        s1, s2 = n1.split(), n2.split()
        l1, l2 = (max(s1, key=len) if s1 else ""), (max(s2, key=len) if s2 else "")
        only1 = sum(not any(_compat(x, y) for y in s2) for x in s1) if s2 else -1
        only2 = sum(not any(_compat(y, x) for x in s1) for y in s2) if s1 else -1
        out[k] = (_hn_code(f1, f2), Levenshtein.normalized_distance(f1, f2) if f1 and f2 else -1,
                  _hn_code(l1, l2), Levenshtein.normalized_distance(l1, l2) if l1 and l2 else -1,
                  only1, only2)
    return out


def idf_tables(frames, name_col="name_core", addr_col="addr_bag"):
    """Document frequency of name / address tokens over the given frames -> normalized IDF in
    [0, 1] (log((N+1)/(df+1)) / log(N+1)), so values are comparable across dataset sizes."""
    from collections import Counter
    nc, ac, n = Counter(), Counter(), 0
    for df in frames:
        n += len(df)
        for x in df[name_col].tolist():
            nc.update(set(x.split()))
        for x in df[addr_col].tolist():
            ac.update(x.split())
    ln = np.log(n + 1)
    to_idf = lambda c: {t: float(np.log((n + 1) / (v + 1)) / ln) for t, v in c.items()}
    return to_idf(nc), to_idf(ac)


def _idf_feats(a_list, b_list, idf, with_shared_max=True):
    """Rarity-weighted overlap: sharing a rare token is strong evidence, and a rare token present
    on only one side ('stephanie' vs 'spencer') is strong evidence against."""
    out = np.full((len(a_list), 4), -1, np.float32)
    get = idf.get
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            continue
        inter, union = sa & sb, sa | sb
        w_u = sum(get(t, 1.0) for t in union)
        w_i = sum(get(t, 1.0) for t in inter)
        out[k, 0] = w_i / w_u if w_u else 0
        out[k, 1] = max((get(t, 1.0) for t in inter), default=0.0)
        out[k, 2] = max((get(t, 1.0) for t in sa - sb), default=0.0)
        out[k, 3] = max((get(t, 1.0) for t in sb - sa), default=0.0)
    return out


def _group_best_gap(group, value):
    best = pd.Series(value).groupby(group).transform("max").to_numpy()
    return best - value


def sibling_features(cands: dict, candn: pd.DataFrame, chunk: int = 2_000_000, verbose: bool = True):
    """For each (S1, record) pair, compare the record with the OTHER records proposed for the same
    S1. True matches of one entity (1-3 in S2 + 1-3 in S3) corroborate each other: a renamed
    record at the same address as a strong sibling is probably a match too.
    Uses only strings and blocking scores (no model outputs), so there is no train/test leakage."""
    s1, cc, cheap = cands["s1"], cands["cand"], cands["cheap"]
    n = len(s1)
    order = np.argsort(s1, kind="stable")
    g = s1[order]
    name, addr = arrow(candn["name_core"]), arrow(candn["addr_norm"])
    bag, key = arrow(candn["addr_bag"]), arrow(candn["name_key"])
    out = {f: np.zeros(n, np.float32) for f in ["sib_addr_eq", "sib_name_eq", "sib_best_sim",
                                                 "sib_best_sim_x_cheap", "sib_best_addr"]}
    size = np.bincount(s1, minlength=int(s1.max()) + 1 if n else 1)
    out["sib_n"] = (size[s1] - 1).astype(np.float32)
    max_k = int(size.max()) if n else 0
    for k in range(1, max_k):
        i = np.flatnonzero(g[:-k] == g[k:])           # sorted positions i and i+k in the same S1
        if len(i) == 0:
            break
        a_idx, b_idx = order[i], order[i + k]           # pair rows
        for b in range(0, len(a_idx), chunk):
            ia, ib = a_idx[b:b + chunk], b_idx[b:b + chunk]
            ca, cb = cc[ia], cc[ib]
            nm = _cp(take_list(name, ca), take_list(name, cb), fuzz.token_set_ratio)
            ad = _cp(take_list(addr, ca), take_list(addr, cb), fuzz.token_set_ratio)
            ba, bb = take_list(bag, ca), take_list(bag, cb)
            aeq = np.fromiter((x == y and x != "" for x, y in zip(ba, bb)), np.float32, len(ba))
            ka, kb = take_list(key, ca), take_list(key, cb)
            neq = np.fromiter((x == y and x != "" for x, y in zip(ka, kb)), np.float32, len(ka))
            sim = 0.5 * (nm + ad)
            for me, other in ((ia, ib), (ib, ia)):       # symmetric: update both endpoints
                np.add.at(out["sib_addr_eq"], me, aeq)
                np.add.at(out["sib_name_eq"], me, neq)
                np.maximum.at(out["sib_best_sim"], me, sim)
                np.maximum.at(out["sib_best_sim_x_cheap"], me, sim * cheap[other] / 100)
                np.maximum.at(out["sib_best_addr"], me, ad)
        if verbose:
            print(f"  sibling offset {k}: {len(i):,} pairs", end="\r")
    if verbose:
        print()
    return out


def compute_features(cands: dict, s1n: pd.DataFrame, candn: pd.DataFrame,
                     chunk: int = 1_000_000, verbose: bool = True, with_siblings: bool = False,
                     idf: tuple = None) -> pd.DataFrame:
    s1c, cc = cands["s1"], cands["cand"]
    n = len(s1c)
    A = {c: arrow(s1n[c]) for c in TEXT_COLS}
    B = {c: arrow(candn[c]) for c in TEXT_COLS}
    F = {f: np.zeros(n, np.float32) for f in FEATURES}

    for b in range(0, n, chunk):
        i, j, sl = s1c[b:b + chunk], cc[b:b + chunk], slice(b, b + chunk)
        g = {c: (take_list(A[c], i), take_list(B[c], j)) for c in TEXT_COLS}
        core1, core2 = g["name_core"]
        F["name_ratio"][sl] = _cp(core1, core2, fuzz.ratio)
        F["name_tsort"][sl] = _cp(*g["name_key"], fuzz.ratio)
        F["name_tset"][sl] = _cp(core1, core2, fuzz.token_set_ratio)
        F["name_partial"][sl] = _cp(core1, core2, fuzz.partial_ratio)
        F["name_jw"][sl] = 100 * _cp(core1, core2, JaroWinkler.normalized_similarity)
        F["name_full_tset"][sl] = _cp(*g["name_norm"], fuzz.token_set_ratio)
        d1, d2 = g["name_dba"]
        has = np.array([bool(x or y) for x, y in zip(d1, d2)])
        dx = np.maximum.reduce([_cp(core1, d2, fuzz.token_set_ratio), _cp(d1, core2, fuzz.token_set_ratio),
                                _cp(d1, d2, fuzz.token_set_ratio)])
        F["dba_x"][sl] = np.where(has, dx, -1)
        k1, k2 = g["name_key"]
        F["name_exact"][sl] = np.fromiter((x == y for x, y in zip(k1, k2)), np.float32, len(k1))
        F["addr_tset"][sl] = _cp(*g["addr_norm"], fuzz.token_set_ratio)
        F["addr_partial"][sl] = _cp(*g["addr_norm"], fuzz.partial_ratio)
        F["addr_bag_ratio"][sl] = cands["addr_sim"][sl]
        nf = _num_feats(*g["addr_nums"])
        for col, f in enumerate(["num_shared", "num_jacc", "num_conflict", "num_suffix", "num_n_s1", "num_n_c"]):
            F[f][sl] = nf[:, col]
        c1, c2 = g["country"]
        F["country_eq"][sl] = np.fromiter((x == y for x, y in zip(c1, c2)), np.float32, len(c1))
        F["len_s1"][sl] = np.fromiter(map(len, core1), np.float32, len(core1))
        F["len_c"][sl] = np.fromiter(map(len, core2), np.float32, len(core2))
        F["ntok_diff"][sl] = np.fromiter((abs(x.count(" ") - y.count(" ")) for x, y in zip(core1, core2)),
                                         np.float32, len(core1))
        # v3: phonetic, first word, house numbers, sizes
        F["skel_ratio"][sl] = _cp(*g["name_skel"], fuzz.ratio)
        F["skel_tset"][sl] = _cp(*g["name_skel"], fuzz.token_set_ratio)
        F["first_tok_ratio"][sl] = _cp([x.split(" ", 1)[0] for x in core1],
                                       [y.split(" ", 1)[0] for y in core2], fuzz.ratio)
        hf = _house_feats(*g["addr_norm"], *g["addr_nums"])
        for col, f in enumerate(["hn_first", "hn_first_ed", "hn_long", "hn_long_ed", "num_only_s1", "num_only_c"]):
            F[f][sl] = hf[:, col]
        b1, b2 = g["addr_bag"]
        F["addr_exact"][sl] = np.fromiter((x == y and x != "" for x, y in zip(b1, b2)), np.float32, len(b1))
        F["ntok_s1"][sl] = np.fromiter((x.count(" ") + 1 if x else 0 for x in core1), np.float32, len(core1))
        F["ntok_c"][sl] = np.fromiter((y.count(" ") + 1 if y else 0 for y in core2), np.float32, len(core2))
        # v9: rarity-weighted overlap
        if idf is not None:
            nf_ = _idf_feats(core1, core2, idf[0])
            for col, f in enumerate(["name_idf_jacc", "name_idf_shared_max", "name_idf_miss_s1", "name_idf_miss_c"]):
                F[f][sl] = nf_[:, col]
            af_ = _idf_feats(*g["addr_bag"], idf[1])
            for col, f in [(0, "addr_idf_jacc"), (2, "addr_idf_miss_s1"), (3, "addr_idf_miss_c")]:
                F[f][sl] = af_[:, col]
        # v4: glued names, address length (0 = empty address)
        F["glue_ratio"][sl] = _cp(*g["name_concat"], fuzz.ratio)
        F["glue_partial"][sl] = _cp(*g["name_concat"], fuzz.partial_ratio)
        a1, a2 = g["addr_norm"]
        F["addr_len_s1"][sl] = np.fromiter(map(len, a1), np.float32, len(a1))
        F["addr_len_c"][sl] = np.fromiter(map(len, a2), np.float32, len(a2))
        if verbose:
            print(f"  features {min(b + chunk, n):,}/{n:,}", end="\r")
    if verbose:
        print()

    F["is_s3"][:] = candn["entity_id"].str.startswith("S3").to_numpy(dtype=bool)[cc]
    F["nkeys"][:] = cands["nkeys"]
    F["knn_sim"][:] = cands.get("knn_sim", np.full(n, -1, np.float32))
    F["knn_rank"][:] = cands.get("knn_rank", np.full(n, -1, np.int8))
    F["knnn_sim"][:] = cands.get("knnn_sim", np.full(n, -1, np.float32))
    F["knnn_rank"][:] = cands.get("knnn_rank", np.full(n, -1, np.int8))
    F["cheap"][:] = cands["cheap"]

    # how ambiguous is the name among Source 1? (50% of S1 names are shared with another S1)
    s1_freq = s1n.groupby(["country", "name_key"])["entity_id"].transform("size").to_numpy()
    F["s1_key_freq"][:] = s1_freq[s1c]
    key_count = s1n.groupby(["country", "name_key"]).size()
    c_idx = pd.MultiIndex.from_arrays([candn["country"], candn["name_key"]])
    F["c_key_freq"][:] = key_count.reindex(c_idx).fillna(0).to_numpy()[cc]

    # how many S1 entities live at this exact address? (unique address + renamed business)
    a_freq = s1n.groupby(["country", "addr_bag"])["entity_id"].transform("size").to_numpy()
    a_freq = np.where(s1n["addr_bag"].to_numpy() == "", -1, a_freq)
    F["s1_addr_freq"][:] = a_freq[s1c]
    a_count = s1n.groupby(["country", "addr_bag"]).size()
    ca_idx = pd.MultiIndex.from_arrays([candn["country"], candn["addr_bag"]])
    ca = a_count.reindex(ca_idx).fillna(0).to_numpy()
    F["c_addr_freq"][:] = np.where(candn["addr_bag"].to_numpy() == "", -1, ca)[cc]

    # competition context
    F["rank_in_c"][:] = rank_desc(cc, cands["cheap"])
    F["gap_in_c"][:] = _group_best_gap(cc, cands["cheap"])
    F["n_s1_for_c"][:] = np.bincount(cc, minlength=len(candn))[cc]
    F["rank_in_s1"][:] = rank_desc(s1c, cands["cheap"])
    F["gap_in_s1"][:] = _group_best_gap(s1c, cands["cheap"])
    F["n_c_for_s1"][:] = np.bincount(s1c, minlength=len(s1n))[s1c]
    F["gap_name_in_c"][:] = _group_best_gap(cc, F["name_tset"])
    F["gap_addr_in_c"][:] = _group_best_gap(cc, F["addr_tset"])
    if with_siblings:                      # excluded from the model (train/test density shift)
        for f, v in sibling_features(cands, candn, verbose=verbose).items():
            F[f][:] = v
    return pd.DataFrame(F)