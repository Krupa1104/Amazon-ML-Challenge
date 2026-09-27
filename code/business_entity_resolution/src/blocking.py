"""Candidate generation (blocking) that scales to millions of records on a 16 GB laptop.

Idea
----
Every record gets up to K integer "blocking keys" (hashed, so no big string tables):
  N1, N2      its 2 rarest name tokens              -> catches address changes
  NN          the 2 rarest name tokens together      -> 'anand + jain': common words, rare pair
  P1, P2, PP  same on the phonetic skeleton          -> 'inphra'~'infra', 'gujrat'~'gujarat'
  A1, A2      its 2 rarest address words             -> catches renamed businesses / scripts
  N1A1, N2A1, N1A2, P1A1   name x address combos     -> precise keys for generic names/streets
  FULL        the whole normalized name (sorted tokens) -> unique names with empty addresses
  GLUE        first 6 letters of the glued name       -> 'myagrocom' ~ 'my agro', '0rtho' ~ 'ortho'
  A1#         rarest address word + last 2 digits of the house number
              (robust to the dropped-leading-digit noise: 1025 -> 025)
The key family is stored in the low 4 bits, so each family has its own block-size cap.
Tokens that occur only once in the whole dataset are useless as keys (nothing to match), so
"rarest" means rarest among tokens seen at least twice. Keys whose block would produce more
than its family's cap are dropped (the more specific families still cover those records).

S1 x candidate pairs sharing any key are generated with a vectorized sorted join, in
partitions of S1 rows so memory stays bounded. Each raw pair gets a cheap score (fuzzy ratio
of sorted name tokens + address token bag); we keep the top `per_s1` per S1 (under a name-heavy and an
address-heavy ranking), then the top `per_cand` S1 entities per candidate record (a record has at most one true owner).

Everything works on integer row codes: s1 code = row in s1n, cand code = row in candn.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz, process

SENT = np.iinfo(np.int64).min
K = 20
ADDR_KEY_STOP = {
    "st", "rd", "ave", "dr", "blvd", "ln", "cir", "ct", "pl", "hwy", "pkwy", "sq", "ste", "apt",
    "bldg", "fl", "unit", "near", "opp", "behind", "flat", "plot", "pobox", "city", "village",
    "of", "the", "and", "sec", "ngr", "floor", "tower", "block", "phase", "main", "cross",
    "road", "street", "suite", "rue", "de", "du", "des", "la", "le", "les", "careof", "town",
}


def arrow(col: pd.Series) -> pa.Array:
    """pandas string column -> contiguous pyarrow array (fast .take())."""
    a = pa.array(col)
    return a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a


def take_list(arr: pa.Array, idx: np.ndarray) -> list:
    return arr.take(pa.array(idx)).to_pylist()


def cheap_score(name_sim, addr_sim):
    """max of a name-heavy and an address-heavy blend (0-100)."""
    return np.maximum(0.7 * name_sim + 0.3 * addr_sim, 0.3 * name_sim + 0.7 * addr_sim).astype(np.float32)


def rank_desc(group: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each element within its group, by descending score."""
    order = np.lexsort((-score, group))
    g = group[order]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(g)]))
    ranks = np.empty(len(g), dtype=np.int32)
    ranks[order] = np.arange(len(g)) - first
    return ranks


# ------------------------------------------------------------------ keys
# key type is stored in the low 4 bits so each family can have its own block-size cap
T_N, T_A, T_NA, T_ANUM, T_NN, T_P, T_PP, T_PA, T_FULL, T_GLUE = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
KEY_NAMES = {T_N: "name token", T_A: "address word", T_NA: "name x address", T_ANUM: "address+number",
             T_NN: "2 name tokens", T_P: "sound token", T_PP: "2 sound tokens", T_PA: "sound x address",
             T_FULL: "full name", T_GLUE: "glued name[:6]"}
DEFAULT_CAPS = {T_N: 20000, T_A: 5000, T_NA: 5000, T_ANUM: 5000, T_NN: 5000,
                T_P: 20000, T_PP: 5000, T_PA: 5000, T_FULL: 5000, T_GLUE: 5000}


def _h(t, *parts):
    return (hash(parts) & ~15) | t


def _token_counts(frames):
    name_c, addr_c, skel_c = Counter(), Counter(), Counter()
    for df in frames:
        for core, dba in zip(df["name_core"].tolist(), df["name_dba"].tolist()):
            name_c.update(set((core + " " + dba).split()))
        for bag in df["addr_bag"].tolist():
            addr_c.update(bag.split())
        for sk in df["name_skel"].tolist():
            skel_c.update(set(sk.split()))
    return name_c, addr_c, skel_c


def _rarest(tokens, counts, n, stop=()):
    toks = [t for t in tokens if counts.get(t, 0) >= 2 and t not in stop]
    toks.sort(key=lambda t: (counts[t], t))
    return toks[:n]


def build_keys(df: pd.DataFrame, name_c: Counter, addr_c: Counter, skel_c: Counter) -> np.ndarray:
    keys = np.full((len(df), K), SENT, dtype=np.int64)
    cols = [df[c].tolist() for c in ("country", "name_core", "name_dba", "addr_bag", "addr_nums", "name_skel",
                                     "name_key", "name_concat")]
    for i, (cty, core, dba, bag, nums, skel, nkey, glue) in enumerate(zip(*cols)):
        n = _rarest(set((core + " " + dba).split()), name_c, 2)
        a4 = _rarest([t for t in bag.split() if len(t) >= 3 and not t.isdigit()], addr_c, 4, ADDR_KEY_STOP)
        a = a4[:2]
        p = _rarest([t for t in set(skel.split()) if len(t) >= 2], skel_c, 2)
        ks = [_h(T_N, cty, t) for t in n] + [_h(T_A, cty, t) for t in a] + [_h(T_P, cty, t) for t in p]
        if nkey:
            ks.append(_h(T_FULL, cty, nkey))
        if len(glue) >= 4:
            ks.append(_h(T_GLUE, cty, glue[:6]))
        if len(n) > 1:
            ks.append(_h(T_NN, cty, *sorted(n)))
        if len(p) > 1:
            ks.append(_h(T_PP, cty, *sorted(p)))
        if a:
            if n:
                ks.append(_h(T_NA, cty, n[0], a[0]))
                if len(n) > 1:
                    ks.append(_h(T_NA, cty, n[1], a[0]))
                if len(a) > 1:
                    ks.append(_h(T_NA, cty, n[0], a[1]))
            if p:
                ks.append(_h(T_PA, cty, p[0], a[0]))
                if len(a) > 1:
                    ks.append(_h(T_PA, cty, p[0], a[1]))
            # v5: rarest name / sound token x 3rd-4th rarest address words, so a shared word
            # that is not among the 2 rarest on both sides still meets (bhiwandi case)
            for extra in a4[2:]:
                if n:
                    ks.append(_h(T_NA, cty, n[0], extra))
                if p:
                    ks.append(_h(T_PA, cty, p[0], extra))
            if nums:
                longest = max(nums.split(), key=len)
                ks.append(_h(T_ANUM, cty, a[0], longest[-2:]))
        keys[i, :len(ks)] = ks[:K]
    return keys


def _flatten(keys: np.ndarray):
    rows = np.repeat(np.arange(keys.shape[0], dtype=np.int32), keys.shape[1])
    flat = keys.ravel()
    ok = flat != SENT
    return flat[ok], rows[ok]


# ------------------------------------------------------------------ main entry
def generate_candidates(s1n: pd.DataFrame, candn: pd.DataFrame, caps: dict = None,
                        per_s1: int = 30, per_cand: int = 4, min_cheap: float = 30.0,
                        part_pairs: int = 15_000_000, batch: int = 2_000_000, verbose: bool = True,
                        knn: dict = None, knn_keep_top: int = 2):
    """Returns dict of numpy arrays: s1, cand (int codes), nkeys, name_sim, addr_sim, cheap,
    knn_sim, knn_rank (-1 when the pair was not found by similarity search).
    knn: optional output of retrieval.knn_candidates, merged with the key-based pairs; the
    nearest `knn_keep_top` neighbours of every record always survive the per-record cut."""
    log = print if verbose else (lambda *a, **k: None)
    log("counting tokens ...")
    caps = {**DEFAULT_CAPS, **(caps or {})}
    counts = _token_counts([s1n, candn])
    log("building keys ...")
    k1, r1 = _flatten(build_keys(s1n, *counts))
    kc, rc = _flatten(build_keys(candn, *counts))
    del counts

    # block sizes -> drop keys that would explode
    u1, n1 = np.unique(k1, return_counts=True)
    uc, nc = np.unique(kc, return_counts=True)
    common, i1, ic = np.intersect1d(u1, uc, assume_unique=True, return_indices=True)
    size = n1[i1].astype(np.int64) * nc[ic]
    ktype = common & 15
    lut = np.zeros(16, np.int64)
    for t, v in caps.items():
        lut[int(t)] = v
    cap = lut[ktype]
    ok = size <= cap
    keep = common[ok]
    m1, mc = np.isin(k1, keep), np.isin(kc, keep)
    log(f"keys: {len(u1):,} S1-side, {len(uc):,} cand-side, {len(common):,} shared, {len(keep):,} kept")
    log("  raw pairs by key family (kept / dropped by cap):")
    for t, name in KEY_NAMES.items():
        sel = ktype == t
        log(f"    {name:16} {int(size[sel & ok].sum()):>14,} / {int(size[sel & ~ok].sum()):>16,}  (cap {caps[t]:,})")
    k1, r1, kc, rc = k1[m1], r1[m1], kc[mc], rc[mc]

    order = np.argsort(kc, kind="stable")
    kc, rc = kc[order], rc[order]
    lo = np.searchsorted(kc, k1, "left")
    cnt = np.searchsorted(kc, k1, "right") - lo
    total = int(cnt.sum())
    P = max(1, math.ceil(total / part_pairs))
    log(f"raw S1-key x cand-key hits: {total:,} -> {P} partition(s)")

    NC = len(candn)
    s1_key, c_key = arrow(s1n["name_key"]), arrow(candn["name_key"])
    s1_bag, c_bag = arrow(s1n["addr_bag"]), arrow(candn["addr_bag"])
    out = []
    for p in range(P):
        sel = (r1 % P) == p
        n = cnt[sel]
        if n.sum() == 0:
            continue
        s1_rep = np.repeat(r1[sel], n)
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        c_rep = rc[np.repeat(lo[sel], n) + offs]
        pc, nkeys = np.unique(s1_rep.astype(np.int64) * NC + c_rep, return_counts=True)
        del s1_rep, c_rep, offs
        s1c, cc = pc // NC, pc % NC
        name_sim = np.empty(len(pc), np.float32)
        addr_sim = np.empty(len(pc), np.float32)
        for b in range(0, len(pc), batch):
            i, j = s1c[b:b + batch], cc[b:b + batch]
            name_sim[b:b + batch] = process.cpdist(take_list(s1_key, i), take_list(c_key, j),
                                                   scorer=fuzz.ratio, workers=-1)
            addr_sim[b:b + batch] = process.cpdist(take_list(s1_bag, i), take_list(c_bag, j),
                                                   scorer=fuzz.ratio, workers=-1)
        cheap = cheap_score(name_sim, addr_sim)
        ok = cheap >= min_cheap
        s1c, cc, nkeys, name_sim, addr_sim, cheap = s1c[ok], cc[ok], nkeys[ok], name_sim[ok], addr_sim[ok], cheap[ok]
        # keep the best per S1 under a name-heavy AND an address-heavy ranking, so a renamed
        # business at an identical address is not crowded out by same-name look-alikes
        ok = ((rank_desc(s1c, 0.7 * name_sim + 0.3 * addr_sim) < per_s1) |
              (rank_desc(s1c, 0.3 * name_sim + 0.7 * addr_sim) < per_s1))
        out.append((s1c[ok], cc[ok], nkeys[ok].astype(np.int16), name_sim[ok], addr_sim[ok], cheap[ok]))
        log(f"  partition {p + 1}/{P}: {len(pc):,} unique pairs -> kept {ok.sum():,}")

    s1c, cc, nkeys, name_sim, addr_sim, cheap = (np.concatenate(x) for x in zip(*out))
    extra = [k for k in (knn or {}) if k not in ("s1", "cand")]
    ext = {k: np.full(len(s1c), -1, knn[k].dtype) for k in extra}
    if knn is not None and len(knn["s1"]):
        pc_key = s1c.astype(np.int64) * NC + cc
        pc_knn = knn["s1"].astype(np.int64) * NC + knn["cand"]
        o = np.argsort(pc_knn)
        pos = np.minimum(np.searchsorted(pc_knn[o], pc_key), len(o) - 1)
        hit = pc_knn[o][pos] == pc_key                    # key pairs also found by kNN
        src = o[pos[hit]]
        for k in extra:
            ext[k][hit] = knn[k][src]
        del src
        new = ~np.isin(pc_knn, pc_key)                    # pairs ONLY similarity search found
        del o, pos, pc_key
        ns1, nc_ = knn["s1"][new], knn["cand"][new]
        n_name = np.empty(len(ns1), np.float32)
        n_addr = np.empty(len(ns1), np.float32)
        for b in range(0, len(ns1), batch):
            i, j = ns1[b:b + batch], nc_[b:b + batch]
            n_name[b:b + batch] = process.cpdist(take_list(s1_key, i), take_list(c_key, j), scorer=fuzz.ratio, workers=-1)
            n_addr[b:b + batch] = process.cpdist(take_list(s1_bag, i), take_list(c_bag, j), scorer=fuzz.ratio, workers=-1)
        log(f"similarity search: {len(pc_knn):,} pairs, {hit.sum():,} already from keys, {new.sum():,} new")
        s1c = np.concatenate([s1c, ns1.astype(np.int64)]); cc = np.concatenate([cc, nc_.astype(np.int64)])
        nkeys = np.concatenate([nkeys, np.zeros(len(ns1), nkeys.dtype)])
        name_sim = np.concatenate([name_sim, n_name]); addr_sim = np.concatenate([addr_sim, n_addr])
        cheap = np.concatenate([cheap, cheap_score(n_name, n_addr)])
        for k in extra:
            ext[k] = np.concatenate([ext[k], knn[k][new]])
    top = np.zeros(len(s1c), bool)
    for k in extra:
        if k.endswith("_rank"):
            top |= (ext[k] >= 0) & (ext[k] < knn_keep_top)
    ok = (rank_desc(cc, cheap) < per_cand) | top
    res = dict(s1=s1c[ok].astype(np.int64), cand=cc[ok].astype(np.int64), nkeys=nkeys[ok],
               name_sim=name_sim[ok], addr_sim=addr_sim[ok], cheap=cheap[ok],
               **{k: v[ok] for k, v in ext.items()})
    log(f"final candidate pairs: {len(res['s1']):,} "
        f"({len(res['s1']) / max(len(s1n), 1):.1f} per S1)")
    return res


# ------------------------------------------------------------------ evaluation helpers
def gt_codes(gt_pairs: pd.DataFrame, s1_ids, cand_ids):
    """Ground truth as int codes: returns (gt_s1 code, gt pair code)."""
    s1 = pd.Index(s1_ids).get_indexer(gt_pairs["s1_id"])
    c = pd.Index(cand_ids).get_indexer(gt_pairs["cand_id"])
    ok = (s1 >= 0) & (c >= 0)
    return s1[ok].astype(np.int64), s1[ok].astype(np.int64) * len(cand_ids) + c[ok]


def blocking_recall(cands: dict, gt_s1, gt_pc, n_cand: int, eval_mask: np.ndarray) -> float:
    """Share of true pairs (for S1s in eval_mask) that survive blocking = recall ceiling."""
    pc = cands["s1"].astype(np.int64) * n_cand + cands["cand"]
    sel = eval_mask[gt_s1]
    rec = np.isin(gt_pc[sel], pc).mean()
    per = np.bincount(cands["s1"], minlength=len(eval_mask))[eval_mask]
    print(f"[blocking] pair recall {rec:.4f} | candidates per S1: mean {per.mean():.1f}, "
          f"p95 {np.percentile(per, 95):.0f} | S1 with no candidates {np.mean(per == 0):.3f}")
    return rec


def oracle_ceiling(cands: dict, gt_s1, gt_pc, n_s1: int, n_cand: int, eval_mask, label=""):
    """Best possible macro F0.5 if the model were perfect on these candidates."""
    from .metrics import macro_f05_codes
    pc = cands["s1"] * n_cand + cands["cand"]
    hit = np.isin(pc, gt_pc)
    sc = macro_f05_codes(cands["s1"][hit], pc[hit], gt_s1, gt_pc, n_s1, eval_mask)
    print(f"[oracle{(' ' + label) if label else ''}] best achievable VAL macro F0.5 with these candidates: {sc:.4f}")
    return sc
