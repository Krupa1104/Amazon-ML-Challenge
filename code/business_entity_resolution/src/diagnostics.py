"""Error analysis tools.

blocking_miss_report: for true validation pairs that never reached the model, find out WHY:
  no_shared_key   the two records share none of the 8 blocking keys       -> key design
  capped_only     they share a key, but every shared key's block was too big (max_block)
  cheap_floor     they share a usable key, but the cheap score < min_cheap
  per_cand_cut    the candidate record already had `per_cand` S1s with higher cheap scores
  per_s1_cut      otherwise: cut by the per-S1 limit
model_error_report: among pairs that DID reach the model, what the model got wrong.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .blocking import DEFAULT_CAPS, SENT, cheap_score, _token_counts, arrow, build_keys, take_list


def _show(s1n, candn, s1c, cc, extra=None, n=10, title=""):
    print(f"\n--- {title} ---")
    A = s1n.iloc[s1c[:n]]
    B = candn.iloc[cc[:n]]
    for k in range(min(n, len(s1c))):
        a, b = A.iloc[k], B.iloc[k]
        tag = "" if extra is None else f"  [{extra[k]}]"
        print(f"S1 [{a.country}] {a.name_norm} | {a.addr_norm}\n"
              f"{b.entity_id[:2]} [{b.country}] {b.name_norm} | {b.addr_norm}{tag}\n")


def blocking_miss_report(s1n, candn, cands, gt_s1, gt_pc, val_mask, caps=None,
                         per_cand=3, min_cheap=30.0, sample=50_000, n_examples=8, seed=0):
    NC = len(candn)
    pc = cands["s1"] * NC + cands["cand"]
    vpc = gt_pc[val_mask[gt_s1]]
    missed = vpc[~np.isin(vpc, pc)]
    print(f"missed validation true pairs: {len(missed):,} of {len(vpc):,} ({len(missed) / len(vpc):.3f})")
    if len(missed) == 0:
        print("nothing missed - blocking recall is 100% on validation")
        return np.array([], dtype=object)
    rng = np.random.default_rng(seed)
    if len(missed) > sample:
        missed = rng.choice(missed, sample, replace=False)
    ms1, mc = missed // NC, missed % NC

    print("rebuilding keys (a few minutes) ...")
    caps = {**DEFAULT_CAPS, **(caps or {})}
    counts = _token_counts([s1n, candn])
    K1, KC = build_keys(s1n, *counts), build_keys(candn, *counts)
    del counts
    u1, n1 = np.unique(K1[K1 != SENT], return_counts=True)
    uc, nc = np.unique(KC[KC != SENT], return_counts=True)

    def size(keys):
        a = np.searchsorted(u1, keys); a = np.minimum(a, len(u1) - 1)
        b = np.searchsorted(uc, keys); b = np.minimum(b, len(uc) - 1)
        s = np.where(u1[a] == keys, n1[a], 0).astype(np.int64) * np.where(uc[b] == keys, nc[b], 0)
        return s

    k1, kc = K1[ms1], KC[mc]
    eq = (k1[:, :, None] == kc[:, None, :]) & (k1[:, :, None] != SENT)
    shared_any = eq.any(axis=(1, 2))
    shared_vals = np.where(eq.any(axis=2), k1, SENT)                 # (n, K)
    sizes = np.where(shared_vals != SENT, size(shared_vals.ravel()).reshape(shared_vals.shape), 0)
    lut = np.zeros(16, np.int64)
    for t, v in caps.items():
        lut[int(t)] = v
    cap_arr = lut[shared_vals & 15]
    usable = ((shared_vals != SENT) & (sizes <= cap_arr)).any(axis=1)

    name_sim = process.cpdist(take_list(arrow(s1n["name_key"]), ms1), take_list(arrow(candn["name_key"]), mc),
                              scorer=fuzz.ratio, workers=-1)
    addr_sim = process.cpdist(take_list(arrow(s1n["addr_bag"]), ms1), take_list(arrow(candn["addr_bag"]), mc),
                              scorer=fuzz.ratio, workers=-1)
    cheap = cheap_score(name_sim, addr_sim)

    n_for_c = np.bincount(cands["cand"], minlength=NC)
    min_for_c = pd.Series(cands["cheap"]).groupby(cands["cand"]).min().reindex(np.arange(NC)).to_numpy()
    cat = np.full(len(missed), "per_s1_cut", dtype=object)
    cat[(n_for_c[mc] >= per_cand) & (cheap <= np.nan_to_num(min_for_c[mc], nan=0))] = "per_cand_cut"
    cat[cheap < min_cheap] = "cheap_floor"
    cat[shared_any & ~usable] = "capped_only"
    cat[~shared_any] = "no_shared_key"

    print("\nwhy true pairs were missed (share of missed pairs):")
    print(pd.Series(cat).value_counts(normalize=True).round(3).to_string())
    cty = s1n["country"].to_numpy()[ms1]
    print("\nby country:\n", pd.crosstab(cty, cat, normalize="index").round(3).to_string())
    print(f"\nmedian cheap score of missed pairs: {np.median(cheap):.1f} "
          f"(name {np.median(name_sim):.1f}, address {np.median(addr_sim):.1f})")
    for c in ["no_shared_key", "capped_only", "cheap_floor", "per_cand_cut", "per_s1_cut"]:
        idx = np.flatnonzero(cat == c)[:n_examples]
        if len(idx):
            _show(s1n, candn, ms1[idx], mc[idx], [f"name {name_sim[i]:.0f} addr {addr_sim[i]:.0f}" for i in idx],
                  n=n_examples, title=c)
    return cat


def model_error_report(s1n, candn, cands, prob, keep_mask, gt_pc, val_mask, n_examples=8, seed=0):
    """keep_mask: output of model.assign(). Prints false merges and missed matches (val only)."""
    NC = len(candn)
    s1c, cc = cands["s1"], cands["cand"]
    pc = s1c * NC + cc
    is_true = np.isin(pc, gt_pc)
    v = val_mask[s1c]
    fp = np.flatnonzero(v & keep_mask & ~is_true)
    fn = np.flatnonzero(v & ~keep_mask & is_true)
    assigned_c = np.zeros(NC, bool); assigned_c[cc[keep_mask]] = True
    fn_lost = fn[assigned_c[cc[fn]]]            # record went to another S1
    fn_low = fn[~assigned_c[cc[fn]]]            # nobody got it: probability below threshold
    print(f"val false merges: {len(fp):,} | val missed (reachable) matches: {len(fn):,}"
          f"  -> below threshold {len(fn_low):,}, given to another S1 {len(fn_lost):,}")
    rng = np.random.default_rng(seed)
    for name, idx in [("FALSE MERGES (predicted, not true)", fp), ("MISSED: below threshold", fn_low),
                      ("MISSED: given to another S1", fn_lost)]:
        pick = rng.choice(idx, min(n_examples, len(idx)), replace=False) if len(idx) else idx
        _show(s1n, candn, s1c[pick], cc[pick], [f"p={prob[i]:.2f}" for i in pick], n=n_examples, title=name)
