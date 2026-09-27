"""Learned candidate filter: the last stage of candidate generation.

Blocking casts a wide net (~12-15 candidates per Source 1 entity). This small LightGBM uses
ONLY the cheap signals blocking already computed (no heavy features) and drops pairs that
are clearly hopeless. What survives is what the full matching model runs on, so it is
exactly what goes into candidate_pairs.tsv.

The threshold is picked on the VALIDATION fold as the highest one that still keeps
`target_recall` of the true pairs blocking found, optionally with a hard per-S1 cap.
"""
import numpy as np
import pandas as pd

from .blocking import rank_desc

FILTER_FEATURES = ["name_sim", "addr_sim", "cheap", "nkeys", "rank_in_c", "gap_in_c", "n_s1_for_c",
                   "rank_in_s1", "gap_in_s1", "n_c_for_s1", "rank_name_in_s1", "rank_addr_in_s1",
                   "knn_sim", "knn_rank", "knnn_sim", "knnn_rank"]
FILTER_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200,
                     feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, n_jobs=-1, verbose=-1)


def _gap(group, v):
    return pd.Series(v).groupby(group).transform("max").to_numpy() - v


def filter_features(c: dict, n_s1: int, n_cand: int) -> pd.DataFrame:
    s1, cc = c["s1"], c["cand"]
    return pd.DataFrame({
        "name_sim": c["name_sim"], "addr_sim": c["addr_sim"], "cheap": c["cheap"],
        "nkeys": c["nkeys"].astype(np.float32),
        "rank_in_c": rank_desc(cc, c["cheap"]), "gap_in_c": _gap(cc, c["cheap"]),
        "n_s1_for_c": np.bincount(cc, minlength=n_cand)[cc],
        "rank_in_s1": rank_desc(s1, c["cheap"]), "gap_in_s1": _gap(s1, c["cheap"]),
        "n_c_for_s1": np.bincount(s1, minlength=n_s1)[s1],
        "rank_name_in_s1": rank_desc(s1, c["name_sim"]), "rank_addr_in_s1": rank_desc(s1, c["addr_sim"]),
        "knn_sim": c.get("knn_sim", np.full(len(s1), -1, np.float32)),
        "knn_rank": c.get("knn_rank", np.full(len(s1), -1, np.int8)).astype(np.float32),
        "knnn_sim": c.get("knnn_sim", np.full(len(s1), -1, np.float32)),
        "knnn_rank": c.get("knnn_rank", np.full(len(s1), -1, np.int8)).astype(np.float32),
    }).astype(np.float32)


def train_filter(F: pd.DataFrame, y: np.ndarray, train_rows: np.ndarray, max_rows: int = 4_000_000, seed: int = 0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    rows = train_rows if len(train_rows) <= max_rows else rng.choice(train_rows, max_rows, replace=False)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.9)
    dtr = lgb.Dataset(F.iloc[rows[:cut]][FILTER_FEATURES], y[rows[:cut]])
    dva = lgb.Dataset(F.iloc[rows[cut:]][FILTER_FEATURES], y[rows[cut:]], reference=dtr)
    m = lgb.train(FILTER_PARAMS, dtr, num_boost_round=400, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def filter_keep(s1, p, threshold: float, max_per_s1: int = 0) -> np.ndarray:
    keep = p >= threshold
    if max_per_s1:
        keep &= rank_desc(s1, p) < max_per_s1
    return keep


def choose_threshold(s1, p, y, val_rows, n_s1_val: int, target_recall: float = 0.995, max_per_s1: int = 0,
                     max_cands_per_s1: float = 0.0):
    """Prints the trade-off (candidates per S1 vs recall kept) and returns a threshold.
    Rule: keep >= target_recall of blocking's true pairs; if `max_cands_per_s1` is set, the
    candidate set must also stay at or below that size (on validation) - then the threshold with
    the highest recall under that size limit wins."""
    s1v, pv, yv = s1[val_rows], p[val_rows], y[val_rows]
    pos = max(yv.sum(), 1)
    rows = []
    grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    for t in grid:
        k = filter_keep(s1v, pv, t, max_per_s1)
        rows.append((t, yv[k].sum() / pos, k.sum() / n_s1_val))
    tab = pd.DataFrame(rows, columns=["threshold", "true_pairs_kept", "cands_per_S1"])
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    ok = tab[tab.true_pairs_kept >= target_recall]
    thr = float(ok.threshold.max()) if len(ok) else float(tab.threshold.min())
    if max_cands_per_s1:
        size_ok = tab[tab.cands_per_S1 <= max_cands_per_s1]
        if len(size_ok) and tab.loc[tab.threshold == thr, "cands_per_S1"].iloc[0] > max_cands_per_s1:
            thr = float(size_ok.threshold.min())          # loosest threshold that fits the size limit
    row = tab[tab.threshold == thr].iloc[0]
    print(f"chosen filter threshold {thr}: keeps {row.true_pairs_kept:.4f} of blocking's true pairs, "
          f"{row.cands_per_S1:.2f} candidates per S1 (validation)")
    return thr


def subset(c: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in c.items()}
