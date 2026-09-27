"""Hyperparameter search and two-stage stacking for the matching model.

Stage 1: LightGBM on the pair features (tuned).
Stage 2: LightGBM on the pair features + CONTEXT from stage-1 probabilities of the competing
         pairs: how this pair's probability compares with the other S1 entities proposed for
         the same record (the record can only have one owner) and with the other records
         proposed for the same S1. Only ranks / gaps / margins are used (no counts), so the
         features do not depend on how dense a dataset is (the v5 lesson).
Stage 2 is trained on OUT-OF-FOLD stage-1 probabilities, so it never sees a stage-1 prediction
made on the pair's own training data.
"""
import numpy as np
import pandas as pd

from .blocking import rank_desc
from .model import LGB_PARAMS

CONTEXT = ["p1", "p1_rank_in_c", "p1_gap_in_c", "p1_margin_in_c", "p1_rank_in_s1", "p1_gap_in_s1",
           "p1_second_in_s1"]


def _best_second(group, v):
    """Per element: best and second-best value in its group."""
    order = np.lexsort((-v, group))
    g, vs = group[order], v[order]
    start = np.r_[True, g[1:] != g[:-1]]
    gid = np.cumsum(start) - 1
    first_idx = np.flatnonzero(start)
    best_g = vs[first_idx]
    has2 = np.r_[first_idx[1:], len(g)] - first_idx > 1
    second_g = np.where(has2, vs[np.minimum(first_idx + 1, len(g) - 1)], -1.0)
    best, second = np.empty(len(v), np.float32), np.empty(len(v), np.float32)
    best[order], second[order] = best_g[gid], second_g[gid]
    return best, second


def context_features(s1, cand, p) -> pd.DataFrame:
    p = p.astype(np.float32)
    best_c, second_c = _best_second(cand, p)
    other_c = np.where(p >= best_c, second_c, best_c)        # best competing S1 for this record
    best_s, second_s = _best_second(s1, p)
    return pd.DataFrame({
        "p1": p,
        "p1_rank_in_c": rank_desc(cand, p).astype(np.float32),
        "p1_gap_in_c": best_c - p,
        "p1_margin_in_c": p - other_c,
        "p1_rank_in_s1": rank_desc(s1, p).astype(np.float32),
        "p1_gap_in_s1": best_s - p,
        "p1_second_in_s1": second_s,
    })


SEARCH_SPACE = [
    dict(num_leaves=127, min_child_samples=100, feature_fraction=0.8, lambda_l2=1.0),
    dict(num_leaves=255, min_child_samples=100, feature_fraction=0.8, lambda_l2=1.0),
    dict(num_leaves=255, min_child_samples=300, feature_fraction=0.6, lambda_l2=5.0),
    dict(num_leaves=511, min_child_samples=200, feature_fraction=0.7, lambda_l2=5.0),
    dict(num_leaves=63, min_child_samples=50, feature_fraction=0.9, lambda_l2=0.0),
    dict(num_leaves=255, min_child_samples=50, feature_fraction=0.7, lambda_l2=10.0, min_sum_hessian_in_leaf=1.0),
]


def tune_params(X, y, train_rows, val_rows, features, n_train=3_000_000, n_val=1_000_000, rounds=600, seed=0):
    """Quick search on a sample: each config trained with lr 0.1 and early stopping, scored by
    validation log-loss on VALIDATION-fold pairs (held-out S1 entities)."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    tr = rng.choice(train_rows, min(n_train, len(train_rows)), replace=False)
    va = rng.choice(val_rows, min(n_val, len(val_rows)), replace=False)
    dtr = lgb.Dataset(X.iloc[tr][features], y[tr], free_raw_data=False)
    dva = lgb.Dataset(X.iloc[va][features], y[va], reference=dtr, free_raw_data=False)
    results = []
    for k, cfg in enumerate(SEARCH_SPACE):
        params = {**LGB_PARAMS, **cfg, "learning_rate": 0.1}
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        ll = m.best_score["valid_0"]["binary_logloss"]
        results.append((k, ll, m.best_iteration))
        print(f"  config {k}: {cfg} -> val logloss {ll:.5f} (iter {m.best_iteration})")
    best = min(results, key=lambda r: r[1])
    print(f"best config {best[0]} (logloss {best[1]:.5f})")
    return {**LGB_PARAMS, **SEARCH_SPACE[best[0]]}


def oof_predict(X, y, rows, s1_codes, features, params, rounds, folds=2, seed=0):
    """Out-of-fold stage-1 probabilities for `rows`, folds split by S1 entity."""
    import lightgbm as lgb
    fold = (pd.util.hash_pandas_object(pd.Series(s1_codes[rows]), index=False).to_numpy() // 7) % folds
    out = np.zeros(len(rows), np.float32)
    for k in range(folds):
        tr, te = rows[fold != k], rows[fold == k]
        m = lgb.train({**params, "seed": seed + k}, lgb.Dataset(X.iloc[tr][features], y[tr]),
                      num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        for b in range(0, len(te), 2_000_000):
            out[np.flatnonzero(fold == k)[b:b + 2_000_000]] = m.predict(X.iloc[te[b:b + 2_000_000]][features])
        print(f"  OOF fold {k + 1}/{folds} done")
    return out
