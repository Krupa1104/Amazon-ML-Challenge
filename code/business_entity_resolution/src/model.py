"""LightGBM pair classifier + the 'one owner per record' decision rule + threshold tuning."""
import numpy as np
import pandas as pd

from .features import MODEL_FEATURES
from .metrics import macro_f05_codes

LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_jobs=-1, verbose=-1)


def train_model(X: pd.DataFrame, y: np.ndarray, train_rows: np.ndarray, max_rows: int = 12_000_000,
                rounds: int = 5000, seed: int = 0, features=None, params=None):
    """Trains on a random sample of train_rows; 10% of that sample is used for early stopping."""
    import lightgbm as lgb
    features = list(features or MODEL_FEATURES)
    rng = np.random.default_rng(seed)
    rows = train_rows if len(train_rows) <= max_rows else rng.choice(train_rows, max_rows, replace=False)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.9)
    dtr = lgb.Dataset(X.iloc[rows[:cut]][features], y[rows[:cut]], free_raw_data=True)
    dva = lgb.Dataset(X.iloc[rows[cut:]][features], y[rows[cut:]], reference=dtr)
    model = lgb.train({**LGB_PARAMS, **(params or {})}, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    print(f"best iteration {model.best_iteration}")
    return model


def refit_all(X: pd.DataFrame, y: np.ndarray, rows: np.ndarray, rounds: int, features, seed: int = 0):
    """Final model on train + validation rows with a fixed number of rounds (no early stopping)."""
    import lightgbm as lgb
    d = lgb.Dataset(X.iloc[rows][list(features)], y[rows], free_raw_data=True)
    return lgb.train({**LGB_PARAMS, "seed": seed}, d, num_boost_round=rounds,
                     callbacks=[lgb.log_evaluation(500)])


def predict(model, X: pd.DataFrame, chunk: int = 2_000_000) -> np.ndarray:
    cols = model.feature_name()                  # always the exact features the model was trained on
    out = np.empty(len(X), np.float32)
    for b in range(0, len(X), chunk):
        out[b:b + chunk] = model.predict(X.iloc[b:b + chunk][cols], num_iteration=model.best_iteration)
    return out


def assign(s1: np.ndarray, cand: np.ndarray, prob: np.ndarray, threshold: float):
    """Each candidate record goes to at most ONE Source 1 entity: its highest-probability S1,
    and only if that probability >= threshold. Returns boolean mask over pairs."""
    order = np.lexsort((-prob, cand))
    first = np.r_[True, cand[order][1:] != cand[order][:-1]]
    best = np.zeros(len(prob), bool)
    best[order[first]] = True
    return best & (prob >= threshold)


def tune_threshold(s1, cand, prob, gt_s1, gt_pc, n_s1, n_cand, eval_mask,
                   grid=np.round(np.r_[np.arange(0.20, 0.90, 0.05), np.arange(0.90, 0.991, 0.01)], 2)):
    pc = s1 * n_cand + cand
    best = assign(s1, cand, prob, -1.0)        # each record's best S1 does not depend on the threshold
    s1b, pcb, pb = s1[best], pc[best], prob[best]
    results = []
    for t in grid:
        m = pb >= t
        results.append((t, macro_f05_codes(s1b[m], pcb[m], gt_s1, gt_pc, n_s1, eval_mask)))
    res = pd.DataFrame(results, columns=["threshold", "macro_f05"])
    best_row = res.loc[res.macro_f05.idxmax()]
    print(res.to_string(index=False))
    print(f"best threshold {best_row.threshold:.2f} -> val macro F0.5 {best_row.macro_f05:.4f}")
    return float(best_row.threshold), res


def tune_threshold_by_country(s1, cand, prob, gt_s1, gt_pc, n_s1, n_cand, eval_mask, s1_country,
                              grid=np.round(np.r_[np.arange(0.30, 0.90, 0.025), np.arange(0.90, 0.991, 0.01)], 3),
                              return_scores: bool = False):
    """One threshold per country (entities of different countries never compete for a record,
    so each country can be tuned on its own). Countries unseen in training use the global one."""
    pc = s1 * n_cand + cand
    best = assign(s1, cand, prob, -1.0)
    s1b, pcb, pb = s1[best], pc[best], prob[best]
    thr, best_scores = {}, {}
    for c in np.unique(s1_country):
        em = eval_mask & (s1_country == c)
        if em.sum() == 0:
            continue
        scores = [macro_f05_codes(s1b[pb >= t], pcb[pb >= t], gt_s1, gt_pc, n_s1, em) for t in grid]
        k = int(np.argmax(scores))
        thr[str(c)] = float(grid[k])
        best_scores[str(c)] = float(scores[k])
        print(f"  {c}: best threshold {grid[k]:.3f} -> val macro F0.5 {scores[k]:.4f}")
    return (thr, best_scores) if return_scores else thr


def thresholds_for(s1, s1_country, thr_by_country: dict, default: float) -> np.ndarray:
    """Per-pair threshold array from the S1 entity's country."""
    lut = {c: thr_by_country.get(str(c), default) for c in np.unique(s1_country)}
    per_s1 = np.array([lut[c] for c in s1_country], np.float32)
    return per_s1[s1]