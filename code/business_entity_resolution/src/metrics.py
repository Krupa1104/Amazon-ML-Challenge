"""Official metric: macro-averaged F_beta (beta=0.5) over Source 1 entities.

Per entity:  F = (1+b^2)*TP / ((1+b^2)*TP + b^2*FN + FP)   (identical to the PDF formula)
  * true matches empty & predicted empty      -> 1.0
  * true matches empty & anything predicted   -> 0.0
  * true matches exist & nothing predicted    -> 0.0
"""
import numpy as np
import pandas as pd


def per_entity_fbeta(pred_pairs: pd.DataFrame, gt_pairs: pd.DataFrame, eval_s1_ids,
                     beta: float = 0.5) -> pd.DataFrame:
    ids = pd.Index(pd.unique(np.asarray(eval_s1_ids)), name="s1_id")
    pred = pred_pairs.loc[pred_pairs["s1_id"].isin(ids), ["s1_id", "cand_id"]].drop_duplicates()
    gt = gt_pairs.loc[gt_pairs["s1_id"].isin(ids), ["s1_id", "cand_id"]]
    n_pred = pred.groupby("s1_id").size().reindex(ids, fill_value=0)
    n_gt = gt.groupby("s1_id").size().reindex(ids, fill_value=0)
    tp = pred.merge(gt, on=["s1_id", "cand_id"]).groupby("s1_id").size().reindex(ids, fill_value=0)
    b2 = beta ** 2
    fn, fp = n_gt - tp, n_pred - tp
    denom = ((1 + b2) * tp + b2 * fn + fp).to_numpy(dtype=float)
    f = np.divide((1 + b2) * tp.to_numpy(dtype=float), denom,
                  out=np.ones_like(denom), where=denom > 0)  # denom==0 only when both empty
    return pd.DataFrame({"n_gt": n_gt.to_numpy(), "n_pred": n_pred.to_numpy(),
                         "tp": tp.to_numpy(), "f05": f}, index=ids)


def macro_f05(pred_pairs, gt_pairs, eval_s1_ids) -> float:
    return float(per_entity_fbeta(pred_pairs, gt_pairs, eval_s1_ids)["f05"].mean())


def score_report(pred_pairs, gt_pairs, eval_s1_ids, s1_country: pd.Series = None, name="model"):
    """Prints overall macro F0.5, micro precision/recall, and breakdowns.
    s1_country: Series indexed by s1_id -> country (optional)."""
    d = per_entity_fbeta(pred_pairs, gt_pairs, eval_s1_ids)
    P = d.tp.sum() / max(d.n_pred.sum(), 1)
    R = d.tp.sum() / max(d.n_gt.sum(), 1)
    print(f"[{name}] macro F0.5 = {d.f05.mean():.4f} | pair precision {P:.4f} | pair recall {R:.4f}"
          f" | entities {len(d):,}")
    sing = d.n_gt == 0
    print(f"    singletons: F0.5 {d.f05[sing].mean():.4f} (n={sing.sum():,})"
          f" | non-singletons: F0.5 {d.f05[~sing].mean():.4f}")
    if s1_country is not None:
        c = s1_country.reindex(d.index)
        print("    by country:", d.groupby(c.to_numpy())["f05"].mean().round(4).to_dict())
    return d


def blocking_report(cand_pairs: pd.DataFrame, gt_pairs: pd.DataFrame, eval_s1_ids, name="blocking"):
    """Recall ceiling = share of true pairs that survive blocking; plus candidate-set size."""
    ids = pd.Index(pd.unique(np.asarray(eval_s1_ids)))
    cand = cand_pairs.loc[cand_pairs["s1_id"].isin(ids), ["s1_id", "cand_id"]].drop_duplicates()
    gt = gt_pairs.loc[gt_pairs["s1_id"].isin(ids), ["s1_id", "cand_id"]]
    hit = gt.merge(cand, on=["s1_id", "cand_id"]).shape[0]
    per = cand.groupby("s1_id").size().reindex(ids, fill_value=0)
    print(f"[{name}] pair recall {hit / max(len(gt), 1):.4f} | candidates/S1 mean {per.mean():.1f}"
          f" median {per.median():.0f} p95 {per.quantile(.95):.0f} | total {len(cand):,}")
    return hit / max(len(gt), 1)


# ------------------------------------------------------------------ fast integer-code version
def macro_f05_codes(pred_s1, pred_pc, gt_s1, gt_pc, n_s1: int, eval_mask: np.ndarray,
                    beta: float = 0.5, return_per_entity: bool = False):
    """Same metric on integer codes (fast for millions of pairs).
    pred_pc / gt_pc: pair codes s1*N_CAND + cand. eval_mask: bool array over S1 codes."""
    b2 = beta ** 2
    is_tp = np.isin(pred_pc, gt_pc)
    n_pred = np.bincount(pred_s1, minlength=n_s1)
    n_gt = np.bincount(gt_s1, minlength=n_s1)
    tp = np.bincount(pred_s1[is_tp], minlength=n_s1)
    denom = (1 + b2) * tp + b2 * (n_gt - tp) + (n_pred - tp)
    f = np.divide((1 + b2) * tp, denom, out=np.ones(n_s1), where=denom > 0)
    if return_per_entity:
        return f[eval_mask].mean(), f, n_gt, n_pred, tp
    return f[eval_mask].mean()
