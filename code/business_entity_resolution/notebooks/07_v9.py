# %% [markdown]
# # v9: targets India + France on top of v8
#  * rarity-weighted (IDF) name/address overlap features (sharing a rare token is strong evidence;
#    a rare token missing on one side is strong evidence against)
#  * French text fixes for the pair features (elision l'/d' before vowels, chemin/allee/impasse/
#    route/quai/saint, cedex/bis dropped)
#  * a separate model per training country (US, India) next to the pooled model; each country
#    uses whichever scores better on ITS OWN validation entities; unseen countries (France) use
#    the pooled model and its global threshold
# Reuses v8's cached blocking + filter (no blocking recomputed). ~3-3.5 h. Restart kernel first.
# Needs from the v8 run: data_cache/cands_v8_train.npz, data_cache/cands_v7_test.npz,
# models/filter_v8.txt, models/config_v8.json

# %% Setup
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import CACHE_DIR, OUTPUT_DIR, REPO_ROOT
from src.io_utils import load_gt_pairs, write_submission_codes, run_validator
from src.normalize import load_normalized, load_skeleton, load_concat, load_fr_fixed, NORM_COLS
from src.split import split_ids, drop_mask
from src.blocking import gt_codes, blocking_recall, oracle_ceiling
from src.prefilter import FILTER_FEATURES, filter_features, filter_keep, subset
from src.features import compute_features, idf_tables
from src.model import train_model, predict, assign, tune_threshold, tune_threshold_by_country, thresholds_for
from src.metrics import macro_f05_codes

VERSION, SRC = "v9", "v8"
DROP_S1_PCT = 19           # same as v8 (the cached v8 candidates were built this way)
MIN_COUNTRY_ROWS = 500_000 # train a country-specific model only with enough training pairs
MODEL_DIR = REPO_ROOT / "models"
cfg8 = json.load(open(MODEL_DIR / f"config_{SRC}.json"))
fthr, fcap = cfg8["filter_threshold"], cfg8["filter_max_per_s1"]
fmodel = lgb.Booster(model_file=str(MODEL_DIR / f"filter_{SRC}.txt"))
COLS = ["entity_id", "country"] + NORM_COLS


def load_one(split, source):
    df = load_normalized(split, source, columns=COLS)
    df["name_skel"] = load_skeleton(split, source).to_numpy()
    df["name_concat"] = load_concat(split, source).to_numpy()
    fr = load_fr_fixed(split, source)                     # v9 French fixes for the pair features
    for c in fr.columns:
        df[c] = fr[c].to_numpy()
    return df


def load_split(split):
    s1n = load_one(split, "source1")
    candn = pd.concat([load_one(split, "source2"), load_one(split, "source3")], ignore_index=True)
    return s1n, candn


# test candidates from the v8 run (v7's cache name when test blocking was reused from v7)
TEST_CANDS = CACHE_DIR / "cands_v7_test.npz"
if not TEST_CANDS.exists():
    TEST_CANDS = CACHE_DIR / "cands_v8_test.npz"


def filtered(cands, n1, nc):
    p = fmodel.predict(filter_features(cands, n1, nc)[FILTER_FEATURES])
    return subset(cands, filter_keep(cands["s1"], p, fthr, fcap))


def cached_features(split, cands, s1n, candn, idf):
    f = CACHE_DIR / f"features_{VERSION}_{split}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t = time.time()
    X = compute_features(cands, s1n, candn, idf=idf)
    X.to_parquet(f, index=False)
    print(f"features took {(time.time() - t) / 60:.1f} min")
    return X


# %% TRAIN data (same S1 subset as v8), cached v8 candidates + v8 filter
s1n, candn = load_split("train")
s1n = s1n[drop_mask(s1n["entity_id"], DROP_S1_PCT)].reset_index(drop=True)
N1, NC = len(s1n), len(candn)
gt_s1, gt_pc = gt_codes(load_gt_pairs(), s1n["entity_id"], candn["entity_id"])
val_mask = split_ids(s1n["entity_id"]).to_numpy() == "val"
cands = filtered(dict(np.load(CACHE_DIR / f"cands_{SRC}_train.npz")), N1, NC)
print(f"S1 {N1:,} | records {NC:,} | candidate pairs {len(cands['s1']):,}")
blocking_recall(cands, gt_s1, gt_pc, NC, val_mask)
oracle_ceiling(cands, gt_s1, gt_pc, N1, NC, val_mask, "after filter")

# %% Features (with rarity weights from the train corpus)
idf = idf_tables([s1n, candn])
X = cached_features("train", cands, s1n, candn, idf)
del idf
pc = cands["s1"] * NC + cands["cand"]
y = np.isin(pc, gt_pc).astype(np.int8)
cty = s1n["country"].to_numpy()
pair_cty = cty[cands["s1"]]
train_rows = np.flatnonzero(~val_mask[cands["s1"]])
print(X.shape, "| positive rate", round(float(y.mean()), 3))

# %% Pooled model (all countries) - also the model for unseen countries such as France
POOL_FILE = MODEL_DIR / f"lgbm_{VERSION}_pooled.txt"
if POOL_FILE.exists():
    pooled = lgb.Booster(model_file=str(POOL_FILE)); pooled.best_iteration = pooled.current_iteration()
else:
    pooled = train_model(X, y, train_rows)
    pooled.save_model(str(POOL_FILE), num_iteration=pooled.best_iteration)
imp = pd.Series(pooled.feature_importance("gain"), index=pooled.feature_name()).sort_values(ascending=False)
print((imp / imp.sum()).round(3).head(15).to_string())
p_pool = predict(pooled, X)
thr_global = tune_threshold(cands["s1"], cands["cand"], p_pool, gt_s1, gt_pc, N1, NC, val_mask)[0]
thr_pool, sc_pool = tune_threshold_by_country(cands["s1"], cands["cand"], p_pool, gt_s1, gt_pc, N1, NC,
                                              val_mask, cty, return_scores=True)

# %% Country-specific models: keep each only if it beats the pooled model on that country
p_final = p_pool.copy()
use_country_model = {}
for c in np.unique(cty):
    rows_c = train_rows[pair_cty[train_rows] == c]
    if len(rows_c) < MIN_COUNTRY_ROWS:
        continue
    f = MODEL_DIR / f"lgbm_{VERSION}_{c}.txt"
    if f.exists():
        mc = lgb.Booster(model_file=str(f)); mc.best_iteration = mc.current_iteration()
    else:
        print(f"training {c}-only model on {len(rows_c):,} pairs ...")
        mc = train_model(X, y, rows_c)
        mc.save_model(str(f), num_iteration=mc.best_iteration)
    sel = np.flatnonzero(pair_cty == c)
    p_try = p_pool.copy()
    p_try[sel] = predict(mc, X.iloc[sel])
    _, sc_c = tune_threshold_by_country(cands["s1"], cands["cand"], p_try, gt_s1, gt_pc, N1, NC,
                                        val_mask & (cty == c), cty, return_scores=True)
    better = sc_c[str(c)] > sc_pool[str(c)]
    print(f"==> {c}: pooled {sc_pool[str(c)]:.4f} vs {c}-only {sc_c[str(c)]:.4f} -> use "
          f"{'country' if better else 'pooled'} model")
    use_country_model[str(c)] = bool(better)
    if better:
        p_final[sel] = p_try[sel]

# %% Final validation report (per-country thresholds on the chosen models)
thr_c = tune_threshold_by_country(cands["s1"], cands["cand"], p_final, gt_s1, gt_pc, N1, NC, val_mask, cty)
m = assign(cands["s1"], cands["cand"], p_final, thresholds_for(cands["s1"], cty, thr_c, thr_global))
score, f, n_gt, n_pred, tp = macro_f05_codes(cands["s1"][m], pc[m], gt_s1, gt_pc, N1, val_mask,
                                             return_per_entity=True)
v = val_mask
print(f"VAL macro F0.5 {score:.4f} (v8 on the same validation: 0.9577) | pair precision "
      f"{tp[v].sum() / max(n_pred[v].sum(), 1):.4f} | pair recall {tp[v].sum() / n_gt[v].sum():.4f}")
print("by country:", {c: float(round(f[v & (cty == c)].mean(), 4)) for c in np.unique(cty)},
      "(v8: India 0.9385, US 0.9706)")
VAL_RATE = float(n_pred[v].mean())
json.dump({"thr_by_country": thr_c, "thr_unseen": float(thr_global), "use_country_model": use_country_model,
           "val_score": float(score), "val_pred_per_s1": VAL_RATE, "filter_from": SRC},
          open(MODEL_DIR / f"config_{VERSION}.json", "w"))

# %% Free TRAIN memory
for _v in ["s1n", "candn", "cands", "X", "y", "pc", "p_pool", "p_final", "p_try", "m", "f", "n_gt",
           "n_pred", "tp", "gt_s1", "gt_pc", "train_rows", "pair_cty"]:
    globals().pop(_v, None)
gc.collect()

# %% TEST: cached v7 test candidates -> v8 filter -> v9 features -> chosen models -> submission
cfg = json.load(open(MODEL_DIR / f"config_{VERSION}.json"))
t1, tc = load_split("test")
tcands = filtered(dict(np.load(TEST_CANDS)), len(t1), len(tc))
per = np.bincount(tcands["s1"], minlength=len(t1))
print(f"TEST candidate set per S1: mean {per.mean():.2f}, median {np.median(per):.0f}, p95 {np.percentile(per, 95):.0f}")
tidf = idf_tables([t1, tc])
TX = cached_features("test", tcands, t1, tc, tidf)
del tidf
tcty = t1["country"].to_numpy()
tprob = predict(pooled, TX)
for c, use in cfg["use_country_model"].items():
    if use:
        mc = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{VERSION}_{c}.txt"))
        mc.best_iteration = mc.current_iteration()
        sel = np.flatnonzero(tcty[tcands["s1"]] == c)
        tprob[sel] = predict(mc, TX.iloc[sel])
print("thresholds:", cfg["thr_by_country"], "| unseen countries (France):", cfg["thr_unseen"],
      "| country models:", cfg["use_country_model"])
tm = assign(tcands["s1"], tcands["cand"], tprob,
            thresholds_for(tcands["s1"], tcty, cfg["thr_by_country"], cfg["thr_unseen"]))
rate = np.bincount(tcands["s1"][tm], minlength=len(t1))
print(f"TEST: predicted matches per S1 {rate.mean():.2f} vs VAL {cfg['val_pred_per_s1']:.2f}  (v8: 3.18 vs 3.17)")
if rate.mean() > cfg["val_pred_per_s1"] * 1.05:
    print("!! SHIFT ALARM: test predicts >5% more matches per S1 than validation - paste output before uploading")
print("S1 with >=1 match by country:", {c: float(round((rate[tcty == c] > 0).mean(), 3)) for c in np.unique(tcty)})

out_dir = OUTPUT_DIR / VERSION
write_submission_codes(tcands["s1"][tm], tcands["cand"][tm], tcands["s1"], tcands["cand"],
                       t1["entity_id"].to_numpy(), tc["entity_id"].to_numpy(), out_dir=out_dir)
run_validator(out_dir)
print("upload:", out_dir / "matching_results.tsv")