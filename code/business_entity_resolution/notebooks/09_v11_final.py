# %% [markdown]
# # v11 = v9 + seed ensemble (US/India) + self-training for unseen countries (France)
# 1) Seed ensemble: N_SEEDS copies of v9's pooled model and of the India model (same settings,
#    different random seeds for row/feature sampling), probabilities averaged. Kept only if it
#    beats v9 on the SAME validation set (v9: 0.9624).
# 2) France: pseudo-labels = the ensemble's most confident unseen-country test decisions,
#    one pooled model retrained with them, re-scores ONLY unseen-country pairs.
# Uses v9's cached features/models (no blocking, no feature recomputation). Every trained model
# and prediction is cached, so a crash resumes where it stopped. ~4-5 h with N_SEEDS = 3
# (set N_SEEDS = 2 for ~3 h). Restart the kernel first.

# %% Setup
import gc
import json
import sys
from pathlib import Path

REPO = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import CACHE_DIR, OUTPUT_DIR, REPO_ROOT
from src.io_utils import load_gt_pairs, write_submission_codes, run_validator
from src.normalize import load_normalized
from src.split import split_ids, drop_mask
from src.blocking import gt_codes
from src.prefilter import FILTER_FEATURES, filter_features, filter_keep, subset
from src.model import (LGB_PARAMS, train_model, predict, assign, tune_threshold,
                       tune_threshold_by_country, thresholds_for)
from src.metrics import macro_f05_codes

VERSION, BASE, FILTER_FROM = "v11", "v9", "v8"
N_SEEDS = 3                          # models per ensemble (seed 0 = v9's existing model)
PSEUDO_POS, PSEUDO_NEG, PSEUDO_WEIGHT = 0.95, 0.05, 0.7
DROP_S1_PCT = 19
M = REPO_ROOT / "models"
cfg9 = json.load(open(M / f"config_{BASE}.json"))
cfg8 = json.load(open(M / f"config_{FILTER_FROM}.json"))
fthr, fcap = cfg8["filter_threshold"], cfg8["filter_max_per_s1"]
fmodel = lgb.Booster(model_file=str(M / f"filter_{FILTER_FROM}.txt"))
COUNTRY_MODELS = [c for c, use in cfg9["use_country_model"].items() if use]   # e.g. ["India"]


def booster(path):
    b = lgb.Booster(model_file=str(path)); b.best_iteration = b.current_iteration()
    return b


def ids(split, source, cols=("entity_id", "country")):
    return load_normalized(split, source, columns=list(cols))


# test candidates from the v8 run (v7's cache name when test blocking was reused from v7)
TEST_CANDS = CACHE_DIR / "cands_v7_test.npz"
if not TEST_CANDS.exists():
    TEST_CANDS = CACHE_DIR / "cands_v8_test.npz"


def filtered(c, n1, nc):
    p = fmodel.predict(filter_features(c, n1, nc)[FILTER_FEATURES])
    return subset(c, filter_keep(c["s1"], p, fthr, fcap))


def seed_model(kind, seed, rows):
    """kind: 'pooled' or a country. seed 0 = the v9 model; others trained once and cached."""
    base = M / (f"lgbm_{BASE}_pooled.txt" if kind == "pooled" else f"lgbm_{BASE}_{kind}.txt")
    if seed == 0:
        return booster(base)
    f = M / f"lgbm_{VERSION}_{kind}_seed{seed}.txt"
    if f.exists():
        return booster(f)
    print(f"training {kind} model, seed {seed} on {len(rows):,} pairs ...")
    m = train_model(X, y, rows, seed=seed,
                    params={"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed})
    m.save_model(str(f), num_iteration=m.best_iteration)
    return booster(f)


def cached_pred(name, model, frame):
    f = CACHE_DIR / f"pred_{VERSION}_{name}.npy"
    if f.exists():
        return np.load(f)
    p = predict(model, frame)
    np.save(f, p)
    return p


# %% TRAIN side (cached v9 features)
s1 = ids("train", "source1")
s1 = s1[drop_mask(s1["entity_id"], DROP_S1_PCT)].reset_index(drop=True)
cand_ids = pd.concat([ids("train", s, ["entity_id"]) for s in ("source2", "source3")], ignore_index=True)["entity_id"]
N1, NC = len(s1), len(cand_ids)
gt_s1, gt_pc = gt_codes(load_gt_pairs(), s1["entity_id"], cand_ids)
val_mask = split_ids(s1["entity_id"]).to_numpy() == "val"
cands = filtered(dict(np.load(CACHE_DIR / f"cands_{FILTER_FROM}_train.npz")), N1, NC)
X = pd.read_parquet(CACHE_DIR / f"features_{BASE}_train.parquet")
assert len(X) == len(cands["s1"]), "cached v9 train features do not line up"
pc = cands["s1"] * NC + cands["cand"]
y = np.isin(pc, gt_pc).astype(np.int8)
cty = s1["country"].to_numpy()
pair_cty = cty[cands["s1"]]
train_rows = np.flatnonzero(~val_mask[cands["s1"]])
seen = set(np.unique(cty))


def evaluate(p, label):
    thr_c = tune_threshold_by_country(cands["s1"], cands["cand"], p, gt_s1, gt_pc, N1, NC, val_mask, cty)
    thr_g = tune_threshold(cands["s1"], cands["cand"], p, gt_s1, gt_pc, N1, NC, val_mask)[0]
    m = assign(cands["s1"], cands["cand"], p, thresholds_for(cands["s1"], cty, thr_c, thr_g))
    sc, f, n_gt, n_pred, tp = macro_f05_codes(cands["s1"][m], pc[m], gt_s1, gt_pc, N1, val_mask,
                                              return_per_entity=True)
    v = val_mask
    print(f"[{label}] VAL macro F0.5 {sc:.4f} | by country",
          {c: float(round(f[v & (cty == c)].mean(), 4)) for c in np.unique(cty)},
          f"| predicted matches per S1 {n_pred[v].mean():.2f}")
    return float(sc), thr_c, float(thr_g), float(n_pred[v].mean())


# %% 1) Seed ensemble on validation (v9 = seed 0)
pooled = [seed_model("pooled", k, train_rows) for k in range(N_SEEDS)]
p_single = cached_pred("val_pooled_seed0", pooled[0], X)
p_ens = np.mean([p_single] + [cached_pred(f"val_pooled_seed{k}", pooled[k], X) for k in range(1, N_SEEDS)], axis=0)
country_ens = {}
for c in COUNTRY_MODELS:
    rows_c = train_rows[pair_cty[train_rows] == c]
    country_ens[c] = [seed_model(c, k, rows_c) for k in range(N_SEEDS)]
    sel = np.flatnonzero(pair_cty == c)
    preds = [cached_pred(f"val_{c}_seed{k}", country_ens[c][k], X.iloc[sel]) for k in range(N_SEEDS)]
    p_single[sel] = preds[0]
    p_ens[sel] = np.mean(preds, axis=0)
sc9, thr_c9, thr_g9, rate9 = evaluate(p_single, "v9 single models")
scE, thr_cE, thr_gE, rateE = evaluate(p_ens, f"{N_SEEDS}-seed ensemble")
USE_ENS = scE > sc9
print(f"==> using {'ENSEMBLE' if USE_ENS else 'v9 single models'} for seen countries")
thr_c, thr_g, VAL_RATE = (thr_cE, thr_gE, rateE) if USE_ENS else (thr_c9, thr_g9, rate9)
del p_single, p_ens
gc.collect()

# %% TEST: seen-country probabilities (ensemble or v9) exactly as chosen
t1 = ids("test", "source1")
tc_ids = pd.concat([ids("test", s, ["entity_id"]) for s in ("source2", "source3")], ignore_index=True)["entity_id"].to_numpy()
tcands = filtered(dict(np.load(TEST_CANDS)), len(t1), len(tc_ids))
TX = pd.read_parquet(CACHE_DIR / f"features_{BASE}_test.parquet")
assert len(TX) == len(tcands["s1"]), "cached v9 test features do not line up"
tcty = t1["country"].to_numpy()
tpair_cty = tcty[tcands["s1"]]
ks = range(N_SEEDS) if USE_ENS else range(1)
tprob = np.mean([cached_pred(f"test_pooled_seed{k}", pooled[k], TX) for k in ks], axis=0)
for c in COUNTRY_MODELS:
    sel = np.flatnonzero(tpair_cty == c)
    tprob[sel] = np.mean([cached_pred(f"test_{c}_seed{k}", country_ens[c][k], TX.iloc[sel]) for k in ks], axis=0)
unseen_pairs = ~np.isin(tpair_cty, list(seen))
print("unseen test countries:", sorted(set(np.unique(tcty)) - seen), f"| {unseen_pairs.sum():,} pairs")

# %% 2) Self-training for unseen countries (France)
best = assign(tcands["s1"], tcands["cand"], tprob, -1.0)
pos = np.flatnonzero(unseen_pairs & best & (tprob >= PSEUDO_POS))
neg = np.flatnonzero(unseen_pairs & (tprob <= PSEUDO_NEG))
print(f"pseudo-labels: {len(pos):,} positive, {len(neg):,} negative")
ST_FILE = M / f"lgbm_{VERSION}_selftrain.txt"
feats = pooled[0].feature_name()
if ST_FILE.exists():
    st = booster(ST_FILE)
else:
    rng = np.random.default_rng(0)
    tr = rng.permutation(train_rows)
    cut = int(len(tr) * 0.9)
    Xtr = pd.concat([X.iloc[tr[:cut]][feats], TX.iloc[pos][feats], TX.iloc[neg][feats]], ignore_index=True)
    ytr = np.concatenate([y[tr[:cut]], np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
    wtr = np.concatenate([np.ones(cut), np.full(len(pos) + len(neg), PSEUDO_WEIGHT)]).astype(np.float32)
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr, free_raw_data=True)
    dva = lgb.Dataset(X.iloc[tr[cut:]][feats], y[tr[cut:]], reference=dtr)
    del Xtr
    st = lgb.train(LGB_PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
    st.save_model(str(ST_FILE), num_iteration=st.best_iteration)
    st = booster(ST_FILE)
    del dtr, dva
    gc.collect()
thr_unseen = tune_threshold(cands["s1"], cands["cand"], cached_pred("val_selftrain", st, X), gt_s1, gt_pc,
                            N1, NC, val_mask)[0]
del X
gc.collect()

# %% Final decisions -> submission
tprob_final = tprob.copy()
sel = np.flatnonzero(unseen_pairs)
tprob_final[sel] = predict(st, TX.iloc[sel])
tm = assign(tcands["s1"], tcands["cand"], tprob_final, thresholds_for(tcands["s1"], tcty, thr_c, thr_unseen))
r = np.bincount(tcands["s1"][tm], minlength=len(t1))
print(f"TEST: predicted matches per S1 {r.mean():.2f} vs VAL {VAL_RATE:.2f}  (v9: 3.12 vs 3.18)")
if r.mean() > VAL_RATE * 1.05:
    print("!! SHIFT ALARM: test predicts >5% more matches per S1 than validation - paste output before uploading")
print("matches per S1 by country:", {c: float(round(r[tcty == c].mean(), 3)) for c in np.unique(tcty)},
      "(v9 test: France/India/US S1 with >=1 match 0.934/0.927/0.939)")
print("S1 with >=1 match by country:", {c: float(round((r[tcty == c] > 0).mean(), 3)) for c in np.unique(tcty)})
print("thresholds:", thr_c, "| unseen:", thr_unseen, "| ensemble:", USE_ENS)
json.dump({"thr_by_country": thr_c, "thr_unseen": float(thr_unseen), "use_ensemble": bool(USE_ENS),
           "n_seeds": N_SEEDS, "val_score": scE if USE_ENS else sc9, "val_pred_per_s1": VAL_RATE,
           "country_models": COUNTRY_MODELS, "pseudo": {"pos": int(len(pos)), "neg": int(len(neg))}},
          open(M / f"config_{VERSION}.json", "w"))

out_dir = OUTPUT_DIR / VERSION
write_submission_codes(tcands["s1"][tm], tcands["cand"][tm], tcands["s1"], tcands["cand"],
                       t1["entity_id"].to_numpy(), tc_ids, out_dir=out_dir)
run_validator(out_dir)
print("upload:", out_dir / "matching_results.tsv")