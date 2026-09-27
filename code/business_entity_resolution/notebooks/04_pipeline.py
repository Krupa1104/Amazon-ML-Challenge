# %% [markdown]
# # Final pipeline: blocking -> features -> LightGBM -> one-owner assignment -> submission
# v2: multi-key blocking, 34 features, LightGBM, one-owner assignment
# v3: per-family caps, pair/phonetic keys, house-number / phonetic / address-uniqueness features
# v4: full-name and glued-name keys, glued-name + address-length features, 4 S1s per record,
#     learned candidate filter (small candidate set per S1), fast threshold tuning
# v5: rarest name/sound token x up to 4 address words, sibling-corroboration features,
#     oracle ceilings (how much of the gap is blocking vs model), up to 5000 rounds
# v6: similarity-search candidates (char n-gram TF-IDF -> LSA -> FAISS, 5 nearest S1 per record)
#     merged with key blocking; kNN similarity/rank features; sibling features dropped
#     (they shifted between train and test); test-vs-validation shift alarm
# v7: second, NAME-ONLY similarity search (finds same business at a different address),
#     candidate-set size limit in the filter, hyperparameter search, all train rows,
#     two-stage stacking (stage 2 sees stage-1 probabilities of competing pairs, trained
#     out-of-fold), per-country thresholds; the better stage is chosen on validation
# v8: TEST-LIKE TRAINING: hide 19% of train S1 entities so their S2/S3 records become
#     ownerless distractors (train density 4.7 -> ~5.8 records per S1, same as test);
#     v6 model settings (no tuning / stacking), per-country thresholds
# Caches/models/outputs are named by VERSION, so earlier versions' files stay untouched.
# Run cell by cell (Shift+Enter). Heavy steps cache their results in data_cache/, so if the
# kernel dies you can restart and re-run: finished steps load from disk in seconds.
# Rough laptop timings (16 GB): phonetic skeleton ~3 min (once), blocking ~20-45 min,
# features ~15-25 min, training ~20-40 min; test side ~half that. Restart the kernel first.

# %% Setup
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, OUTPUT_DIR, REPO_ROOT
from src.io_utils import load_gt_pairs, write_submission_codes, run_validator
from src.normalize import load_normalized, load_skeleton, load_concat, NORM_COLS
from src.split import split_ids, drop_mask
from src.blocking import generate_candidates, gt_codes, blocking_recall, oracle_ceiling, DEFAULT_CAPS
from src.features import compute_features, MODEL_FEATURES as FEATURES
from src.model import (train_model, predict, assign, tune_threshold,
                       tune_threshold_by_country, thresholds_for)
from src.stack import CONTEXT, context_features, tune_params, oof_predict
from src.metrics import macro_f05_codes
from src.retrieval import Embedder, knn_candidates, merge_knn
from src.prefilter import (FILTER_FEATURES, filter_features, train_filter, choose_threshold,
                           filter_keep, subset)

VERSION = "v8"
DROP_S1_PCT = 19     # hide this % of train S1 entities -> their records become distractors (test-like density)
USE_TUNING = False   # v7 lesson: bigger tuned model + stacking did not transfer to test
USE_STACK = False
TEST_CACHE_FROM = "v7"   # test similarity search + blocking are unchanged: reuse v7's cached files
KNN_FROM = "v6"      # reuse the cached combined (name+address) similarity search of this version
OOF_FOLDS = 2        # out-of-fold stage-1 predictions for training stage 2
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
COLS = ["entity_id", "country"] + NORM_COLS
BLOCK_PARAMS = dict(caps=DEFAULT_CAPS, per_s1=30, per_cand=4, min_cheap=30.0)
KNN_K = 5            # nearest Source 1 entities per S2/S3 record (name + address search)
KNN_K_NAME = 4       # nearest Source 1 entities per record (name-only search)
# Learned filter: keep this share of the true pairs blocking found (lower -> smaller candidate set).
# Optional hard cap on candidates per Source 1 entity (0 = no cap).
FILTER_RECALL = 0.995
FILTER_MAX_PER_S1 = 0
FILTER_MAX_CANDS = 5.0   # size limit: at most this many candidates per S1 on validation (~6 on test)


def load_one(split, source):
    df = load_normalized(split, source, columns=COLS)
    df["name_skel"] = load_skeleton(split, source).to_numpy()
    df["name_concat"] = load_concat(split, source).to_numpy()
    return df


def load_split(split):
    s1n = load_one(split, "source1")
    candn = pd.concat([load_one(split, "source2"), load_one(split, "source3")], ignore_index=True)
    return s1n, candn


def cached_knn(split, s1n, candn, emb, name="knn", version=None):
    f = CACHE_DIR / f"{name}_{version or VERSION}_{split}.npz"
    if f.exists():
        return dict(np.load(f))
    t = time.time()
    r = knn_candidates(emb, s1n, candn, k=KNN_K if name == "knn" else KNN_K_NAME)
    np.savez(f, **r)
    print(f"similarity search took {(time.time() - t) / 60:.1f} min")
    return r


def cached_candidates(split, s1n, candn, knn):
    f = CACHE_DIR / f"cands_{VERSION}_{split}.npz"
    if f.exists():
        return dict(np.load(f))
    t = time.time()
    c = generate_candidates(s1n, candn, knn=knn, **BLOCK_PARAMS)
    np.savez(f, **c)
    print(f"blocking took {(time.time() - t) / 60:.1f} min")
    return c


def cached_features(split, cands, s1n, candn, tag):
    f = CACHE_DIR / f"features_{VERSION}_{tag}_{split}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t = time.time()
    X = compute_features(cands, s1n, candn)
    X.to_parquet(f, index=False)
    print(f"features took {(time.time() - t) / 60:.1f} min")
    return X


# %% Load normalized TRAIN data (+ phonetic skeleton and glued name, computed once and cached)
s1n, candn = load_split("train")
KEEP = drop_mask(s1n["entity_id"], DROP_S1_PCT)
OLD2NEW = np.full(len(s1n), -1, np.int64)
OLD2NEW[KEEP] = np.arange(KEEP.sum())
s1n = s1n[KEEP].reset_index(drop=True)
print(f"S1 {len(s1n):,} (kept {KEEP.mean():.3f}) | S2+S3 {len(candn):,} | "
      f"records per S1 {len(candn) / len(s1n):.2f} (test: 5.75, original train: 4.68)")


def remap_knn(r):
    """kNN cache computed on the full S1 table -> codes of the kept S1 table."""
    new = OLD2NEW[r["s1"]]
    ok = new >= 0
    return {k: (new[ok] if k == "s1" else v[ok]) for k, v in r.items()}

# %% Ground truth as integer codes + the shared validation split
gt = load_gt_pairs()
gt_s1, gt_pc = gt_codes(gt, s1n["entity_id"], candn["entity_id"])
del gt
val_mask = split_ids(s1n["entity_id"]).to_numpy() == "val"
N1, NC = len(s1n), len(candn)
print(f"true pairs {len(gt_pc):,} | val S1 {val_mask.sum():,}")

# %% Similarity search x2: name+address (reused from v6 if cached) and NAME-ONLY (new)
def get_embedder(path, **kw):
    if path.exists():
        return Embedder.load(path)
    t = time.time()
    e = Embedder(**kw).fit([s1n, candn])
    e.save(path)
    print(f"embedder fitted in {(time.time() - t) / 60:.1f} min")
    return e


EMB_FILE = MODEL_DIR / f"embedder_{KNN_FROM}.pkl"          # combined name+address
EMB_NAME_FILE = MODEL_DIR / "embedder_name_v7.pkl"  # name only (fitted in v7)
emb = get_embedder(EMB_FILE)
emb_name = get_embedder(EMB_NAME_FILE, dim=128, w_addr=0.0)
def train_knn(name, version, emb_, k):
    """Reuse a similarity-search cache built on the FULL S1 table (remapped), or compute it
    fresh on the kept S1 table when running from scratch."""
    f = CACHE_DIR / f"{name}_{version}_train.npz"
    if f.exists():
        return remap_knn(dict(np.load(f)))
    t = time.time()
    r = knn_candidates(emb_, s1n, candn, k=k)
    np.savez(CACHE_DIR / f"{name}_{VERSION}_train.npz", **r)
    print(f"similarity search took {(time.time() - t) / 60:.1f} min")
    return r


knn_a = train_knn("knn", KNN_FROM, emb, KNN_K)
knn_b = train_knn("knnname", "v7", emb_name, KNN_K_NAME)
print("name+address search alone:"); blocking_recall(knn_a, gt_s1, gt_pc, NC, val_mask)
print("name-only search alone:"); blocking_recall(knn_b, gt_s1, gt_pc, NC, val_mask)
knn = merge_knn({"knn": knn_a, "knnn": knn_b}, NC)
del knn_a, knn_b
print("both searches together:"); blocking_recall(knn, gt_s1, gt_pc, NC, val_mask)

# %% Blocking (Member A's stage) merged with similarity search -> recall ceiling on validation
cands = cached_candidates("train", s1n, candn, knn)
del knn
gc.collect()
blocking_recall(cands, gt_s1, gt_pc, NC, val_mask)
oracle_ceiling(cands, gt_s1, gt_pc, N1, NC, val_mask, "after blocking")

# %% Learned candidate filter: last candidate-generation stage (its output = candidate_pairs.tsv)
F = filter_features(cands, N1, NC)
y_all = np.isin(cands["s1"] * NC + cands["cand"], gt_pc).astype(np.int8)
fmodel = train_filter(F, y_all, np.flatnonzero(~val_mask[cands["s1"]]))
fmodel.save_model(str(MODEL_DIR / f"filter_{VERSION}.txt"), num_iteration=fmodel.best_iteration)
p1 = fmodel.predict(F[FILTER_FEATURES], num_iteration=fmodel.best_iteration)
fthr = choose_threshold(cands["s1"], p1, y_all, np.flatnonzero(val_mask[cands["s1"]]), val_mask.sum(),
                        FILTER_RECALL, FILTER_MAX_PER_S1, FILTER_MAX_CANDS)
cands = subset(cands, filter_keep(cands["s1"], p1, fthr, FILTER_MAX_PER_S1))
del F, y_all, p1
gc.collect()
FTAG = f"f{fthr}_c{FILTER_MAX_PER_S1}"
print("after the filter (this is the candidate set the model sees):")
blocking_recall(cands, gt_s1, gt_pc, NC, val_mask)
oracle_ceiling(cands, gt_s1, gt_pc, N1, NC, val_mask, "after filter")

# %% Features (Member B's stage)
X = cached_features("train", cands, s1n, candn, FTAG)
pc = cands["s1"] * NC + cands["cand"]
y = np.isin(pc, gt_pc).astype(np.int8)
print(X.shape, "| positive rate", y.mean().round(3))

# %% Hyperparameter search on a sample (validation log-loss on held-out S1 entities)
train_rows = np.flatnonzero(~val_mask[cands["s1"]])
val_rows = np.flatnonzero(val_mask[cands["s1"]])
PARAMS_FILE = MODEL_DIR / f"params_{VERSION}.json"
if not USE_TUNING:
    from src.model import LGB_PARAMS
    PARAMS = dict(LGB_PARAMS)
elif PARAMS_FILE.exists():
    PARAMS = json.load(open(PARAMS_FILE))
else:
    PARAMS = tune_params(X, y, train_rows, val_rows, FEATURES)
    json.dump(PARAMS, open(PARAMS_FILE, "w"))
print("params:", PARAMS)


def evaluate(p, label):
    """Per-country thresholds on validation + full report. Returns (score, thresholds, global thr)."""
    thr = tune_threshold(cands["s1"], cands["cand"], p, gt_s1, gt_pc, N1, NC, val_mask)[0]
    thr_c = tune_threshold_by_country(cands["s1"], cands["cand"], p, gt_s1, gt_pc, N1, NC, val_mask, cty)
    m = assign(cands["s1"], cands["cand"], p, thresholds_for(cands["s1"], cty, thr_c, thr))
    sc, f, n_gt, n_pred, tp = macro_f05_codes(cands["s1"][m], pc[m], gt_s1, gt_pc, N1, val_mask,
                                              return_per_entity=True)
    v = val_mask
    print(f"[{label}] VAL macro F0.5 {sc:.4f} | pair precision {tp[v].sum() / max(n_pred[v].sum(), 1):.4f}"
          f" | pair recall {tp[v].sum() / n_gt[v].sum():.4f} | singletons {f[v & (n_gt == 0)].mean():.4f}")
    print(f"[{label}] by country:", {c: float(round(f[v & (cty == c)].mean(), 4)) for c in np.unique(cty)},
          f"| predicted matches per S1 {n_pred[v].mean():.2f} (true {n_gt[v].mean():.2f})")
    return float(sc), thr_c, float(thr), float(n_pred[v].mean())


cty = s1n["country"].to_numpy()

# %% Stage 1: tuned LightGBM on TRAIN-fold S1 entities (early stopping)
S1_FILE = MODEL_DIR / f"lgbm_{VERSION}_stage1.txt"
if S1_FILE.exists():
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(S1_FILE)); model.best_iteration = model.current_iteration()
else:
    model = train_model(X, y, train_rows, params=PARAMS)
    model.save_model(str(S1_FILE), num_iteration=model.best_iteration)
BEST_IT = int(model.best_iteration)
imp = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values(ascending=False)
print((imp / imp.sum()).round(3).head(12).to_string())

# %% Stage-1 probabilities: validation rows from the stage-1 model, train rows OUT-OF-FOLD
OOF_FILE = CACHE_DIR / f"oof_{VERSION}_{FTAG}.npy"
p1 = predict(model, X)
if not USE_STACK:
    pass
elif OOF_FILE.exists():
    p1[train_rows] = np.load(OOF_FILE)
else:
    oof = oof_predict(X, y, train_rows, cands["s1"], model.feature_name(), PARAMS, BEST_IT, OOF_FOLDS)
    np.save(OOF_FILE, oof)
    p1[train_rows] = oof
    del oof
sc1, thr_c1, thr1, rate1 = evaluate(p1, "stage 1")

# %% Stage 2: pair features + stage-1 context of the competing pairs
if USE_STACK:
    C = context_features(cands["s1"], cands["cand"], p1)
    for col in CONTEXT:
        X[col] = C[col].to_numpy()
    del C
    FEATURES2 = list(model.feature_name()) + CONTEXT
    model2 = train_model(X, y, train_rows, params={**PARAMS, "learning_rate": 0.1}, rounds=3000, features=FEATURES2)
    model2.save_model(str(MODEL_DIR / f"lgbm_{VERSION}_stage2.txt"), num_iteration=model2.best_iteration)
    p2 = predict(model2, X)
    sc2, thr_c2, thr2, rate2 = evaluate(p2, "stage 2")
else:
    sc2 = -1.0

USE_STAGE2 = USE_STACK and sc2 > sc1
score, thr_c, thr, VAL_RATE = (sc2, thr_c2, thr2, rate2) if USE_STAGE2 else (sc1, thr_c1, thr1, rate1)
print(f"==> using stage {2 if USE_STAGE2 else 1}: VAL macro F0.5 {score:.4f}")
json.dump({"threshold": thr, "thr_by_country": thr_c, "val_pred_per_s1": VAL_RATE, "val_score": score,
           "use_stage2": bool(USE_STAGE2), "params": PARAMS, "best_iteration": BEST_IT,
           "filter_threshold": fthr, "filter_max_per_s1": FILTER_MAX_PER_S1, "knn_k": KNN_K,
           "knn_k_name": KNN_K_NAME, "knn_from": KNN_FROM,
           "block_params": {**BLOCK_PARAMS, "caps": {str(k): v for k, v in DEFAULT_CAPS.items()}}},
          open(MODEL_DIR / f"config_{VERSION}.json", "w"))

# %% Free TRAIN memory before test
for _v in ["s1n", "candn", "cands", "X", "y", "pc", "p1", "p2", "gt_s1", "gt_pc", "train_rows",
           "val_rows", "fmodel", "emb", "emb_name", "model", "model2"]:
    globals().pop(_v, None)
gc.collect()

# %% TEST: same pipeline with the trained model -> submission files -> validator
import lightgbm as lgb
cfg = json.load(open(MODEL_DIR / f"config_{VERSION}.json"))
model = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{VERSION}_stage1.txt"))
model.best_iteration = model.current_iteration()
if cfg["use_stage2"]:
    model2 = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{VERSION}_stage2.txt"))
    model2.best_iteration = model2.current_iteration()
print("test uses stage", 2 if cfg["use_stage2"] else 1)
thr, fthr, fcap = cfg["threshold"], cfg["filter_threshold"], cfg["filter_max_per_s1"]
emb = Embedder.load(MODEL_DIR / f"embedder_{cfg['knn_from']}.pkl")
emb_name = Embedder.load(MODEL_DIR / "embedder_name_v7.pkl")
fmodel = lgb.Booster(model_file=str(MODEL_DIR / f"filter_{VERSION}.txt"))

t1, tc = load_split("test")
print("test countries:", t1["country"].value_counts().to_dict())
TC_FILE = CACHE_DIR / f"cands_{TEST_CACHE_FROM}_test.npz"
if TC_FILE.exists():                      # test blocking is identical to v7's: reuse it
    tcands = dict(np.load(TC_FILE))
else:
    tknn = merge_knn({"knn": cached_knn("test", t1, tc, emb, "knn", cfg["knn_from"]),
                      "knnn": cached_knn("test", t1, tc, emb_name, "knnname", "v7")}, len(tc))
    tcands = cached_candidates("test", t1, tc, tknn)
    del tknn
p1 = fmodel.predict(filter_features(tcands, len(t1), len(tc))[FILTER_FEATURES])
tcands = subset(tcands, filter_keep(tcands["s1"], p1, fthr, fcap))
del p1
per = np.bincount(tcands["s1"], minlength=len(t1))
print(f"TEST candidate set: {len(tcands['s1']):,} pairs | per S1 mean {per.mean():.2f}, "
      f"median {np.median(per):.0f}, p95 {np.percentile(per, 95):.0f}")
TX = cached_features("test", tcands, t1, tc, f"f{fthr}_c{fcap}")
tprob = predict(model, TX)
if cfg["use_stage2"]:
    C = context_features(tcands["s1"], tcands["cand"], tprob)
    for col in CONTEXT:
        TX[col] = C[col].to_numpy()
    del C
    tprob = predict(model2, TX)
tcty = t1["country"].to_numpy()
# Countries unseen in training (France) get the STRICTEST tuned threshold: F0.5 punishes false
# merges twice as hard as misses, so be cautious where we have no labels.
UNSEEN_THR = max([thr] + list(cfg["thr_by_country"].values()))
print("thresholds:", cfg["thr_by_country"], "| unseen countries (e.g. France):", UNSEEN_THR)
tm = assign(tcands["s1"], tcands["cand"], tprob,
            thresholds_for(tcands["s1"], tcty, cfg["thr_by_country"], UNSEEN_THR))

TEST_RATE = np.bincount(tcands["s1"][tm], minlength=len(t1)).mean()
print(f"TEST: predicted matches per S1 {TEST_RATE:.2f} vs VAL {cfg['val_pred_per_s1']:.2f}  (v4 test: 3.17, LB 0.944)")
if TEST_RATE > cfg["val_pred_per_s1"] * 1.05:
    print("!! SHIFT ALARM: test predicts >5% more matches per S1 than validation - paste output before uploading")
has = np.bincount(tcands["s1"][tm], minlength=len(t1)) > 0
cty = t1["country"].to_numpy()
print(f"matches {tm.sum():,} | S1 with >=1 match {has.mean():.3f}")
print("S1 with >=1 match by country:", {c: float(round(has[cty == c].mean(), 3)) for c in np.unique(cty)})

out_dir = OUTPUT_DIR / VERSION
write_submission_codes(tcands["s1"][tm], tcands["cand"][tm], tcands["s1"], tcands["cand"],
                       t1["entity_id"].to_numpy(), tc["entity_id"].to_numpy(), out_dir=out_dir)
run_validator(out_dir)
print("upload:", out_dir / "matching_results.tsv")