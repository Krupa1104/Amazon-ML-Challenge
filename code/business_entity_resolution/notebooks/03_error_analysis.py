# %% [markdown]
# # Error analysis: why are points lost? (set VERSION below)
# Uses only cached files from 02 (normalized data, candidates, features, model), so nothing
# heavy is recomputed. Restart the kernel first. Paste the printed output back.

# %% Setup + load cached train data
import json
import sys
from pathlib import Path

REPO = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, REPO_ROOT
from src.io_utils import load_gt_pairs
from src.normalize import load_normalized, load_skeleton, load_concat, NORM_COLS
from src.split import split_ids
from src.blocking import gt_codes
from src.diagnostics import blocking_miss_report, model_error_report

VERSION = "v5"     # which pipeline run to analyse
COLS = ["entity_id", "country"] + NORM_COLS


def load_one(source):
    df = load_normalized("train", source, columns=COLS)
    df["name_skel"] = load_skeleton("train", source).to_numpy()
    df["name_concat"] = load_concat("train", source).to_numpy()
    return df


s1n = load_one("source1")
candn = pd.concat([load_one("source2"), load_one("source3")], ignore_index=True)
gt_s1, gt_pc = gt_codes(load_gt_pairs(), s1n["entity_id"], candn["entity_id"])
val_mask = split_ids(s1n["entity_id"]).to_numpy() == "val"
cands = dict(np.load(CACHE_DIR / f"cands_{VERSION}_train.npz"))
cfg = json.load(open(REPO_ROOT / "models" / f"config_{VERSION}.json"))
if "filter_threshold" in cfg:   # v4+: analyse the filtered candidate set the model actually saw
    import lightgbm as lgb
    from src.prefilter import FILTER_FEATURES, filter_features, filter_keep, subset
    fm = lgb.Booster(model_file=str(REPO_ROOT / "models" / f"filter_{VERSION}.txt"))
    p1 = fm.predict(filter_features(cands, len(s1n), len(candn))[FILTER_FEATURES])
    cands = subset(cands, filter_keep(cands["s1"], p1, cfg["filter_threshold"], cfg["filter_max_per_s1"]))
    FEAT_FILE = CACHE_DIR / f"features_{VERSION}_f{cfg['filter_threshold']}_c{cfg['filter_max_per_s1']}_train.parquet"
else:
    FEAT_FILE = CACHE_DIR / f"features_{VERSION}_train.parquet"
print("loaded")

# %% 1) Blocking: why do 12.7% of true pairs never reach the model?
cat = blocking_miss_report(s1n, candn, cands, gt_s1, gt_pc, val_mask, per_cand=4)

# %% 2) Model: false merges and missed matches among pairs it did see
import lightgbm as lgb
from src.model import predict, assign

model = lgb.Booster(model_file=str(REPO_ROOT / "models" / f"lgbm_{VERSION}.txt"))
model.best_iteration = model.current_iteration()
thr = json.load(open(REPO_ROOT / "models" / f"config_{VERSION}.json"))["threshold"]
X = pd.read_parquet(FEAT_FILE)
prob = predict(model, X)
del X
keep = assign(cands["s1"], cands["cand"], prob, thr)
model_error_report(s1n, candn, cands, prob, keep, gt_pc, val_mask)
