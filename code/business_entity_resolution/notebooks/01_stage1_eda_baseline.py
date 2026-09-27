# %% [markdown]
# # Stage 1: EDA, shared harness, first baseline
# Run cell by cell in VS Code: click "Run Cell" above each `# %%` line (or Shift+Enter).
#
# First run normalizes all data and caches it in data_cache/ (several minutes, once).

# %% Setup
import sys
from pathlib import Path

REPO = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

from src.config import DATA_DIR, CACHE_DIR, OUTPUT_DIR
from src.io_utils import load_source, load_gt_pairs, write_submission, run_validator
from src.normalize import load_normalized, normalize_name, normalize_address
from src.split import split_ids, subsample
from src.metrics import score_report, blocking_report
from src.baseline import exact_key_match

print("data:", DATA_DIR, "| exists:", DATA_DIR.exists())
print("cache:", CACHE_DIR)

# %% Quick check of the normalizers on tricky examples
for n in ["Orelee's Barbershop", "Smart Healthcare Private Limited", "Callicoat & Dailey Inc",
          "ACME Corp. d/b/a Acme Robotics", "Société Générale SARL", "The B+ Retail, Inc."]:
    print(f"{n!r:45} -> {normalize_name(n)}")
for a in ["2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD",
          "294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI",
          "303, 3Rd Floor Sakar 5 B/H Natraj Cinema Ashram Road, Ahmedabad, Gujarat",
          "12 bd Saint-Michel, 75005 Paris"]:
    print(f"{a[:60]!r:62} -> {normalize_address(a)}")

# %% Normalize + cache all TRAIN sources (slow the first time only)
cols = ["entity_id", "country", "name_key", "name_core", "addr_bag", "addr_nums"]
s1n = load_normalized("train", "source1", columns=cols)
s2n = load_normalized("train", "source2", columns=cols)
s3n = load_normalized("train", "source3", columns=cols)
candn = pd.concat([s2n, s3n], ignore_index=True)
del s2n, s3n
print(f"S1 {len(s1n):,} | S2+S3 {len(candn):,}")

# %% Ground truth + split (identical for every teammate)
gt = load_gt_pairs()
fold = split_ids(s1n["entity_id"])
val_ids = fold.index[fold == "val"].to_numpy()
train_ids = fold.index[fold == "train"].to_numpy()
s1_country = s1n.set_index("entity_id")["country"]
print(f"GT pairs {len(gt):,} | train S1 {len(train_ids):,} | val S1 {len(val_ids):,}")

# %% EDA: countries, distractors, name-key ambiguity
print("S1 countries:", s1n["country"].value_counts().to_dict())
print("S2+S3 countries:", candn["country"].value_counts().to_dict())
owned = candn["entity_id"].isin(gt["cand_id"].unique())
print(f"S2/S3 records owned by some S1: {owned.mean():.3f} -> distractors {1 - owned.mean():.3f}")

n_match = gt.groupby("s1_id").size().reindex(s1n["entity_id"], fill_value=0)
print("singleton rate by country:",
      (n_match == 0).groupby(s1n["country"].to_numpy()).mean().round(4).to_dict())
dup_key = s1n.duplicated(subset=["country", "name_key"], keep=False)
print(f"S1 entities whose name_key is shared with another S1: {dup_key.mean():.3f}")

# %% Look at real matches (do this a lot: it's where feature ideas come from)
raw1 = load_source("train", "source1").set_index("entity_id")
raw23 = pd.concat([load_source("train", "source2"), load_source("train", "source3")]).set_index("entity_id")
rng = np.random.default_rng(0)
for sid in rng.choice(val_ids, 5, replace=False):
    r = raw1.loc[sid]
    print(f"\n=== {sid} [{r.country}] {r.business_name} | {r.business_address}")
    for cid in gt.loc[gt.s1_id == sid, "cand_id"]:
        c = raw23.loc[cid]
        print(f"   {cid[:2]} {c.business_name} | {c.business_address} [{c.country}]")

# %% Baselines on the validation split
empty = pd.DataFrame({"s1_id": pd.Series(dtype=str), "cand_id": pd.Series(dtype=str)})
score_report(empty, gt, val_ids, s1_country, name="predict-nothing")

pred_a = exact_key_match(s1n, candn, reject_num_conflict=False)
score_report(pred_a, gt, val_ids, s1_country, name="exact name_key")

pred_b = exact_key_match(s1n, candn, reject_num_conflict=True)
score_report(pred_b, gt, val_ids, s1_country, name="exact name_key + number check")

# %% Free the train data before test
del s1n, candn, raw1, raw23, gt, pred_a, pred_b
import gc; gc.collect()

# %% First end-to-end TEST submission with the best baseline
t1 = load_normalized("test", "source1", columns=cols)
tc = pd.concat([load_normalized("test", "source2", columns=cols),
                load_normalized("test", "source3", columns=cols)], ignore_index=True)
print("test countries:", t1["country"].value_counts().to_dict())

test_pred = exact_key_match(t1, tc, reject_num_conflict=True)
# For this baseline there is no separate model, so the candidate set equals the matches.
write_submission(test_pred, test_pred, t1["entity_id"])
print(f"S1 with >=1 match: {test_pred.s1_id.nunique() / len(t1):.3f}")
run_validator()
