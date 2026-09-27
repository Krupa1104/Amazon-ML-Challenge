"""Deterministic train/val split of Source 1 entities.

Uses a stable hash of the entity_id, so every teammate gets the IDENTICAL split on any
machine without sharing a file. Because each S2/S3 record belongs to at most one S1,
splitting by S1 keeps every match set entirely on one side (no leakage).

Evaluation protocol (mimics the test set):
  run blocking + inference over ALL Source 1 entities (so they compete for records),
  train models only on pairs whose s1_id is in 'train', report metrics only on 'val'.
"""
import numpy as np
import pandas as pd

from .config import VAL_PERCENT


def hash_bucket(ids, n: int = 100) -> np.ndarray:
    h = pd.util.hash_pandas_object(pd.Series(np.asarray(ids, dtype=object)), index=False)
    return (h.to_numpy() % n).astype(int)


def split_ids(s1_ids, val_percent: int = VAL_PERCENT) -> pd.Series:
    """Returns Series indexed by s1_id with values 'train' / 'val'."""
    ids = np.asarray(s1_ids, dtype=object)
    fold = np.where(hash_bucket(ids) < val_percent, "val", "train")
    return pd.Series(fold, index=pd.Index(ids, name="s1_id"), name="fold")


def subsample(ids, percent: int = 10) -> np.ndarray:
    """Stable `percent`% subsample of any id list (e.g. the val ids) for fast experiments.
    Uses different hash digits than the split, so it is an unbiased sample of `ids`.
    Keep the SEARCH POOL (all S2/S3) full so difficulty stays realistic."""
    ids = np.asarray(ids, dtype=object)
    h = pd.util.hash_pandas_object(pd.Series(ids), index=False).to_numpy()
    return ids[((h // 100) % 100) < percent]


def drop_mask(ids, percent: int) -> np.ndarray:
    """True for entities to KEEP. Drops a stable `percent`% of Source 1 entities (hash digits
    independent of the validation split). Their S2/S3 records stay in the data as ownerless
    distractors, which raises the records-per-S1 density to match the test set."""
    ids = np.asarray(ids, dtype=object)
    h = pd.util.hash_pandas_object(pd.Series(ids), index=False).to_numpy()
    return ((h // 10_000) % 100) >= percent