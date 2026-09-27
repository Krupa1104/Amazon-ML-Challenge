"""Loading raw data (with parquet caching), ground truth, and writing submission files."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, DATA_DIR, OUTPUT_DIR, UTILS_DIR

RAW_COLS = ["entity_id", "business_name", "business_address", "country"]


def _read_tsv(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: a business literally named "NA" stays "NA", not NaN.
    # QUOTE_NONE: fields are split strictly on tabs; stray quote characters stay in the text.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def load_source(split: str, source: str, columns=None) -> pd.DataFrame:
    """split: 'train' | 'test';  source: 'source1' | 'source2' | 'source3'.
    First call reads the TSV and caches it as parquet; later calls take seconds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pq = CACHE_DIR / f"raw_{split}_{source}.parquet"
    if not pq.exists():
        df = _read_tsv(DATA_DIR / split / f"{split}_{source}.tsv")
        df.to_parquet(pq, index=False)
        del df
    return pd.read_parquet(pq, columns=columns, dtype_backend="pyarrow")


def load_gt_pairs() -> pd.DataFrame:
    """Ground truth as a long table of true pairs: columns [s1_id, cand_id].
    Singletons (S1 with no matches) simply have no rows here."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pq = CACHE_DIR / "gt_pairs.parquet"
    if not pq.exists():
        gt = _read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
        gt = gt[gt["matched_entity_ids"] != ""]
        pairs = (gt.assign(cand_id=gt["matched_entity_ids"].str.split(","))
                   .explode("cand_id")[["source1_entity_id", "cand_id"]]
                   .rename(columns={"source1_entity_id": "s1_id"}))
        pairs["cand_id"] = pairs["cand_id"].str.strip()
        pairs = pairs[pairs["cand_id"] != ""].drop_duplicates().reset_index(drop=True)
        pairs.to_parquet(pq, index=False)
    return pd.read_parquet(pq, dtype_backend="pyarrow")


def write_id_lists(pairs: pd.DataFrame, all_s1_ids, path: Path, list_col: str) -> Path:
    """Write one row per Source 1 entity with a comma-joined, de-duplicated ID list.
    pairs: DataFrame with columns [s1_id, cand_id]. Entities without pairs get an empty list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = pairs[["s1_id", "cand_id"]].drop_duplicates()
    lists = p.groupby("s1_id", sort=False)["cand_id"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": pd.Index(all_s1_ids).unique()})
    out[list_col] = out["source1_entity_id"].map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return path


def write_submission(match_pairs: pd.DataFrame, cand_pairs: pd.DataFrame, all_s1_ids,
                     out_dir: Path = OUTPUT_DIR):
    """Writes output/matching_results.tsv and output/candidate_pairs.tsv.
    Matches are unioned into candidates so matches are always a subset of candidates."""
    cand_pairs = pd.concat([cand_pairs[["s1_id", "cand_id"]], match_pairs[["s1_id", "cand_id"]]])
    m = write_id_lists(match_pairs, all_s1_ids, Path(out_dir) / "matching_results.tsv", "matched_entity_ids")
    c = write_id_lists(cand_pairs, all_s1_ids, Path(out_dir) / "candidate_pairs.tsv", "candidate_entity_ids")
    return m, c


def run_validator(out_dir: Path = OUTPUT_DIR) -> int:
    """Runs the official utils/validate_submission.py and prints its report."""
    cmd = [sys.executable, str(UTILS_DIR / "validate_submission.py"),
           "--matching", str(Path(out_dir) / "matching_results.tsv"),
           "--candidate", str(Path(out_dir) / "candidate_pairs.tsv"),
           "--test-dir", str(DATA_DIR / "test")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-5000:], res.stderr[-2000:])
    return res.returncode


def write_lists_from_codes(s1_codes, cand_codes, s1_ids, cand_ids, path: Path, list_col: str) -> Path:
    """Fast writer for millions of pairs given integer codes (rows of s1_ids / cand_ids)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s1_ids = np.asarray(s1_ids, dtype=object)
    cand_ids = np.asarray(cand_ids, dtype=object)
    pc = np.unique(np.asarray(s1_codes, np.int64) * len(cand_ids) + np.asarray(cand_codes, np.int64))
    s1c, cc = pc // len(cand_ids), pc % len(cand_ids)          # sorted by s1 code
    bounds = np.searchsorted(s1c, np.arange(len(s1_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{list_col}\n")
        for k, sid in enumerate(s1_ids):
            a, b = bounds[k], bounds[k + 1]
            f.write(f"{sid}\t{','.join(cand_ids[cc[a:b]])}\n")
    return path


def write_submission_codes(match_s1, match_c, cand_s1, cand_c, s1_ids, cand_ids, out_dir: Path = OUTPUT_DIR):
    """Matches are unioned into candidates so they are always a subset."""
    cs1 = np.concatenate([cand_s1, match_s1]); cc = np.concatenate([cand_c, match_c])
    m = write_lists_from_codes(match_s1, match_c, s1_ids, cand_ids, Path(out_dir) / "matching_results.tsv", "matched_entity_ids")
    c = write_lists_from_codes(cs1, cc, s1_ids, cand_ids, Path(out_dir) / "candidate_pairs.tsv", "candidate_entity_ids")
    return m, c
