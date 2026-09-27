"""Baseline 1: exact match on (country, name_key).

name_key = legal-suffix-stripped name tokens, sorted -> survives 'Pvt Ltd' vs 'Private Limited',
'&' vs 'and', punctuation, and word-order swaps. Typos are NOT handled (that's Stage 2).

Precision guards:
  * only S1 keys that are UNIQUE among Source 1 are used (a key shared by two S1
    entities is ambiguous, so we don't guess);
  * optional: reject a pair when both addresses have numbers and share none.
Because an S1 key is unique, every S2/S3 record is assigned to at most one S1 entity.
"""
import pandas as pd

KEY = ["country", "name_key"]


def _nums_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return True                      # missing info is not a conflict
    return not set(a.split()).isdisjoint(b.split())


def exact_key_match(s1n: pd.DataFrame, candn: pd.DataFrame, reject_num_conflict: bool = True) -> pd.DataFrame:
    """s1n / candn: normalized frames with entity_id, country, name_key, addr_nums.
    Returns pairs [s1_id, cand_id]."""
    s1 = s1n.loc[s1n["name_key"] != "", ["entity_id", *KEY, "addr_nums"]]
    s1 = s1.drop_duplicates(subset=KEY, keep=False)          # unique keys only
    cand = candn.loc[candn["name_key"] != "", ["entity_id", *KEY, "addr_nums"]]
    pairs = s1.merge(cand, on=KEY, suffixes=("_s1", "_c"))
    if reject_num_conflict and len(pairs):
        ok = [_nums_compatible(a, b) for a, b in zip(pairs["addr_nums_s1"].tolist(),
                                                     pairs["addr_nums_c"].tolist())]
        pairs = pairs[ok]
    return (pairs.rename(columns={"entity_id_s1": "s1_id", "entity_id_c": "cand_id"})
                 [["s1_id", "cand_id"]].reset_index(drop=True))
