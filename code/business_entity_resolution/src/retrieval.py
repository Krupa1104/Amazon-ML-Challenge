"""Similarity-search candidate generation (v6).

Key-based blocking needs an exact shared token; it misses pairs whose names are similar but
spelled differently ('marathwada po1ymar' ~ 'marathwada polymer') when addresses also differ.
Here every record becomes a dense vector of its character patterns and each S2/S3 record
searches its k nearest Source 1 entities (a record has at most one owner, so a handful of
neighbours per record is enough and the candidate set stays small).

  text      name_core + dba  |  normalized address       (separate TF-IDF spaces, weighted)
  features  hashed char 3-grams within words (typos, glued words, phonetic spellings)
  vectors   TF-IDF -> TruncatedSVD (LSA) to `dim` dims, L2-normalized (cosine = dot product)
  search    FAISS IVF (inner product), one index per country, nprobe ~16

All CPU; memory-bounded by encoding records in chunks. No external data, no pretrained model.
"""
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


class Embedder:
    def __init__(self, dim: int = 96, n_features: int = 2 ** 18, w_name: float = 0.8, w_addr: float = 0.6,
                 seed: int = 0):
        self.dim, self.w_name, self.w_addr, self.seed = dim, w_name, w_addr, seed
        self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=n_features,
                                    alternate_sign=False, norm=None, dtype=np.float32)

    @staticmethod
    def texts(df: pd.DataFrame):
        names = [f"{a} {b}".strip() for a, b in zip(df["name_core"].tolist(), df["name_dba"].tolist())]
        return names, df["addr_norm"].tolist()

    def _sparse(self, names, addrs):
        xn = normalize(self.tf_n.transform(self.hv.transform(names)))
        if getattr(self, "w_addr", 0) == 0:          # name-only embedder
            return (self.w_name * xn).tocsr()
        xa = normalize(self.tf_a.transform(self.hv.transform(addrs)))
        return sp.hstack([self.w_name * xn, self.w_addr * xa], format="csr")

    def fit(self, frames, sample: int = 400_000):
        rng = np.random.default_rng(self.seed)
        parts = []
        for df in frames:
            take = min(len(df), sample // len(frames))
            parts.append(df.iloc[np.sort(rng.choice(len(df), take, replace=False))])
        names, addrs = self.texts(pd.concat(parts, ignore_index=True))
        self.tf_n = TfidfTransformer(sublinear_tf=True).fit(self.hv.transform(names))
        self.tf_a = TfidfTransformer(sublinear_tf=True).fit(self.hv.transform(addrs))
        self.svd = TruncatedSVD(self.dim, algorithm="randomized", n_iter=5, random_state=self.seed)
        self.svd.fit(self._sparse(names, addrs))
        return self

    def transform(self, df: pd.DataFrame, chunk: int = 250_000) -> np.ndarray:
        out = np.empty((len(df), self.dim), np.float32)
        for b in range(0, len(df), chunk):
            names, addrs = self.texts(df.iloc[b:b + chunk])
            v = self.svd.transform(self._sparse(names, addrs)).astype(np.float32)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            out[b:b + chunk] = v / np.maximum(norms, 1e-6)
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def _index(xb: np.ndarray, nprobe: int, seed: int = 0):
    import faiss
    d = xb.shape[1]
    if len(xb) < 50_000:
        idx = faiss.IndexFlatIP(d)
    else:
        nlist = int(np.clip(4 * np.sqrt(len(xb)), 256, 8192))
        idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        train = xb[rng.choice(len(xb), min(len(xb), 60 * nlist), replace=False)]
        idx.train(train)
        idx.nprobe = nprobe
    idx.add(xb)
    return idx


def knn_candidates(emb: Embedder, s1n: pd.DataFrame, candn: pd.DataFrame, k: int = 5, nprobe: int = 16,
                   chunk: int = 500_000, verbose: bool = True) -> dict:
    """For every S2/S3 record: its k most similar Source 1 entities (same country).
    Returns dict: s1, cand (int codes), sim (cosine), rank (0 = nearest)."""
    log = print if verbose else (lambda *a, **kw: None)
    s1_cty, c_cty = s1n["country"].to_numpy(), candn["country"].to_numpy()
    res = {"s1": [], "cand": [], "sim": [], "rank": []}
    for cty in pd.unique(c_cty):
        s1_rows = np.flatnonzero(s1_cty == cty)
        c_rows = np.flatnonzero(c_cty == cty)
        if len(s1_rows) == 0:
            continue
        log(f"  [{cty}] encoding {len(s1_rows):,} S1 ...")
        idx = _index(emb.transform(s1n.iloc[s1_rows]), nprobe)
        kk = min(k, len(s1_rows))
        for b in range(0, len(c_rows), chunk):
            rows = c_rows[b:b + chunk]
            D, I = idx.search(emb.transform(candn.iloc[rows]), kk)
            ok = I >= 0
            res["s1"].append(s1_rows[I[ok]])
            res["cand"].append(np.repeat(rows, kk).reshape(-1, kk)[ok])
            res["sim"].append(D[ok].astype(np.float32))
            res["rank"].append(np.tile(np.arange(kk, dtype=np.int8), (len(rows), 1))[ok])
            log(f"  [{cty}] searched {min(b + chunk, len(c_rows)):,}/{len(c_rows):,} records", end="\r")
        log()
        del idx
    return {k_: np.concatenate(v).astype(np.int64 if k_ in ("s1", "cand") else v[0].dtype)
            for k_, v in res.items()}


def merge_knn(runs: dict, n_cand: int) -> dict:
    """Union of several kNN runs, e.g. {"knn": combined, "knnn": name-only}.
    Output fields: s1, cand, and <prefix>_sim / <prefix>_rank per run (-1 where absent)."""
    pcs = {p: r["s1"].astype(np.int64) * n_cand + r["cand"] for p, r in runs.items()}
    allpc = np.unique(np.concatenate(list(pcs.values())))
    out = {"s1": (allpc // n_cand).astype(np.int32), "cand": (allpc % n_cand).astype(np.int32)}
    for p, r in runs.items():
        pos = np.searchsorted(allpc, pcs[p])
        sim = np.full(len(allpc), -1, np.float32); rank = np.full(len(allpc), -1, np.int8)
        sim[pos], rank[pos] = r["sim"], r["rank"]
        out[f"{p}_sim"], out[f"{p}_rank"] = sim, rank
    return out
