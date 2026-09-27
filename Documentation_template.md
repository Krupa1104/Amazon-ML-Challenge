# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** The Jollibees  
**Team Members:** Jeevitha S, Krupa R, Preksha N  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We built a scalable two-stage entity-resolution pipeline: candidate generation (typed hashed
key blocking + character n-gram similarity search with FAISS + a learned pruning filter)
reduces ~1.7 × 10¹³ possible test pairs to **9.3M candidates (5.38 per Source 1 entity)**,
and a LightGBM pair model with a **one-owner assignment rule** (each S2/S3 record goes to at
most one S1 entity) makes precision-first decisions. Key innovations: rarity-weighted (IDF)
name/address features, phonetic and "glued-name" normalization for noisy Indian and web-style
names, test-like validation, and self-training for France (unseen in training).
**Final result: validation macro F0.5 0.9626, public leaderboard 0.952.**

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the training data (2,206,821 S1 entities; 10,320,219 S2+S3 records):

- **Every S2/S3 record belongs to at most one S1 entity** (0 records appear in two
  ground-truth lists) and ~26% of S2/S3 records belong to no S1 at all (distractors).
- **50.4% of S1 names are shared with another S1** in the same country (chains, generic
  names) → a name alone never decides a match; the address is essential.
- 5.6% of S1 entities are singletons; a matched S1 has ~3.5 matches (1–11), split roughly
  evenly between S2 and S3.
- **Test differs from train:** US 38% / India 47% / France 15% (France unseen), and 5.75
  S2/S3 records per S1 vs 4.68 in training.
- **Name noise:** legal-suffix variants (Pvt/Private, Ltd/Limited), typos, word-order swaps,
  DBA / "t/a" names, **phonetic re-spellings** ("praivet"=private, "inphra"=infra,
  "gujrat"=gujarat), **Devanagari names**, **glued / domain-style names** ("myagrocom" = My
  Agro), look-alike digits ("0rthopedic"), appended phone numbers, repeated words, renamed
  businesses (random trade names at the same address).
- **Address noise:** abbreviations, component reordering ("OH, Columbus, 5559 …"), **dropped
  leading digit of house numbers** (1025 vs 025), state name ↔ code (Texas/TX, West
  Bengal/WB, Bengali script), city aliases (Calcutta/Kolkata), prefixes ("Door No", "H.No",
  "#"), PO boxes, landmarks ("Near …", "Opp …", "B/H …"), missing and empty addresses.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — Blocking + similarity search + learned candidate filter + gradient-boosted classifier with an assignment constraint  
**Core Innovation:** Treating matching as *"which S1 entity, if any, owns this record?"*
(one-owner assignment) combined with a multi-signal candidate generator (10 typed key
families, two FAISS similarity indexes, and a learned filter) that keeps the candidate set
small (5.38 per S1) while retaining a validation oracle ceiling of 0.9765.

Pipeline:
```
raw TSVs → normalization (names, addresses, phonetic, glued, French fixes)
         → candidate generation: key blocking + similarity search → learned filter
           → candidate_pairs.tsv
         → pair features (63) → LightGBM (pooled + India, 3-seed ensemble;
           France: self-trained) → one-owner assignment + per-country thresholds
           → matching_results.tsv
```

**Validation design:** 20% of S1 entities held out by a stable hash of `entity_id` (whole
match sets stay on one side, no leakage); blocking and inference run over all S1 entities so
they compete for records as on test; metrics = official macro F0.5 on validation entities
only. To mimic the denser test set, 19% of training S1 entities are hidden (their records
become ownerless distractors, raising density from 4.68 to ~5.8 records per S1). Before each
upload we compared predicted matches per S1 on test vs validation to detect over-merging.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (hashed integers, country-scoped, built from each record's *rarest*
  tokens; tokens seen only once in the dataset are skipped):
  - rarest name tokens; the 2 rarest name tokens together ("anand + jain": common words,
    rare combination)
  - **phonetic skeleton** tokens (unify similar sounds, drop vowels) and pairs of them
  - rarest address words; rarest name token × each of the 4 rarest address words;
    phonetic token × address words
  - address word + last 2 digits of the house number (robust to the dropped leading digit)
  - full normalized name (unique names with empty addresses)
  - first 6 letters of the "glued" name (web / domain-style names, look-alike digits)
  - **Similarity search:** hashed character 3-grams → TF-IDF → TruncatedSVD (LSA) dense
    vectors; FAISS IVF index per country; every S2/S3 record retrieves its 5 nearest S1
    entities (name + address index) and 4 nearest (name-only index)
  - Each key family has its own block-size cap (20,000 pairs for single name/phonetic
    tokens, 5,000 for the others); pairs are generated with a vectorized sorted join in
    memory-bounded partitions
- **Pruning:** cheap score = max(0.7·name + 0.3·address, 0.3·name + 0.7·address) similarity;
  top 30 per S1 under a name-heavy *and* an address-heavy ranking, at most 4 S1 per record;
  then a **learned filter** (small LightGBM on blocking signals only) keeps ≥ 99.5% of the
  true pairs blocking found, with at most 5 candidates per S1 on validation. Its output is
  exactly what the matching model scores and is written to `candidate_pairs.tsv`.
- **Candidate pairs generated:** **9,323,951 on test** (1,732,544 S1 entities; **mean 5.38
  per S1**, median 5, p95 9; 99.8% of S1 entities have at least one candidate). Naive search
  space ≈ 1.73M × 9.96M ≈ 1.7 × 10¹³ pairs → reduction ratio ≈ 99.99995%. For reference, a
  matched S1 has ~3.5 true matches, so the model sees only ~2 extra candidates per entity.
- **How you ensured true matches were not lost:**
  - Several independent, complementary signals (exact tokens, phonetic forms, glued names,
    address + number, two similarity-search indexes), so a pair missed by one is caught by
    another; similarity search needs no shared exact token.
  - Measured on validation after every change: pair recall after the filter **0.936**,
    oracle macro F0.5 on the final candidates **0.9765** (the best score a perfect model
    could reach).
  - Error analysis classified every missed true pair (no shared key / block too large /
    cheap-score cut / per-record cut) and each dominant cause got a targeted key family
    (e.g. pair keys for common names, the house-number key, phonetic and glued keys).
  - Dual ranking in pruning so renamed businesses at an identical address are not crowded
    out by same-name chains; each record's 2 nearest neighbours always reach the filter.

---

## 4. Matching Model

**Features used** (63, computed in vectorized chunks with RapidFuzz):
- Name features: Levenshtein ratio, token-sort, token-set, partial ratio and Jaro-Winkler on
  the legal-suffix-stripped core name; token-set on the full name; DBA cross-match; exact
  sorted-key flag; first-word similarity (different person, same surname); **phonetic
  skeleton** ratio / token-set; **glued-name** ratio / partial; **rarity-weighted (IDF)**
  token Jaccard, rarity of the rarest shared token, rarity of the rarest token present on
  only one side.
- Address features: token-set, partial ratio, token-bag ratio, exact token-bag flag,
  address lengths (empty-address flag); **numbers:** shared digit groups, Jaccard, conflict
  flag, suffix compatibility (1025 vs 025), first and longest number compared (exact /
  truncated / conflict, edit distance), numbers present on only one side (the most
  important features); IDF-weighted address token overlap and missing rare address tokens.
- Other: country equality; source (S2 vs S3); name / address ambiguity (how many S1 entities
  share this name key / this exact address); competition context (rank and gap of the pair
  among all S1 entities proposed for the same record and among all records proposed for the
  same S1, number of competitors); similarity-search cosine and rank (both indexes); number
  of shared blocking keys. IDF statistics are computed per split on its own records.

**Model type:** LightGBM binary classifier (MIT licence), trained from scratch
(`learning_rate 0.08, num_leaves 127, min_child_samples 100, feature_fraction 0.8, bagging
0.8, lambda_l2 1`, up to 5,000 rounds, early stopping). A pooled model plus an India-only
model (each training country uses whichever scores better on its own validation entities);
3 seeds per model averaged (seed ensemble). France (no labels): **self-training** — the
ensemble's most confident French test decisions (761,558 positive and 568,240 negative
pseudo-labels, weight 0.7) retrain a pooled model that re-scores only French pairs.

**Threshold selection method:** macro F0.5 optimization on the validation set, **per
country** (US 0.65, India 0.675; countries unseen in training use the global validation
threshold 0.65), combined with the **one-owner rule**: each S2/S3 record is assigned only to
its highest-probability S1 entity, and only if that probability clears the threshold.
Entities with no candidate above the threshold get an empty list.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9626** on validation (US 0.9741, India 0.9453; pair
  precision 0.994, pair recall 0.912; singletons 0.98); **public leaderboard 0.952**.
  The ~0.01 gap is explained by the test country mix (47% India, our harder country) and
  France (unseen, ~0.92 estimated).
- **Common false positives (wrong merges):** neighbouring businesses on the same street
  with close house numbers ("2035 vs 2046 Cutter Point Dr", "6433 vs 6434"); same surname,
  different person ("Stephanie Seagraves" vs "Spencer Seagraves"); chains / generic names
  with nearly identical addresses. Addressed with house-number, one-sided-number, first-word
  and rarity features and the precision-first threshold.
- **Common false negatives (missed matches):** renamed businesses (random trade names) at an
  identical address; generic names with empty addresses (genuinely ambiguous among several
  S1 entities); true matches whose house numbers differ ("16858 vs 16856"); heavy phonetic
  re-spellings where neither name nor address shares a rare token. The remaining gap between
  the model (0.9626) and the candidate ceiling (0.9765) is mostly these ambiguous cases.

---

## 6. Conclusion
A scalable blocking + similarity search + learned-filter candidate generator (5.38
candidates per S1 out of ~10M possible) feeding a LightGBM matcher with a one-owner rule
reached 0.9626 macro F0.5 on validation and 0.952 on the public leaderboard, running
entirely on a 16 GB CPU laptop. The key lessons: feature and candidate-generation
improvements transferred to the test set, while density-dependent features (sibling counts)
and extra model complexity (deep trees, stacking) improved validation but not the
leaderboard; checking test-vs-validation prediction rates and building a test-like
validation set were essential.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:

| Path | Role |
|---|---|
| `src/normalize.py` | name / address normalization, phonetic skeleton, glued names, French fixes |
| `src/blocking.py` | typed hashed blocking keys, per-family caps, cheap scoring, pruning, kNN merge |
| `src/retrieval.py` | char n-gram TF-IDF → LSA → FAISS similarity search per country |
| `src/prefilter.py` | learned candidate filter (last candidate-generation stage) |
| `src/features.py` | pair features |
| `src/model.py` | LightGBM training, one-owner assignment, per-country thresholds |
| `src/metrics.py`, `src/split.py` | official macro F0.5, oracle ceiling; deterministic hash split |
| `src/io_utils.py`, `src/config.py` | loading with parquet cache, submission writers, paths |
| `src/stack.py`, `src/baseline.py`, `src/diagnostics.py` | earlier experiments, baseline, error analysis |

**Entry points** (run in order; each ends with the official validator):
1. `notebooks/04_pipeline.py` — normalization, similarity search, key blocking, learned filter
2. `notebooks/07_v9.py` — rarity features, French fixes, pooled + per-country models
3. `notebooks/09_v11_final.py` — seed ensemble + France self-training → writes
   `output/v11/matching_results.tsv` and `output/v11/candidate_pairs.tsv`

Exact setup and run instructions: `README.md`; pinned dependencies: `requirements.txt`
(pandas, NumPy, pyarrow, SciPy, scikit-learn, LightGBM, RapidFuzz, faiss-cpu, anyascii).
**Compliance:** no external data, APIs, geocoders or postal / gazetteer datasets; only small
hand-written normalization dictionaries (legal suffixes, street types, landmark words, US
state codes, ~25 Indian state codes, 8 historical city names, French street words); all
statistics computed on the provided records; no pretrained models (all models trained from
scratch; LightGBM is MIT-licensed); country treated as an open set.

### B. Additional Results

| Version | Main change | Validation macro F0.5 | Public LB |
|---|---|---|---|
| v1 | exact normalized-name baseline | 0.389 | 0.419 |
| v3 | typed key blocking + features + LightGBM + one-owner rule | 0.948 | 0.937 |
| v4 | full-name / glued keys, learned candidate filter | 0.953 | 0.944 |
| v5 | sibling-count features (removed: over-merged on test) | 0.958 | 0.916 |
| v6 | similarity search (LSA + FAISS) merged with blocking | 0.960 | 0.950 |
| v7 | deeper tuned trees + stacking (reverted) | 0.9635 | 0.946 |
| v9 | IDF features, French fixes, per-country models, test-like validation | 0.9624* | — |
| **v11** | **3-seed ensemble + France self-training** | **0.9626*** | **0.952** |

\* on the test-like validation set (19% of S1 hidden → more distractors).