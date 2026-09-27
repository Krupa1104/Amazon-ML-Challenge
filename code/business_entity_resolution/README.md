# Business Entity Resolution — Amazon ML Challenge 2026

Matches every Source 1 business to its Source 2 / Source 3 records with a scalable
candidate-generation stage (key blocking + character n-gram similarity search + a learned
pruning filter) followed by a LightGBM pair model and a one-owner assignment rule.
Final submission = **v11** (validation macro F0.5 0.9626, public leaderboard 0.952,
5.38 candidates per Source 1 entity on test).

## 1. Environment

- Python 3.12 (tested 3.12.4, Windows 11, 16 GB RAM, CPU only — no GPU needed)
- `pip install -r requirements.txt`
- No internet access is needed at run time; no external data, APIs or pretrained models are used.

## 2. Data layout

Place the challenge's `student_resource` folder next to this folder:

```
<parent>/
├── student_resource/
│   ├── dataset/train/  train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   ├── dataset/test/   test_source1.tsv, test_source2.tsv, test_source3.tsv
│   └── utils/validate_submission.py
└── business_entity_resolution/        <- this folder
```

Other locations work too: set `ER_DATA_DIR` (dataset folder) and `ER_UTILS_DIR` (utils folder).

## 3. Reproduce end-to-end (data → blocking → matching → output)

Run the three scripts **in this order**. Each is a Python file with `# %%` cells; run it
top-to-bottom (VS Code: "Run Below" on the first cell, or `python <file>.py` from inside
`notebooks/`). Every heavy step caches its result in `data_cache/` and every model in
`models/`, so an interrupted run resumes where it stopped.

| Step | Script | What it does | Time (16 GB laptop) |
|---|---|---|---|
| 1 | `notebooks/04_pipeline.py` (VERSION v8) | normalization (+ phonetic / glued variants), similarity search (2 indexes), key blocking, learned candidate filter, first model | ~3–4 h (first run includes normalization) |
| 2 | `notebooks/07_v9.py` | rarity (IDF) features, French text fixes, pooled + per-country models chosen on validation | ~3 h |
| 3 | `notebooks/09_v11_final.py` | 3-seed ensemble (kept only if better on validation) + self-training for countries unseen in training (France); writes the submission and runs the official validator | ~4–5 h |

Final files: `output/v11/matching_results.tsv` and `output/v11/candidate_pairs.tsv`.
Each script ends by running `student_resource/utils/validate_submission.py` on its output.

Notes
- Results reproduce up to small run-to-run differences (multi-threaded LightGBM / FAISS).
- `candidate_pairs.tsv` is exactly the set the matching model runs inference on (the output
  of the learned filter, the last candidate-generation stage); every match is a subset of it.
- Earlier experiment scripts are kept for transparency: `01_stage1_eda_baseline.py` (EDA +
  v1 baseline), `03_error_analysis.py` (diagnostics). They are not needed to reproduce v11.

## 4. Code map (`src/`)

| Module | Role |
|---|---|
| `config.py` | paths (overridable with `ER_*` environment variables) |
| `io_utils.py` | TSV loading with parquet cache, ground truth, submission writers, validator call |
| `normalize.py` | name / address normalization, phonetic skeleton, glued names, French fixes |
| `split.py` | deterministic hash split by Source 1 entity (20% validation); test-like distractor subset |
| `blocking.py` | typed hashed blocking keys with per-family caps, cheap scoring, pruning, kNN merge |
| `retrieval.py` | char n-gram TF-IDF → LSA vectors → FAISS nearest-neighbour search per country |
| `prefilter.py` | learned candidate filter (last candidate-generation stage) |
| `features.py` | pair features (fuzzy, numeric, rarity-weighted, competition context, similarity search) |
| `model.py` | LightGBM training, one-owner assignment, global / per-country threshold tuning |
| `metrics.py` | official macro F0.5 and blocking recall / oracle ceiling |
| `stack.py`, `baseline.py`, `diagnostics.py` | earlier experiments and error analysis |