# Studienarbeit Eval Harness

Minimal, reproducible evaluation harness for the three-arm context-strategy study:
**No-Context vs. Full-Context-Stuffing vs. RAG**, measured on answer quality (vs. fixed gold)
and cost/latency. The research object is the *evaluation* — there is no served API here, just
dataset loading, a 3-arm experiment runner, LLM-judged + deterministic scoring, predefined
statistics, and Excel/plot output.

The experiment runs on **PubMedQA** (`pqa_labeled`): yes/no/maybe questions over pooled
abstract sections, scored by label accuracy.

## The experimental control

Identical model, identical prompt template, identical questions — *only* the text in the
`{context_block}` slot differs across arms (`src/sae/arms/arms.py`):

| Arm | Context |
| --- | --- |
| `no_context` | empty — parametric knowledge only |
| `full_context` | every paragraph in the record's corpus, no selection |
| `rag` | top-`k` chunks from `src/sae/retrieval/retriever.py` (cosine search over cached, normalized embeddings) |

## Design decisions baked in

- **Judge ≠ generator** — enforced in `config.load_config`; prevents self-preference bias.
  Loading also fails on an unpinned `"REPLACE"` model.
- **PubMedQA → label accuracy** (`metrics/accuracy.py`), not RAGAS `answer_correctness`:
  the gold is a categorical `yes`/`no`/`maybe`, and the free-text gold (`long_answer`, the
  abstract's conclusion) is deliberately excluded from the context. `em`/`f1` are still written
  to the Parquet but are structurally near-zero here and are not reported.
- **Two faithfulness columns.** `faithfulness` is scored against the fixed full corpus
  (identical across arms, so `no_context` is scorable on the same yardstick) — a cross-arm
  upper bound. `faithfulness_retrieved` is the strict per-arm measure, scored against the
  context the arm actually saw; it is `NaN` for `no_context` by design.
- **Corpus size is an independent variable** (`experiment/runner.py::resize_kb`): the paragraph
  pool is trimmed/padded to `kb_size` while gold paragraphs are always preserved (tracked by
  identity through the shuffle, robust to duplicate paragraph text). `retrieval.k` is a separate
  fixed value, so Full-Context and RAG deliberately see different text volumes — the
  efficiency/quality tradeoff under study.
- **Predefined statistics** (`stats/tests.py`), fixed before running: paired Wilcoxon
  signed-rank + rank-biserial effect size + bootstrap CI, always per fixed `kb_size`,
  Holm-corrected across the metric × arm-pair family. The KB-size hypotheses H3 (within-arm
  trend) and H4 (arm × size interaction) require a sweep with ≥ 2 sizes; the shipped configs
  use a single fixed size, so those tests do not apply to that run.

## Layout

```
config/                 pinned models, temps, seeds, k, N, corpus size
src/sae/data/           PubMedQA loader -> unified Record schema
src/sae/retrieval/      chunking + embedding + top-k
src/sae/arms/           the 3 arms behind one shared prompt
src/sae/llm/            provider-agnostic client (litellm) + token/TTFT/latency + retries
src/sae/metrics/        label accuracy, RAGAS, EM/F1, retrieval diagnostics
src/sae/experiment/     runner (generation) + scoring (metrics) + persistence
src/sae/reporting/      formatted .xlsx export (one sheet per arm)
src/sae/stats/          paired tests, effect sizes, bootstrap CIs, Holm
scripts/                run_experiment / analyze / export_analysis / rescore
tests/                  offline verification (no API key needed)
```

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env        # then fill in real provider keys
```

Provider keys live in `.env` (gitignored), loaded automatically by `config.load_config` via
`python-dotenv`. litellm picks the key by model prefix, so the pinned defaults need:

- `GEMINI_API_KEY` — generator `gemini/gemini-3.5-flash-lite`
- `ANTHROPIC_API_KEY` — judge `anthropic/claude-sonnet-5`

## Configs

| File | Purpose |
| --- | --- |
| `config/default.yaml` | defaults; PubMedQA at its native pooled corpus size (120) |
| `config/pubmedqa_test.yaml` | smoke test — 10 questions × 1 sample, run with `--no-sweep` |
| `config/pubmedqa_verify.yaml` | verification gate — 15 questions × 1 sample at the real `kb_size` |
| `config/pubmedqa_run.yaml` | the full run — 200 questions × 3 arms × 3 samples = 1800 generations |

`pubmedqa_verify.yaml` and `pubmedqa_run.yaml` pin `kb_size_sweep: [100]` and must be run
**without** `--no-sweep`, so every question is resized to the same corpus: PubMedQA's pool grows
with N, and the native size would make Full-Context intractable at N=200.

## Workflow

```bash
pytest -q                                                              # 1. offline logic check
ruff check src tests scripts
python scripts/run_experiment.py --config config/pubmedqa_test.yaml --no-sweep   # 2. smoke test
python scripts/run_experiment.py --config config/pubmedqa_verify.yaml            # 3. gate
python scripts/run_experiment.py --config config/pubmedqa_run.yaml               # 4. full run
python scripts/analyze.py --run results/run_pubmedqa.parquet                     # 5. stats + figures
python scripts/export_analysis.py --run results/run_pubmedqa.parquet             # 6. analysis .xlsx
```

`pytest` is fully offline — no API key, network, or model download. Everything from step 2 on
makes real provider calls. Budget generously: Full-Context is ~40 s per call at a 100-section
corpus (vs. ~2.5 s for No-Context/RAG), so the full run is an overnight job.

## Outputs

Every run writes two artifacts (`experiment/scoring.py::save_run`) into `results/` (gitignored):

- `run_<name>.parquet` — full raw source of truth: contexts, retrieved indices, full corpus,
  all metrics.
- `answers_<name>.xlsx` — curated human-readable workbook, **one sheet per arm**. The column
  list is a static allowlist in `reporting/excel_export.py::REPORT_COLS`, so a new metric
  column elsewhere in the pipeline will not appear here until it is added there too.

`analyze.py` additionally writes tables/CSVs and figures to `results/figures/`;
`export_analysis.py` turns those into a single formatted workbook and computes nothing of its
own (`analyze.py` stays the single source of truth).

To backfill improved metrics onto an existing run without regenerating answers:

```bash
python scripts/rescore.py --run results/run_pubmedqa.parquet --dataset pubmedqa
```

It recomputes the judge-free label columns and the strict `faithfulness_retrieved` (one
faithfulness judge pass), leaves the other RAGAS scores untouched, and writes a
non-destructive `*_rescored` Parquet + Excel. The strict pass is gated by
`judge.strict_faithfulness` in the config (off in `default.yaml` to save judge cost on the
full run).
