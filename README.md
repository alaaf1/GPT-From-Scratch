# GPT From Scratch — Lab 2 (NLP Week 2)

A decoder-only, char-level GPT built from first principles in PyTorch
(no `nn.Transformer` / `nn.MultiheadAttention`), wired into a DVC pipeline
with MLflow experiment tracking via DagsHub.

## Pipeline stages (`dvc.yaml`)

1. **download_data** — `src/data_download.py` fetches the Tiny Shakespeare
   corpus to `data/input.txt` (DVC-tracked, only re-downloaded if `data.url`
   in `params.yaml` changes).
2. **train** — `src/train.py` builds the char vocab, trains the GPT
   (`src/model.py`), logs params/metrics/checkpoint to MLflow each
   `eval_interval` steps, and saves `models/gpt_model.pt` + `models/vocab.json`.
3. **evaluate** — `src/evaluate.py` reloads the checkpoint, computes held-out
   loss + perplexity, generates samples under three sampling strategies
   (low-temperature, temperature=1, top-k=40), and writes `metrics.json`
   (DVC metrics file) and `outputs/generated_samples.txt`.

All hyperparameters live in `params.yaml` — nothing is hardcoded in the
scripts, so `dvc repro` and `dvc exp run --set-param model.n_layer=8` work
without touching code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or use uv
pip install -r requirements.txt

# Initialize dvc + git tracking for this repo (first time only)
git init
dvc init
```

## DagsHub / MLflow configuration

`params.yaml` -> `mlflow:` points at a DagsHub-hosted MLflow tracking server.
Set credentials before running (DagsHub token, *not* your password):

```bash
export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

If `dagshub.init(...)` can't authenticate, both `train.py` and `evaluate.py`
fall back to the plain `mlflow.set_tracking_uri(...)` / local `./mlruns`,
so the pipeline still runs end-to-end offline.

## Running the pipeline

```bash
dvc repro              # runs download_data -> train -> evaluate
dvc metrics show       # prints metrics.json (val_loss, perplexity, n_params)
dvc metrics diff       # compares metrics across commits/experiments
```

To version raw data and model artifacts with DVC remote storage:

```bash
dvc remote add -d origin <your-dagshub-dvc-remote-url>
dvc push
```

## Running an experiment sweep

```bash
dvc exp run --set-param model.n_layer=8 --set-param train.max_iters=3000
dvc exp show
```

Each `dvc exp run` is also logged as a separate MLflow run under the
`gpt-from-scratch` experiment, so DVC param/metric history and MLflow run
history stay in sync.

## Project layout

```
.
├── params.yaml                  # single source of truth for all configs
├── dvc.yaml                     # pipeline DAG (3 stages)
├── requirements.txt
├── data/
│   └── input.txt                # downloaded by stage 1 (gitignored, dvc-tracked)
├── models/
│   ├── gpt_model.pt              # checkpoint (gitignored, dvc-tracked)
│   └── vocab.json                # char-to-int mapping
├── outputs/
│   └── generated_samples.txt     # sample generations per sampling strategy
├── metrics.json                  # val_loss / perplexity / n_params
└── src/
    ├── data_download.py          # Stage 1: data acquisition
    ├── data_utils.py             # Stage 1: tokenizer + batching
    ├── model.py                  # Stages 2-6: embeddings -> attention -> GPT
    ├── train.py                  # Stage 7: training + MLflow logging
    └── evaluate.py                # Stage 9: eval loss, perplexity, generation
```

## Notes on Stage 9 (Evaluation & Analysis)

- **Quantitative**: held-out cross-entropy loss and perplexity (`exp(loss)`),
  written to `metrics.json` for DVC metrics tracking.
- **Qualitative**: text is generated at three settings — low temperature
  (near-greedy), temperature=1 (matches training distribution), and top-k=40 —
  so generation quality/diversity trade-offs can be compared in
  `outputs/generated_samples.txt`.
- Both are also logged as MLflow artifacts/metrics under an `evaluation` run,
  separate from the training run, so they can be diffed across experiments.
