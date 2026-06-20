"""
Stage 9: Evaluation & Analysis.

Loads the trained checkpoint, computes held-out loss/perplexity, generates
sample text under a couple of sampling strategies, and writes a DVC-trackable
metrics.json (so `dvc metrics show` / `dvc metrics diff` work out of the box)
plus logs the same metrics + generated samples to MLflow.

Usage:
    python src/evaluate.py
"""
import json
import math
import os
import sys

import mlflow
import torch
import yaml

sys.path.append(os.path.dirname(__file__))

from data_utils import get_batch, load_and_split, load_vocab, make_encode_decode, resolve_device
from model import GPTLanguageModel


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def init_tracking(params: dict):
    mlflow_cfg = params["mlflow"]
    try:
        import dagshub
        dagshub.init(
            repo_owner=mlflow_cfg["dagshub_repo_owner"],
            repo_name=mlflow_cfg["dagshub_repo_name"],
            mlflow=True,
        )
    except Exception as e:
        print(f"[evaluate] WARNING: dagshub.init failed ({e}); using tracking_uri/local mlruns.")
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])


@torch.no_grad()
def compute_eval_loss(model, val_data, num_batches, batch_size, block_size, device):
    model.eval()
    losses = torch.zeros(num_batches)
    for k in range(num_batches):
        X, Y = get_batch(val_data, batch_size, block_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    params = load_params()
    d_cfg, m_cfg, t_cfg = params["data"], params["model"], params["train"]
    g_cfg, e_cfg = params["generation"], params["evaluation"]

    device = resolve_device(t_cfg["device"])

    if not os.path.exists(t_cfg["checkpoint_path"]):
        raise FileNotFoundError(
            f"{t_cfg['checkpoint_path']} not found. Run `dvc repro train` first."
        )

    stoi, itos = load_vocab(t_cfg["vocab_path"])
    encode, decode = make_encode_decode(stoi, itos)
    _, val_data = load_and_split(d_cfg["raw_path"], d_cfg["train_split"], encode)

    ckpt = torch.load(t_cfg["checkpoint_path"], map_location=device)
    model = GPTLanguageModel(
        vocab_size=ckpt["vocab_size"],
        n_embd=m_cfg["n_embd"],
        n_head=m_cfg["n_head"],
        n_layer=m_cfg["n_layer"],
        block_size=m_cfg["block_size"],
        dropout=m_cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_loss = compute_eval_loss(
        model, val_data, e_cfg["num_eval_batches"], t_cfg["batch_size"], m_cfg["block_size"], device
    )
    perplexity = math.exp(val_loss)
    print(f"[evaluate] val_loss={val_loss:.4f}  perplexity={perplexity:.2f}")

    os.makedirs(os.path.dirname(g_cfg["sample_output_path"]), exist_ok=True)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    samples = {}
    sampling_strategies = {
        "greedy_like_low_temp": {"temperature": 0.3, "top_k": None},
        "default_temp1": {"temperature": 1.0, "top_k": None},
        "top_k_40": {"temperature": 1.0, "top_k": 40},
    }
    for name, kwargs in sampling_strategies.items():
        out_idx = model.generate(context, max_new_tokens=g_cfg["max_new_tokens"], **kwargs)
        samples[name] = decode(out_idx[0].tolist())

    with open(g_cfg["sample_output_path"], "w", encoding="utf-8") as f:
        for name, text in samples.items():
            f.write(f"=== {name} ===\n{text}\n\n")
    print(f"[evaluate] Wrote generated samples to {g_cfg['sample_output_path']}")

    metrics = {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with open(e_cfg["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Wrote metrics to {e_cfg['metrics_path']}")

    init_tracking(params)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(g_cfg["sample_output_path"])
        mlflow.log_artifact(e_cfg["metrics_path"])


if __name__ == "__main__":
    main()
