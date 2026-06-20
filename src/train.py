"""
Stage 7: Training.

Trains the GPT model defined in model.py on the char-level corpus, logging
every run (params, metrics, checkpoint artifact) to MLflow, with the
tracking server pointed at DagsHub so experiments are versioned alongside
the DVC-tracked data/model artifacts.

Usage:
    python src/train.py
"""
import os
import sys

import mlflow
import torch
import yaml

from data_utils import (
    build_vocab,
    get_batch,
    load_and_split,
    make_encode_decode,
    resolve_device,
    save_vocab,
)
from model import GPTLanguageModel

sys.path.append(os.path.dirname(__file__))


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def init_tracking(params: dict):
    """Point MLflow at DagsHub. Falls back to local ./mlruns if DagsHub
    credentials aren't configured, so the pipeline still runs offline."""
    mlflow_cfg = params["mlflow"]
    try:
        import dagshub
        dagshub.init(
            repo_owner=mlflow_cfg["dagshub_repo_owner"],
            repo_name=mlflow_cfg["dagshub_repo_name"],
            mlflow=True,
        )
        print("[train] DagsHub MLflow tracking initialized.")
    except Exception as e:
        print(f"[train] WARNING: dagshub.init failed ({e}); "
              f"falling back to tracking_uri from params.yaml / local mlruns.")
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(mlflow_cfg["experiment_name"])


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    params = load_params()
    d_cfg, m_cfg, t_cfg = params["data"], params["model"], params["train"]

    device = resolve_device(t_cfg["device"])
    torch.manual_seed(t_cfg["seed"])
    print(f"[train] Using device: {device}")

    if not os.path.exists(d_cfg["raw_path"]):
        raise FileNotFoundError(
            f"{d_cfg['raw_path']} not found. Run `dvc repro download_data` "
            f"(or `python src/data_download.py`) first."
        )

    with open(d_cfg["raw_path"], "r", encoding="utf-8") as f:
        text = f.read()
    chars, stoi, itos = build_vocab(text)
    vocab_size = len(chars)
    encode, decode = make_encode_decode(stoi, itos)
    save_vocab(stoi, t_cfg["vocab_path"])

    train_data, val_data = load_and_split(d_cfg["raw_path"], d_cfg["train_split"], encode)
    print(f"[train] vocab_size={vocab_size}, train_tokens={len(train_data)}, val_tokens={len(val_data)}")

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=m_cfg["n_embd"],
        n_head=m_cfg["n_head"],
        n_layer=m_cfg["n_layer"],
        block_size=m_cfg["block_size"],
        dropout=m_cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {n_params / 1e6:.2f} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t_cfg["learning_rate"]))

    init_tracking(params)

    with mlflow.start_run():
        mlflow.log_params({**d_cfg, **m_cfg, **t_cfg, "vocab_size": vocab_size, "n_params": n_params})

        final_losses = {"train": None, "val": None}
        for it in range(t_cfg["max_iters"]):
            if it % t_cfg["eval_interval"] == 0 or it == t_cfg["max_iters"] - 1:
                losses = estimate_loss(
                    model, train_data, val_data,
                    t_cfg["eval_iters"], t_cfg["batch_size"], m_cfg["block_size"], device,
                )
                print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                mlflow.log_metrics(
                    {"train_loss": losses["train"], "val_loss": losses["val"]}, step=it
                )
                final_losses = losses

            xb, yb = get_batch(train_data, t_cfg["batch_size"], m_cfg["block_size"], device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "model_config": m_cfg,
            },
            t_cfg["checkpoint_path"],
        )
        mlflow.log_artifact(t_cfg["checkpoint_path"])
        mlflow.log_artifact(t_cfg["vocab_path"])
        mlflow.log_metrics(
            {"final_train_loss": final_losses["train"], "final_val_loss": final_losses["val"]}
        )

        print(f"[train] Saved checkpoint to {t_cfg['checkpoint_path']}")


if __name__ == "__main__":
    main()
