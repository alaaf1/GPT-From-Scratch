"""
Stage 1 (tokenizer) + data loading helpers shared by train.py / evaluate.py.

Implements a simple char-level tokenizer (no subword/BPE library used,
in keeping with the "build it yourself" spirit of the lab).
"""
import json
import os

import torch


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def save_vocab(stoi: dict, vocab_path: str):
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(stoi, f)


def load_vocab(vocab_path: str):
    with open(vocab_path, "r") as f:
        stoi = json.load(f)
    itos = {int(i): ch for ch, i in stoi.items()}
    return stoi, itos


def make_encode_decode(stoi: dict, itos: dict):
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return encode, decode


def load_and_split(raw_path: str, train_split: float, encode):
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg
