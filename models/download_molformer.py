"""
download_molformer.py — pulls MoLFormer-XL-both-10pct into models/ locally.

Usage:
    python download_molformer.py
"""

import os
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ID = "ibm/MoLFormer-XL-both-10pct"
LOCAL_DIR = "models/molformer-xl-both-10pct"


def download():
    print(f"Downloading {REPO_ID} -> {LOCAL_DIR} ...")
    path = snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR)
    print(f"Downloaded to: {path}")
    return path


def verify(local_dir):
    print("\nVerifying files...")
    required = [
        "config.json",
        "configuration_molformer.py",
        "modeling_molformer.py",
        "tokenization_molformer.py",
        "tokenization_molformer_fast.py",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(local_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing expected files after download: {missing}")

    weight_file = os.path.join(local_dir, "model.safetensors")
    if not os.path.exists(weight_file):
        raise FileNotFoundError("model.safetensors not found — download likely incomplete.")

    size_mb = os.path.getsize(weight_file) / 1e6
    print(f"  model.safetensors: {size_mb:.1f} MB")
    if size_mb < 100:
        raise RuntimeError(f"model.safetensors looks too small ({size_mb:.1f} MB) — re-download.")
    print("  All required files present.")


def smoke_test(local_dir):
    print("\nRunning smoke test (load + forward pass)...")
    model = AutoModel.from_pretrained(
        local_dir,
        deterministic_eval=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        trust_remote_code=True,
    )

    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
    inputs = tokenizer(smiles, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"  pooler_output shape:      {tuple(outputs.pooler_output.shape)}")
    print(f"  last_hidden_state shape:  {tuple(outputs.last_hidden_state.shape)}")

    assert outputs.pooler_output.shape[-1] == 768, "Unexpected hidden size."
    print("  Smoke test passed.")


if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    local_dir = download()
    verify(local_dir)
    smoke_test(local_dir)
    print(f"\nMoLFormer-XL is ready at: {local_dir}")
    print('Add to config.py:  molformer_model_path: str = "models/molformer-xl-both-10pct"')
    print('Set once per shell before training:  export HF_HUB_OFFLINE=1')