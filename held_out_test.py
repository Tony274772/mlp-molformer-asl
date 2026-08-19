"""
held_out_test.py

Run the trained API-Excipient compatibility model on the 24-pair held-out
test set and write predictions (logit, probability, prediction) to a CSV
in outputs/.

Expects, in outputs/:
    - held_out_test_set.csv     (Api_name, Excipient_name, API_CID, Excipient_CID,
                                  API_Smiles, Excipient_Smiles, ground_truth)
    - api_descriptors.csv       (RDKit descriptors for the unique APIs)
    - excipient_descriptors.csv (RDKit descriptors for the unique excipients)

Writes:
    outputs/held_out_test_predictions.csv
        original columns + logit, probability, prediction
        (+ correct, if ground_truth is present)

Usage:
    python held_out_test.py
    python held_out_test.py --threshold 0.5
    python held_out_test.py --checkpoint checkpoints/molformer/cv_fold_1/best_model.pt
"""

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
import torch

from src.runtime import configure_thread_limits, configure_torch_runtime
configure_thread_limits()
configure_torch_runtime(torch)

from src.config import Config
from src.utils import set_seed
from src.molformer_featurization import MolFormerFeaturizer
from src.dataset import get_dataloader_from_dataframe
from src.model import APIExcipientModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_threshold(explicit_threshold, config):
    """Use CLI threshold if given, else the tuned threshold saved during
    training (metrics/molformer/run_metrics.json), else fall back to 0.5."""
    if explicit_threshold is not None:
        return explicit_threshold

    metrics_path = os.path.join(ROOT, config.metrics_dir, "run_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        thresh = payload.get("threshold")
        if thresh is not None:
            logging.info(f"Using tuned threshold from {metrics_path}: {thresh:.3f}")
            return float(thresh)

    logging.warning("No tuned threshold found; defaulting to 0.5.")
    return 0.5


def main():
    parser = argparse.ArgumentParser(description="Run inference on the held-out test set.")
    parser.add_argument(
        "--input", default=os.path.join("outputs", "held_out_test_set.csv"),
        help="Path to the held-out pairs CSV (relative to project root).",
    )
    parser.add_argument(
        "--api_desc", default=os.path.join("outputs", "api_descriptors.csv"),
        help="Path to API RDKit descriptor CSV for the held-out set.",
    )
    parser.add_argument(
        "--excipient_desc", default=os.path.join("outputs", "excipient_descriptors.csv"),
        help="Path to excipient RDKit descriptor CSV for the held-out set.",
    )
    parser.add_argument(
        "--output", default=os.path.join("outputs", "held_out_test_predictions.csv"),
        help="Path to write predictions CSV (relative to project root).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (default: checkpoints/molformer/best_model.pt).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold for the 'prediction' column. "
             "Default: tuned threshold from metrics/molformer/run_metrics.json, else 0.5.",
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)
    device = config.get_device()
    logging.info(f"Using device: {device}")

    input_path = os.path.join(ROOT, args.input)
    api_desc_path = os.path.join(ROOT, args.api_desc)
    exc_desc_path = os.path.join(ROOT, args.excipient_desc)
    output_path = os.path.join(ROOT, args.output)

    for path, label in [
        (input_path, "held-out pairs CSV"),
        (api_desc_path, "API descriptor CSV"),
        (exc_desc_path, "excipient descriptor CSV"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} held-out pairs from {input_path}")

    # The dataset pipeline requires an "Outcome1" label column; the held-out
    # CSV instead carries the true label as "ground_truth". Reuse it so the
    # dataloader works, but keep the original column intact for the output.
    if "Outcome1" not in df.columns:
        if "ground_truth" in df.columns:
            df["Outcome1"] = df["ground_truth"]
        else:
            df["Outcome1"] = 0  # dummy; unused for inference itself

    # Point the descriptor lookup at the held-out set's own descriptor files.
    config.api_descriptors_path = api_desc_path
    config.excipient_descriptors_path = exc_desc_path

    # Featurizer (frozen MoLFormer)
    featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path).to(device)

    loader = get_dataloader_from_dataframe(config, featurizer, df, is_train=False, shuffle=False)

    # Model + checkpoint
    model = APIExcipientModel(config).to(device)
    ckpt_path = args.checkpoint or os.path.join(ROOT, config.checkpoint_dir, "best_model.pt")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Loaded checkpoint: {ckpt_path}")

    threshold = resolve_threshold(args.threshold, config)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            logits = model(batch)
            all_logits.extend(logits.detach().cpu().numpy().tolist())

    logits = np.array(all_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    out_df = df.copy()
    out_df.drop(columns=["Outcome1"], inplace=True, errors="ignore")
    out_df["logit"] = logits
    out_df["probability"] = probs
    out_df["prediction"] = preds

    if "ground_truth" in out_df.columns:
        out_df["correct"] = (out_df["prediction"] == out_df["ground_truth"]).astype(int)
        accuracy = out_df["correct"].mean()
        logging.info(f"Held-out accuracy at threshold {threshold:.3f}: {accuracy:.4f} "
                      f"({out_df['correct'].sum()}/{len(out_df)})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logging.info(f"Wrote predictions to: {output_path}")


if __name__ == "__main__":
    main()