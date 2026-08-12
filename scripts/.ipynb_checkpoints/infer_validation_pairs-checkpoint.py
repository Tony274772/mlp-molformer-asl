"""Run inference on outputs/validation_pairs.csv and write predictions to a new CSV.

Writes `outputs/validation_pairs_with_preds.csv` with added columns:
  - `logit` : raw model logit
  - `prob`  : sigmoid(logit)
  - `pred`  : binary prediction at threshold 0.5
"""
import os
import logging
import pandas as pd
import torch

from src.runtime import configure_thread_limits, configure_torch_runtime
configure_thread_limits()
import torch as _torch
configure_torch_runtime(_torch)

from src.config import Config
from src.molformer_featurization import MolFormerFeaturizer
from src.dataset import get_dataloader_from_dataframe
from src.model import APIExcipientModel


def main():
    logging.basicConfig(level=logging.INFO)
    root = os.path.dirname(os.path.dirname(__file__))
    inp_path = os.path.join(root, "outputs", "validation_pairs.csv")
    out_path = os.path.join(root, "outputs", "validation_pairs_with_preds.csv")

    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    df = pd.read_csv(inp_path)
    # The dataset pipeline expects an `Outcome1` label column. For pure inference,
    # create a dummy column if missing.
    if "Outcome1" not in df.columns:
        df["Outcome1"] = 0
    config = Config()

    # Use validation descriptor files if present in outputs/
    api_desc_path = os.path.join(root, "outputs", "api_descriptors_val.csv")
    exc_desc_path = os.path.join(root, "outputs", "excipient_descriptors_val.csv")
    if os.path.exists(api_desc_path) and os.path.exists(exc_desc_path):
        config.api_descriptors_path = api_desc_path
        config.excipient_descriptors_path = exc_desc_path

    device = config.get_device()
    logging.info(f"Device: {device}")

    featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path).to(device)
    loader = get_dataloader_from_dataframe(config, featurizer, df, is_train=False, shuffle=False)

    model = APIExcipientModel(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        logging.info(f"Loaded checkpoint: {ckpt}")
    else:
        logging.warning(f"Checkpoint not found at {ckpt}; model remains randomly initialized.")

    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            logits = model(batch)
            logits = logits.detach().cpu().numpy()
            all_logits.extend(logits.tolist())

    import numpy as np
    logits = np.array(all_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    out_df = df.copy()
    out_df["logit"] = logits
    out_df["prob"] = probs
    out_df["pred"] = preds

    out_df.to_csv(out_path, index=False)
    logging.info(f"Wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()
