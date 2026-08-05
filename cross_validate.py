"""
5-fold grouped cross-validation for configuration evaluation.

Runs StratifiedGroupKFold on data/train.csv only, grouped by split_group when
available and API_CID otherwise. Saves per-fold validation metrics and aggregate
metrics to metrics/cv_metrics.json.
"""

import logging
import os
import sys
from dataclasses import replace
from src.runtime import configure_thread_limits, configure_torch_runtime

configure_thread_limits()

import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

configure_torch_runtime(torch)

from src.config import Config
from src.dataset import get_dataloader_from_dataframe
from src.evaluate import (
    calculate_metrics_at_threshold,
    evaluate_model_full,
    save_cross_validation_metrics,
    tune_threshold,
)
from src.molformer_featurization import MolFormerFeaturizer
from src.loss import AsymmetricLoss
from src.model import APIExcipientModel
from src.train import train_model
from src.utils import count_parameters, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def evaluate_validation_fold(model, val_loader, criterion, device, threshold):
    """Evaluate one validation fold at its tuned threshold."""
    _, val_loss, y_true, y_prob = evaluate_model_full(model, val_loader, criterion, device)
    metrics = calculate_metrics_at_threshold(y_true, y_prob, threshold)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics["PR-AUC"] = auc(recall, precision)
    metrics["Loss"] = val_loss
    return metrics


def cross_validate_config(config=None, n_splits=5):
    """Run grouped stratified cross-validation and save fold/aggregate metrics."""
    if n_splits != 5:
        raise ValueError("This validation utility is fixed to 5 folds by the spec.")

    config = config or Config()
    set_seed(config.seed)
    device = config.get_device()

    train_path = os.path.join(config.data_dir, "train.csv")
    df = pd.read_csv(train_path)
    group_col = "split_group" if "split_group" in df.columns else "API_CID"

    logging.info("=== 5-Fold Cross-Validation ===")
    logging.info(f"Using device: {device}")
    logging.info(f"Loading folds from {train_path}")
    logging.info(f"Grouping folds by {group_col}")

    featurizer = MolFormerFeaturizer(
        model_path=config.molformer_model_path
    ).to(device)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.seed
    )

    fold_results = []
    y = df["Outcome1"]
    groups = df[group_col]

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(df, y, groups), start=1):
        logging.info("=" * 70)
        logging.info(f"Starting fold {fold_idx}/{n_splits}")

        fold_config = replace(
            config,
            checkpoint_dir=os.path.join(config.checkpoint_dir, f"cv_fold_{fold_idx}")
        )
        set_seed(config.seed + fold_idx)

        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val = df.iloc[val_idx].reset_index(drop=True)

        train_loader = get_dataloader_from_dataframe(
            fold_config,
            featurizer,
            fold_train,
            is_train=True,
            shuffle=True
        )
        val_loader = get_dataloader_from_dataframe(
            fold_config,
            featurizer,
            fold_val,
            is_train=False,
            shuffle=False
        )

        model = APIExcipientModel(fold_config).to(device)
        trainable, total = count_parameters(model)
        logging.info(f"Fold {fold_idx} params: {trainable:,} trainable / {total:,} total")

        criterion = AsymmetricLoss(
            gamma_neg=fold_config.asl_gamma_neg,
            gamma_pos=fold_config.asl_gamma_pos,
            clip=fold_config.asl_clip
        )

        best_model = train_model(fold_config, model, train_loader, val_loader, criterion)
        best_threshold = tune_threshold(
            best_model,
            val_loader,
            criterion,
            device,
            step=fold_config.threshold_step
        )
        val_metrics = evaluate_validation_fold(
            best_model,
            val_loader,
            criterion,
            device,
            best_threshold
        )

        logging.info(
            f"Fold {fold_idx} complete | "
            f"PR-AUC: {val_metrics['PR-AUC']:.4f} | "
            f"Acc: {val_metrics['Accuracy']:.4f} | "
            f"Prec: {val_metrics['Precision']:.4f} | "
            f"Recall: {val_metrics['Recall']:.4f} | "
            f"F1: {val_metrics['F1']:.4f} | "
            f"MCC: {val_metrics['MCC']:.4f}"
        )

        fold_results.append({
            "fold": fold_idx,
            "threshold": best_threshold,
            "best_epoch": getattr(best_model, "best_epoch", None),
            "train_rows": len(fold_train),
            "val_rows": len(fold_val),
            "validation": val_metrics,
        })

    metrics_path = save_cross_validation_metrics(fold_results, config.metrics_dir)
    mean_pr_auc = sum(result["validation"]["PR-AUC"] for result in fold_results) / len(fold_results)
    logging.info(f"Mean validation PR-AUC across 5 folds: {mean_pr_auc:.4f}")
    logging.info(f"Cross-validation complete. Metrics saved to {metrics_path}")

    return {
        "fold_results": fold_results,
        "mean_val_pr_auc": mean_pr_auc,
        "metrics_path": metrics_path,
    }


if __name__ == "__main__":
    cross_validate_config()
