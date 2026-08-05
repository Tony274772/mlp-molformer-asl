"""
Evaluation metrics, threshold tuning, and reporting.
Includes PR-AUC, Accuracy, Precision, Recall, F1, MCC, and confusion matrix.
"""

import json
import torch
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)
import logging


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    """Calculate thresholded classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle edge cases where one class is completely absent in preds
    if len(np.unique(y_pred)) == 1:
        f1 = 0.0
        mcc = 0.0
    else:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "F1": f1,
        "MCC": mcc,
        "Recall": recall,
        "CM": cm
    }


def evaluate_model(model, dataloader, criterion, device):
    """
    Run evaluation on a dataloader.
    Returns:
        metrics: dict of default threshold (0.5) metrics and PR-AUC
        loss: mean loss
        y_true, y_prob: raw arrays for threshold tuning
    """
    model.eval()
    total_loss = 0.0
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    
            logits = model(batch)
            targets = batch["labels"]
            
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            
            probs = torch.sigmoid(logits)
            
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    mean_loss = total_loss / len(dataloader.dataset)
    
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # Default metrics at threshold = 0.5 (for validation monitoring)
    default_metrics = calculate_metrics_at_threshold(y_true, y_prob, 0.5)
    default_metrics["PR-AUC"] = pr_auc
    
    return default_metrics, mean_loss


def tune_threshold(model, val_loader, criterion, device, step=0.001):
    """
    Sweep thresholds to maximize F1 on the validation set.
    """
    logging.info("Tuning threshold on validation set...")
    _, _, y_true, y_prob = evaluate_model_full(model, val_loader, criterion, device)
    
    thresholds = np.arange(0.001, 1.0, step)
    best_f1 = -1.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_prob, thresh)
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_thresh = thresh
            
    logging.info(f"Best threshold found: {best_thresh:.3f} (Val F1: {best_f1:.4f})")
    return best_thresh


def evaluate_model_full(model, dataloader, criterion, device):
    """Helper that returns y_true and y_prob for tuning."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    
            logits = model(batch)
            targets = batch["labels"]
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            probs = torch.sigmoid(logits)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    mean_loss = total_loss / len(dataloader.dataset)
    return {}, mean_loss, np.array(all_targets), np.array(all_probs)


def run_final_evaluation(model, test_loader, criterion, device, threshold):
    """Evaluate on test set using the tuned threshold."""
    logging.info(f"Running final evaluation on test set with threshold {threshold:.3f}...")
    
    _, test_loss, y_true, y_prob = evaluate_model_full(model, test_loader, criterion, device)
    
    precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(pr_recall, precision)
    
    metrics = calculate_metrics_at_threshold(y_true, y_prob, threshold)
    metrics["PR-AUC"] = pr_auc
    metrics["Loss"] = test_loss
    
    return metrics


def print_run_summary(val_metrics, test_metrics, threshold, best_epoch=None):
    """Print the final run summary explicitly as requested in AGENT_PROMPT.md"""
    print("\n" + "="*70)
    print("FINAL RUN SUMMARY")
    print("="*70)
    
    print(f"Optimal Threshold (tuned on val F1) : {threshold:.3f}")
    if best_epoch is not None:
        print(f"Best Checkpoint Epoch              : {best_epoch}")
    print("\nValidation Set Metrics (at optimal threshold):")
    print(f"  PR-AUC : {val_metrics['PR-AUC']:.4f}")
    print(f"  Acc    : {val_metrics['Accuracy']:.4f}")
    print(f"  Prec   : {val_metrics['Precision']:.4f}")
    print(f"  Recall : {val_metrics['Recall']:.4f}")
    print(f"  F1     : {val_metrics['F1']:.4f}")
    print(f"  MCC    : {val_metrics['MCC']:.4f}")
    
    print("\nTest Set Metrics (at optimal threshold):")
    print(f"  PR-AUC : {test_metrics['PR-AUC']:.4f}")
    print(f"  Acc    : {test_metrics['Accuracy']:.4f}")
    print(f"  Prec   : {test_metrics['Precision']:.4f}")
    print(f"  Recall : {test_metrics['Recall']:.4f}")
    print(f"  F1     : {test_metrics['F1']:.4f}")
    print(f"  MCC    : {test_metrics['MCC']:.4f}")
    
    cm = test_metrics["CM"]
    print("\nTest Set Confusion Matrix:")
    print(f"                 Predicted 0    Predicted 1")
    print(f"  Actual 0 (neg)     {cm[0,0]:<10d} {cm[0,1]:<10d}")
    print(f"  Actual 1 (pos)     {cm[1,0]:<10d} {cm[1,1]:<10d}")
    print("="*70 + "\n")


def _json_safe(value):
    """Convert numpy values in metric dictionaries to plain JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _metrics_for_json(metrics):
    """Use explicit names for saved metric fields."""
    payload = _json_safe(metrics)
    if "CM" in payload:
        payload["ConfusionMatrix"] = payload.pop("CM")
    return payload


def save_run_metrics(val_metrics, test_metrics, threshold, out_dir, best_epoch=None):
    """Save final validation and test metrics after training."""
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "run_metrics.json")
    payload = {
        "threshold": float(threshold),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "validation": _metrics_for_json(val_metrics),
        "test": _metrics_for_json(test_metrics),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Saved final metrics to {metrics_path}")
    return metrics_path


def save_cross_validation_metrics(fold_results, out_dir):
    """Save per-fold validation metrics and aggregate cross-validation metrics."""
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "cv_metrics.json")

    scalar_values = {}
    confusion_matrices = []
    saved_folds = []

    for result in fold_results:
        val_metrics = _metrics_for_json(result["validation"])
        saved_folds.append({
            "fold": result["fold"],
            "threshold": float(result["threshold"]),
            "best_epoch": int(result["best_epoch"]) if result.get("best_epoch") is not None else None,
            "train_rows": int(result["train_rows"]),
            "val_rows": int(result["val_rows"]),
            "validation": val_metrics,
        })

        for key, value in val_metrics.items():
            if key == "ConfusionMatrix":
                confusion_matrices.append(np.array(value))
            elif isinstance(value, (int, float)):
                scalar_values.setdefault(key, []).append(float(value))

    aggregate = {}
    for key, values in scalar_values.items():
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    payload = {
        "num_folds": len(saved_folds),
        "folds": saved_folds,
        "aggregate_validation": aggregate,
        "aggregate_confusion_matrix": (
            np.sum(confusion_matrices, axis=0).astype(int).tolist()
            if confusion_matrices else None
        ),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Saved cross-validation metrics to {metrics_path}")
    return metrics_path
