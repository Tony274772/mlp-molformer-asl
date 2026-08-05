"""
Main entry point for training and evaluating the API-Excipient Compatibility model.
"""

import logging
import sys
from src.runtime import configure_thread_limits, configure_torch_runtime

configure_thread_limits()

import torch
configure_torch_runtime(torch)

from src.config import Config
from src.utils import set_seed, count_parameters
from src.molformer_featurization import MolFormerFeaturizer
from src.dataset import get_dataloaders
from src.model import APIExcipientModel
from src.loss import AsymmetricLoss
from src.train import train_model
from src.evaluate import tune_threshold, run_final_evaluation, print_run_summary, save_run_metrics, calculate_metrics_at_threshold, evaluate_model_full, auc, precision_recall_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    config = Config()
    set_seed(config.seed)
    device = config.get_device()
    
    logging.info("=== API-Excipient Compatibility Prediction ===")
    logging.info(f"Using device: {device}")
    
    # 1. Featurization
    featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path)
    featurizer = featurizer.to(device)
    
    # 2. Datasets & DataLoaders
    logging.info("Initializing datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config, featurizer)
    logging.info(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    # 3. Model
    model = APIExcipientModel(config).to(device)
    trainable, total = count_parameters(model)
    logging.info(f"Model initialized. Trainable params: {trainable:,} / Total params: {total:,}")
    
    # 4. Loss
    criterion = AsymmetricLoss(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip
    )
    
    # 5. Training Loop
    best_model = train_model(config, model, train_loader, val_loader, criterion)
    
    # 6. Threshold Tuning (on validation set)
    best_threshold = tune_threshold(best_model, val_loader, criterion, device, step=config.threshold_step)
    
    # Calculate full val metrics at the optimal threshold for the summary
    _, val_loss, val_true, val_prob = evaluate_model_full(best_model, val_loader, criterion, device)
    val_metrics = calculate_metrics_at_threshold(val_true, val_prob, best_threshold)
    precision, recall, _ = precision_recall_curve(val_true, val_prob)
    val_metrics["PR-AUC"] = auc(recall, precision)
    val_metrics["Loss"] = val_loss
    
    # 7. Final Evaluation (on test set)
    test_metrics = run_final_evaluation(best_model, test_loader, criterion, device, best_threshold)
    
    # 8. Print and save summary
    best_epoch = getattr(best_model, "best_epoch", None)
    print_run_summary(val_metrics, test_metrics, best_threshold, best_epoch=best_epoch)
    save_run_metrics(val_metrics, test_metrics, best_threshold, config.metrics_dir, best_epoch=best_epoch)


if __name__ == "__main__":
    main()
