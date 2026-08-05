"""
Sanity check script (Section 9):
Runs the identical model architecture and training procedure, but uses a naive 
random split (60/20/20, no grouping by API identity) instead of the grouped split.
This quantifies how much API-identity leakage inflates the evaluation metrics.
"""

import logging
import sys
import os
from src.runtime import configure_thread_limits, configure_torch_runtime

configure_thread_limits()

import torch
configure_torch_runtime(torch)
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils import set_seed, count_parameters
from src.molformer_featurization import MolFormerFeaturizer
from src.dataset import CompatibilityDataset, create_collate_fn, create_descriptor_lookup
from torch.utils.data import DataLoader
from src.model import APIExcipientModel
from src.loss import AsymmetricLoss
from src.train import train_model
from src.evaluate import tune_threshold, run_final_evaluation, print_run_summary, calculate_metrics_at_threshold, evaluate_model_full, auc, precision_recall_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_naive_split(data_path="data/start_dataset.csv", out_dir="data/naive"):
    """Create a 60/20/20 split without grouping by API_CID."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Simple stratified split, NO GROUPING
    train_val, test = train_test_split(df, test_size=0.2, stratify=df["Outcome1"], random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val["Outcome1"], random_state=42) # 0.25 * 0.8 = 0.2
    
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    logging.info(f"Naive split created in {out_dir}: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train_path, val_path, test_path

def main():
    config = Config()
    # Override config data_dir to point to naive split, and change checkpoint dir
    config.data_dir = "data/naive"
    config.checkpoint_dir = "checkpoints_naive"
    
    set_seed(config.seed)
    device = config.get_device()
    
    logging.info("=== NAIVE SPLIT SANITY CHECK ===")
    logging.info("This run uses a random split (no API grouping) to measure leakage inflation.")
    
    # 0. Create naive split
    create_naive_split(data_path="data/start_dataset.csv", out_dir="data/naive")
    
    # 1. Featurization
    featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path)
    featurizer = featurizer.to(device)
    
    # 2. Datasets & DataLoaders
    train_dataset = CompatibilityDataset(csv_path=f"{config.data_dir}/train.csv", is_train=True, modality_dropout_rate=config.modality_dropout_rate)
    val_dataset = CompatibilityDataset(csv_path=f"{config.data_dir}/val.csv", is_train=False)
    test_dataset = CompatibilityDataset(csv_path=f"{config.data_dir}/test.csv", is_train=False)
    
    descriptor_lookup = create_descriptor_lookup(config)
    collate = create_collate_fn(featurizer, descriptor_lookup)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    # 3. Model & Loss
    model = APIExcipientModel(config).to(device)
    criterion = AsymmetricLoss(gamma_neg=config.asl_gamma_neg, gamma_pos=config.asl_gamma_pos, clip=config.asl_clip)
    
    # 4. Train
    best_model = train_model(config, model, train_loader, val_loader, criterion)
    
    # 5. Evaluate
    best_threshold = tune_threshold(best_model, val_loader, criterion, device, step=config.threshold_step)
    
    _, _, val_true, val_prob = evaluate_model_full(best_model, val_loader, criterion, device)
    val_metrics = calculate_metrics_at_threshold(val_true, val_prob, best_threshold)
    precision, recall, _ = precision_recall_curve(val_true, val_prob)
    val_metrics["PR-AUC"] = auc(recall, precision)
    
    test_metrics = run_final_evaluation(best_model, test_loader, criterion, device, best_threshold)
    
    print("\n>>> NAIVE SPLIT RESULTS <<<")
    print("Compare these metrics against the main run to see inflation from API leakage.")
    print_run_summary(val_metrics, test_metrics, best_threshold)


if __name__ == "__main__":
    main()
