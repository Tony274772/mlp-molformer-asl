"""
Training loop with early stopping, model checkpointing, and LR scheduling.
"""

import logging
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .evaluate import evaluate_model

def train_model(config, model, train_loader, val_loader, criterion):
    """
    Train the model using the provided configuration.
    """
    device = config.get_device()
    model = model.to(device)
    criterion = criterion.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=config.lr_factor, patience=config.lr_patience)
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    
    best_val_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    
    logging.info(f"Starting training on {device} for up to {config.max_epochs} epochs...")
    
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            
            logits = model(batch)
            targets = batch["labels"]
            
            loss = criterion(logits, targets)
            loss.backward()
            
            if config.grad_clip_norm and config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_clip_norm
                )

            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        val_metrics, val_loss = evaluate_model(model, val_loader, criterion, device)
        val_pr_auc = val_metrics["PR-AUC"]
        val_f1 = val_metrics["F1"]
        val_mcc = val_metrics["MCC"]
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info(f"Epoch {epoch:03d}/{config.max_epochs} | LR: {current_lr:.2e} | "
                     f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                     f"Val PR-AUC: {val_pr_auc:.4f} | "
                     f"Val F1: {val_f1:.4f} | Val MCC: {val_mcc:.4f}")
        
        scheduler.step(val_pr_auc)
        
        # Checkpoint & Early stopping
        if val_pr_auc > best_val_pr_auc + config.early_stop_min_delta:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(
                f"  -> Best model saved! "
                f"(Val PR-AUC: {best_val_pr_auc:.4f}; Epoch: {best_epoch})"
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                logging.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(patience={config.early_stop_patience}, min_delta={config.early_stop_min_delta})"
                )
                break
                
    # Load best model before returning
    logging.info(
        f"Training complete. Loading best model from {best_model_path} "
        f"(epoch={best_epoch}, PR-AUC: {best_val_pr_auc:.4f})"
    )
    model.load_state_dict(torch.load(best_model_path))
    model.best_epoch = best_epoch
    model.best_val_pr_auc = best_val_pr_auc
    return model
