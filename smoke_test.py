"""
Quick smoke test for the model and pipeline.
"""
import sys
import logging
from src.runtime import configure_thread_limits, configure_torch_runtime

configure_thread_limits()

import torch
configure_torch_runtime(torch)

from src.config import Config
from src.utils import set_seed
from src.molformer_featurization import MolFormerFeaturizer
from src.dataset import get_dataloaders
from src.model import APIExcipientModel
from src.loss import AsymmetricLoss

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test():
    config = Config(batch_size=4, max_epochs=1)
    set_seed(config.seed)
    device = config.get_device()
    
    logging.info("Loading Featurizer...")
    featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path).to(device)
    
    logging.info("Loading Data...")
    train_loader, _, _ = get_dataloaders(config, featurizer)
    
    logging.info("Initializing Model...")
    model = APIExcipientModel(config).to(device)
    criterion = AsymmetricLoss()
    
    batch = next(iter(train_loader))
    
    logging.info("Running Forward Pass...")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
            
    logits = model(batch)
    loss = criterion(logits, batch["labels"])
    
    logging.info(f"Output logits shape: {logits.shape}")
    logging.info(f"Loss value: {loss.item()}")
    logging.info("Smoke test passed successfully!")

if __name__ == "__main__":
    test()
