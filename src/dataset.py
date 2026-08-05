"""
PyTorch Dataset and DataLoader factory.

Loads train/val/test CSVs, applies modality dropout to excipients during training.
Note: Since the dataset is small (~2k rows), and transformer featurization can
be somewhat slow, we'll rely on the caching inside MolFormerFeaturizer.
To make it fast, the dataset just returns raw strings and labels, and a custom
collate function calls the featurizer.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

from src.descriptors import DescriptorLookup

class CompatibilityDataset(Dataset):
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None, is_train: bool = False, modality_dropout_rate: float = 0.2):
        """
        Args:
            csv_path: path to train.csv, val.csv, or test.csv
            df: in-memory dataframe, used for cross-validation folds
            is_train: if True, applies modality dropout dynamically
            modality_dropout_rate: prob to simulate missing excipient SMILES
        """
        if df is None and csv_path is None:
            raise ValueError("Either csv_path or df must be provided.")
        self.df = df.reset_index(drop=True) if df is not None else pd.read_csv(csv_path)
        self.is_train = is_train
        self.modality_dropout_rate = modality_dropout_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        api_smi = row["API_Smiles"]
        exc_smi = row["Excipient_Smiles"]
        label = float(row["Outcome1"])
        
        # Missing-flag: 1 = SMILES available, 0 = absent/dropped
        # For start_dataset.csv, all have SMILES.
        # But we simulate missing ones during training.
        exc_smiles_available = 1.0
        
        # If exc_smi is literally missing in data (e.g. NaN) - handle just in case
        if pd.isna(exc_smi) or str(exc_smi).strip() == "":
            exc_smiles_available = 0.0
            exc_smi = "" # Empty string for featurizer (will return 0 tokens)
        elif self.is_train and random.random() < self.modality_dropout_rate:
            # Modality dropout: pretend structure is unavailable, but keep the
            # real SMILES in the batch. The model uses exc_smiles_available=0 to
            # replace the structural branch with its learned placeholder, so the
            # featurizer should never see a fake empty SMILES for valid data.
            exc_smiles_available = 0.0

        return {
            "api_cid": row["API_CID"],
            "exc_cid": row["Excipient_CID"],
            "api_smi": api_smi,
            "exc_smi": exc_smi,
            "exc_smiles_available": exc_smiles_available,
            "label": label
        }

def create_collate_fn(featurizer, descriptor_lookup=None):
    """Creates a collate function that uses the provided featurizer."""
    
    def collate_fn(batch):
        api_smiles = [item["api_smi"] for item in batch]
        exc_smiles = [item["exc_smi"] for item in batch]
        
        exc_available = torch.tensor([item["exc_smiles_available"] for item in batch], dtype=torch.float32)
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        
        # Featurize
        api_padded, api_global, api_mask, api_num = featurizer(api_smiles)
        exc_padded, exc_global, exc_mask, exc_num = featurizer(exc_smiles)
        
        batch_dict = {
            "api_tokens": api_padded,
            "api_global": api_global,
            "api_mask": api_mask,
            "api_num": api_num,
            
            "exc_tokens": exc_padded,
            "exc_global": exc_global,
            "exc_mask": exc_mask,
            "exc_num": exc_num,
            
            "exc_available": exc_available,
            "labels": labels
        }

        if descriptor_lookup is not None:
            batch_dict["api_desc"] = torch.tensor(
                [descriptor_lookup.get_api(item["api_cid"]) for item in batch],
                dtype=torch.float32
            )
            batch_dict["exc_desc"] = torch.tensor(
                [descriptor_lookup.get_exc(item["exc_cid"]) for item in batch],
                dtype=torch.float32
            )

        return batch_dict
        
    return collate_fn


def create_descriptor_lookup(config):
    if not config.use_descriptors:
        return None
    return DescriptorLookup(
        config.api_descriptors_path,
        config.excipient_descriptors_path,
        config.descriptor_norm_stats_path
    )

def get_dataloaders(config, featurizer):
    """Return train, val, and test dataloaders."""
    
    train_dataset = CompatibilityDataset(
        csv_path=f"{config.data_dir}/train.csv", 
        is_train=True, 
        modality_dropout_rate=config.modality_dropout_rate
    )
    val_dataset = CompatibilityDataset(
        csv_path=f"{config.data_dir}/val.csv", 
        is_train=False
    )
    test_dataset = CompatibilityDataset(
        csv_path=f"{config.data_dir}/test.csv", 
        is_train=False
    )
    
    descriptor_lookup = create_descriptor_lookup(config)
    collate = create_collate_fn(featurizer, descriptor_lookup)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def get_dataloader_from_dataframe(config, featurizer, df, is_train=False, shuffle=False):
    """Return a dataloader for an in-memory dataframe."""
    dataset = CompatibilityDataset(
        df=df,
        is_train=is_train,
        modality_dropout_rate=config.modality_dropout_rate
    )
    descriptor_lookup = create_descriptor_lookup(config)
    collate = create_collate_fn(featurizer, descriptor_lookup)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=False
    )
