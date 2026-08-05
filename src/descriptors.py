"""Precomputed RDKit descriptor loading and normalization."""

import json
import os

import numpy as np
import pandas as pd


DESCRIPTOR_COLUMNS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "fr_NH2",
    "fr_NH1",
    "fr_ester",
    "fr_amide",
    "fr_aldehyde",
    "fr_ketone",
    "fr_phenol",
    "fr_ether",
    "fr_epoxide",
    "fr_halogen",
    "fr_COO",
]


def _cid_key(cid):
    return str(cid)


def _load_descriptor_lookup(csv_path, id_col):
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in [id_col, *DESCRIPTOR_COLUMNS] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} is missing descriptor columns: {missing_cols}")

    return {
        _cid_key(row[id_col]): row[DESCRIPTOR_COLUMNS].to_numpy(dtype=np.float32)
        for _, row in df.iterrows()
    }


def _lookup_rows(lookup, cids, label):
    rows = []
    for cid in cids:
        key = _cid_key(cid)
        if key not in lookup:
            raise ValueError(f"Missing {label} descriptor row for CID {cid}")
        rows.append(lookup[key])
    return np.stack(rows, axis=0)


def write_descriptor_norm_stats(
    train_csv_path,
    api_csv_path,
    exc_csv_path,
    norm_stats_path,
):
    train_df = pd.read_csv(train_csv_path)
    api_lookup = _load_descriptor_lookup(api_csv_path, "API_CID")
    exc_lookup = _load_descriptor_lookup(exc_csv_path, "Excipient_CID")

    api_rows = _lookup_rows(api_lookup, train_df["API_CID"].drop_duplicates(), "API")
    exc_rows = _lookup_rows(exc_lookup, train_df["Excipient_CID"].drop_duplicates(), "excipient")

    stats = {
        "api_mean": api_rows.mean(axis=0).astype(float).tolist(),
        "api_std": api_rows.std(axis=0).astype(float).tolist(),
        "exc_mean": exc_rows.mean(axis=0).astype(float).tolist(),
        "exc_std": exc_rows.std(axis=0).astype(float).tolist(),
    }

    os.makedirs(os.path.dirname(norm_stats_path), exist_ok=True)
    with open(norm_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    return stats


class DescriptorLookup:
    def __init__(self, api_csv_path, exc_csv_path, norm_stats_path):
        self.api_lookup = _load_descriptor_lookup(api_csv_path, "API_CID")
        self.exc_lookup = _load_descriptor_lookup(exc_csv_path, "Excipient_CID")

        with open(norm_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.api_mean = np.array(stats["api_mean"], dtype=np.float32)
        self.api_std = np.array(stats["api_std"], dtype=np.float32)
        self.exc_mean = np.array(stats["exc_mean"], dtype=np.float32)
        self.exc_std = np.array(stats["exc_std"], dtype=np.float32)

    def get_api(self, api_cid):
        key = _cid_key(api_cid)
        if key not in self.api_lookup:
            raise ValueError(f"Missing API descriptor row for CID {api_cid}")
        return (self.api_lookup[key] - self.api_mean) / (self.api_std + 1e-8)

    def get_exc(self, exc_cid):
        key = _cid_key(exc_cid)
        if key not in self.exc_lookup:
            raise ValueError(f"Missing excipient descriptor row for CID {exc_cid}")
        return (self.exc_lookup[key] - self.exc_mean) / (self.exc_std + 1e-8)
