"""
Compute RDKit physicochemical + functional-group descriptors for every
unique API and every unique Excipient found across train.csv, val.csv,
and test.csv, and write two CSVs:

  api_descriptors.csv        - one row per unique API_CID
  excipient_descriptors.csv  - one row per unique Excipient_CID

21 descriptors per molecule, fixed order:

Physicochemical (10):
 1. MolWt
 2. MolLogP        (Crippen)
 3. TPSA
 4. NumHDonors
 5. NumHAcceptors
 6. NumRotatableBonds
 7. NumAromaticRings
 8. RingCount
 9. FractionCSP3
10. HeavyAtomCount

Reactive functional-group counts (11, from rdkit.Chem.Fragments):
11. fr_NH2        (primary amine)
12. fr_NH1        (secondary amine)
13. fr_ester
14. fr_amide
15. fr_aldehyde
16. fr_ketone
17. fr_phenol
18. fr_ether
19. fr_epoxide
20. fr_halogen
21. fr_COO        (carboxylic acid)

Usage:
    python compute_descriptors.py \
        --train train.csv --val val.csv --test test.csv \
        --outdir .
"""

import argparse
import logging

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Fragments

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DESCRIPTOR_NAMES = [
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


def compute_descriptors(smiles: str) -> np.ndarray:
    """Compute the 21 fixed-order descriptors for one SMILES string.

    Returns a zero vector of length 21 (and logs a warning) if the SMILES
    fails to parse, rather than raising.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Failed to parse SMILES, returning zero vector: %s", smiles)
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)

    values = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Fragments.fr_NH2(mol),
        Fragments.fr_NH1(mol),
        Fragments.fr_ester(mol),
        Fragments.fr_amide(mol),
        Fragments.fr_aldehyde(mol),
        Fragments.fr_ketone(mol),
        Fragments.fr_phenol(mol),
        Fragments.fr_ether(mol),
        Fragments.fr_epoxide(mol),
        Fragments.fr_halogen(mol),
        Fragments.fr_COO(mol),
    ]
    return np.array(values, dtype=np.float32)


def build_descriptor_table(unique_df: pd.DataFrame, id_col: str, smiles_col: str) -> pd.DataFrame:
    """Given a dataframe with unique (id_col, smiles_col) rows, compute
    descriptors for each and return a table: id_col, smiles_col, <21 descriptor cols>.
    """
    cache: dict[str, np.ndarray] = {}
    rows = []
    for _, row in unique_df.iterrows():
        smi = row[smiles_col]
        if smi not in cache:
            cache[smi] = compute_descriptors(smi)
        rows.append([row[id_col], smi, *cache[smi]])

    return pd.DataFrame(rows, columns=[id_col, smiles_col, *DESCRIPTOR_NAMES])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--val", default="val.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    dfs = [pd.read_csv(args.train), pd.read_csv(args.val), pd.read_csv(args.test)]
    combined = pd.concat(dfs, ignore_index=True)

    # --- Unique APIs ---
    unique_apis = (
        combined[["API_CID", "API_Smiles"]]
        .drop_duplicates(subset="API_CID")
        .sort_values("API_CID")
        .reset_index(drop=True)
    )
    api_table = build_descriptor_table(unique_apis, "API_CID", "API_Smiles")
    api_out = f"{args.outdir}/api_descriptors.csv"
    api_table.to_csv(api_out, index=False)
    logger.info("Wrote %d unique APIs -> %s", len(api_table), api_out)

    # --- Unique Excipients ---
    unique_excs = (
        combined[["Excipient_CID", "Excipient_Smiles"]]
        .drop_duplicates(subset="Excipient_CID")
        .sort_values("Excipient_CID")
        .reset_index(drop=True)
    )
    exc_table = build_descriptor_table(unique_excs, "Excipient_CID", "Excipient_Smiles")
    exc_out = f"{args.outdir}/excipient_descriptors.csv"
    exc_table.to_csv(exc_out, index=False)
    logger.info("Wrote %d unique excipients -> %s", len(exc_table), exc_out)


if __name__ == "__main__":
    main()