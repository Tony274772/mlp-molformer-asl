"""
generate_descriptors.py

Computes RDKit descriptors for the unique APIs and unique excipients found in
held_out_test_set.csv, writing two output CSVs in the same schema as the
reference api_descriptors.csv:

    API_CID, API_Smiles, MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors,
    NumRotatableBonds, NumAromaticRings, RingCount, FractionCSP3,
    HeavyAtomCount, fr_NH2, fr_NH1, fr_ester, fr_amide, fr_aldehyde,
    fr_ketone, fr_phenol, fr_ether, fr_epoxide, fr_halogen, fr_COO

(Excipient file uses Excipient_CID / Excipient_Smiles instead.)

Usage:
    python generate_descriptors.py \
        --input held_out_test_set.csv \
        --api_out api_descriptors.csv \
        --excipient_out excipient_descriptors.csv
"""

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Fragments import (
    fr_NH2, fr_NH1, fr_ester, fr_amide, fr_aldehyde,
    fr_ketone, fr_phenol, fr_ether, fr_epoxide, fr_halogen, fr_COO,
)


def compute_descriptors(smiles: str) -> dict:
    """Compute the full descriptor set for a single SMILES string.
    Returns None if the SMILES could not be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "fr_NH2": fr_NH2(mol),
        "fr_NH1": fr_NH1(mol),
        "fr_ester": fr_ester(mol),
        "fr_amide": fr_amide(mol),
        "fr_aldehyde": fr_aldehyde(mol),
        "fr_ketone": fr_ketone(mol),
        "fr_phenol": fr_phenol(mol),
        "fr_ether": fr_ether(mol),
        "fr_epoxide": fr_epoxide(mol),
        "fr_halogen": fr_halogen(mol),
        "fr_COO": fr_COO(mol),
    }


def build_descriptor_table(df: pd.DataFrame, cid_col: str, smiles_col: str,
                            out_cid_col: str, out_smiles_col: str) -> pd.DataFrame:
    """Given a dataframe with a CID column and a SMILES column, dedupe on CID
    and compute descriptors for each unique entity."""
    unique_df = df[[cid_col, smiles_col]].drop_duplicates(subset=[cid_col]).reset_index(drop=True)

    rows = []
    failed = []
    for _, row in unique_df.iterrows():
        cid = row[cid_col]
        smiles = row[smiles_col]
        desc = compute_descriptors(smiles)
        if desc is None:
            failed.append((cid, smiles))
            continue
        record = {out_cid_col: cid, out_smiles_col: smiles}
        record.update(desc)
        rows.append(record)

    if failed:
        print(f"WARNING: {len(failed)} SMILES failed to parse and were skipped:")
        for cid, smiles in failed:
            print(f"    CID={cid}  SMILES={smiles}")

    result = pd.DataFrame(rows)
    # Enforce exact column order matching the reference schema
    ordered_cols = [
        out_cid_col, out_smiles_col, "MolWt", "MolLogP", "TPSA", "NumHDonors",
        "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings", "RingCount",
        "FractionCSP3", "HeavyAtomCount", "fr_NH2", "fr_NH1", "fr_ester",
        "fr_amide", "fr_aldehyde", "fr_ketone", "fr_phenol", "fr_ether",
        "fr_epoxide", "fr_halogen", "fr_COO",
    ]
    result = result[ordered_cols]
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate RDKit descriptors for API and excipient sets.")
    parser.add_argument("--input", default="held_out_test_set.csv", help="Path to held_out_test_set.csv")
    parser.add_argument("--api_out", default="api_descriptors.csv", help="Output path for API descriptors")
    parser.add_argument("--excipient_out", default="excipient_descriptors.csv", help="Output path for excipient descriptors")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    api_desc = build_descriptor_table(
        df, cid_col="API_CID", smiles_col="API_Smiles",
        out_cid_col="API_CID", out_smiles_col="API_Smiles",
    )
    excipient_desc = build_descriptor_table(
        df, cid_col="Excipient_CID", smiles_col="Excipient_Smiles",
        out_cid_col="Excipient_CID", out_smiles_col="Excipient_Smiles",
    )

    api_desc.to_csv(args.api_out, index=False)
    excipient_desc.to_csv(args.excipient_out, index=False)

    print(f"Unique APIs: {len(api_desc)} -> {args.api_out}")
    print(f"Unique Excipients: {len(excipient_desc)} -> {args.excipient_out}")


if __name__ == "__main__":
    main()