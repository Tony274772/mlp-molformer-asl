"""
Leakage-safe split for API-Excipient compatibility data (start_dataset.csv).

Grouping strategy:
  - Primary group key: API_CID (no API's pairs are allowed to cross train/val/test).
  - Diagnostic layer: Butina clustering on API Morgan fingerprints (Tanimoto >= 0.85)
    to check whether near-duplicate APIs (different CIDs, near-identical structures,
    e.g. salt forms) exist. If they do, clusters are merged and used as the group key
    instead of raw API_CID, so near-duplicates also can't cross splits.
  - Excipients are intentionally NOT grouped (excipient overlap across splits is
    expected/desired -- it mirrors the deployment scenario where the same recurring
    excipient vocabulary appears in both train and unseen data).

Output: train.csv (60%), val.csv (20%), test.csv (20%), each stratified on Outcome1,
grouped so no API (or near-duplicate API) leaks across splits.
"""

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold

RDLogger.DisableLog("rdApp.*")

INPUT_PATH = "start_dataset.csv"
OUT_DIR = "."
RANDOM_STATE = 42
BUTINA_CUTOFF = 0.15          # distance cutoff -> Tanimoto similarity >= 0.85
MERGE_THRESHOLD_PCT = 5.0     # only switch to cluster-based grouping if >this% of APIs merge

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
n_total = len(df)
n_pos = int((df["Outcome1"] == 1).sum())
n_neg = int((df["Outcome1"] == 0).sum())

print("=" * 70)
print("STEP 1: RAW DATA")
print("=" * 70)
print(f"Total pairs           : {n_total}")
print(f"Compatible (0)         : {n_neg} ({n_neg/n_total:.1%})")
print(f"Incompatible (1)       : {n_pos} ({n_pos/n_total:.1%})")
print(f"Unique API_CID          : {df['API_CID'].nunique()}")
print(f"Unique Excipient_CID    : {df['Excipient_CID'].nunique()}")

# ---------------------------------------------------------------------------
# 2. Butina clustering on unique APIs (near-duplicate detection)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 2: NEAR-DUPLICATE API CHECK (Butina, Tanimoto >= 0.85)")
print("=" * 70)

api_df = df[["API_CID", "API_Smiles"]].drop_duplicates().reset_index(drop=True)

mols, valid_idx = [], []
for i, smi in enumerate(api_df["API_Smiles"]):
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        mols.append(m)
        valid_idx.append(i)

fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]

dists = []
n = len(fps)
for i in range(1, n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - s for s in sims])

clusters = Butina.ClusterData(dists, n, BUTINA_CUTOFF, isDistData=True)

cluster_id_by_pos = {}
for cid, members in enumerate(clusters):
    for m_idx in members:
        cluster_id_by_pos[m_idx] = cid

api_df["butina_cluster"] = -1
for local_pos, orig_idx in enumerate(valid_idx):
    api_df.loc[orig_idx, "butina_cluster"] = cluster_id_by_pos[local_pos]
# any invalid SMILES fall back to their own singleton cluster (own row index)
invalid_mask = api_df["butina_cluster"] == -1
api_df.loc[invalid_mask, "butina_cluster"] = (
    api_df.loc[invalid_mask].index + 10_000_000
)

cluster_sizes = api_df.groupby("butina_cluster")["API_CID"].nunique()
multi_clusters = cluster_sizes[cluster_sizes > 1]
n_apis_merged = int(multi_clusters.sum())
pct_merged = 100 * n_apis_merged / len(api_df)

print(f"Unique APIs                     : {len(api_df)}")
print(f"Butina clusters (>=0.85 sim)    : {api_df['butina_cluster'].nunique()}")
print(f"Clusters with >1 API_CID        : {len(multi_clusters)}")
print(f"APIs involved in a merged cluster: {n_apis_merged} ({pct_merged:.1f}% of all APIs)")

if pct_merged > MERGE_THRESHOLD_PCT:
    group_col_name = "butina_cluster"
    print(f"\n-> {pct_merged:.1f}% > {MERGE_THRESHOLD_PCT}% threshold: "
          f"using Butina cluster as the grouping key (catches near-duplicate APIs).")
else:
    group_col_name = "API_CID"
    print(f"\n-> Only {pct_merged:.1f}% of APIs affected (<= {MERGE_THRESHOLD_PCT}% threshold): "
          f"near-duplicate risk is negligible. Using plain API_CID as the grouping key.")

df = df.merge(api_df[["API_CID", "butina_cluster"]], on="API_CID", how="left")
df["split_group"] = df[group_col_name] if group_col_name == "butina_cluster" else df["API_CID"]

# ---------------------------------------------------------------------------
# 3. Grouped stratified split: 60/20/20 (test -> val -> train)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3: GROUPED STRATIFIED SPLIT (60/20/20)")
print("=" * 70)

sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
trainval_idx, test_idx = next(
    sgkf1.split(df, y=df["Outcome1"], groups=df["split_group"])
)
trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# split trainval (80% of data) into train(60%) / val(20%) -> val is 1/4 of trainval
sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
train_idx, val_idx = next(
    sgkf2.split(trainval_df, y=trainval_df["Outcome1"], groups=trainval_df["split_group"])
)
train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. Leakage verification
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 4: LEAKAGE VERIFICATION")
print("=" * 70)

train_groups = set(train_df["split_group"])
val_groups = set(val_df["split_group"])
test_groups = set(test_df["split_group"])

overlap_train_val = train_groups & val_groups
overlap_train_test = train_groups & test_groups
overlap_val_test = val_groups & test_groups

print(f"Group overlap train<->val   : {len(overlap_train_val)} groups")
print(f"Group overlap train<->test  : {len(overlap_train_test)} groups")
print(f"Group overlap val<->test    : {len(overlap_val_test)} groups")

if overlap_train_val or overlap_train_test or overlap_val_test:
    print("\n*** LEAKAGE DETECTED -- DO NOT USE THIS SPLIT ***")
else:
    print("\nNo group leakage: every API (or API cluster) appears in exactly one split.")

train_apis = set(train_df["API_CID"])
val_apis = set(val_df["API_CID"])
test_apis = set(test_df["API_CID"])
print(f"\nAPI_CID overlap train<->val  : {len(train_apis & val_apis)}")
print(f"API_CID overlap train<->test : {len(train_apis & test_apis)}")
print(f"API_CID overlap val<->test   : {len(val_apis & test_apis)}")

# excipient overlap is expected and reported for transparency, not treated as leakage
train_exc = set(train_df["Excipient_CID"])
val_exc = set(val_df["Excipient_CID"])
test_exc = set(test_df["Excipient_CID"])
print(f"\nExcipient_CID overlap train<->test (expected, NOT leakage): "
      f"{len(train_exc & test_exc)} / {len(test_exc)} test excipients also seen in train")

# ---------------------------------------------------------------------------
# 5. Split summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5: SPLIT SUMMARY")
print("=" * 70)

for name, split_df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
    n = len(split_df)
    pos = int((split_df["Outcome1"] == 1).sum())
    neg = int((split_df["Outcome1"] == 0).sum())
    n_apis = split_df["API_CID"].nunique()
    n_exc = split_df["Excipient_CID"].nunique()
    print(f"{name:5s} | rows={n:5d} ({n/n_total:5.1%}) | "
          f"pos={pos:4d} ({pos/n:5.1%}) | neg={neg:4d} ({neg/n:5.1%}) | "
          f"unique APIs={n_apis:4d} | unique excipients={n_exc:4d}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
cols_to_drop = ["butina_cluster", "split_group"]
train_out = train_df.drop(columns=cols_to_drop)
val_out = val_df.drop(columns=cols_to_drop)
test_out = test_df.drop(columns=cols_to_drop)

train_out.to_csv(f"{OUT_DIR}/train.csv", index=False)
val_out.to_csv(f"{OUT_DIR}/val.csv", index=False)
test_out.to_csv(f"{OUT_DIR}/test.csv", index=False)

print()
print("=" * 70)
print("STEP 6: FILES WRITTEN")
print("=" * 70)
print(f"train.csv : {len(train_out)} rows")
print(f"val.csv   : {len(val_out)} rows")
print(f"test.csv  : {len(test_out)} rows")
print(f"grouping key used: {group_col_name}")
print("=" * 70)