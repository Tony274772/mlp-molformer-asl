# API–Excipient Compatibility Prediction Model — Implementation Plan

Binary classification model to predict API-excipient pharmaceutical compatibility (compatible=0 vs incompatible=1) using PyTorch with mol2vec featurization, bidirectional cross-attention fusion, and Asymmetric Loss for class imbalance.

## User Review Required

> [!IMPORTANT]
> **Functional-category branch (Section 4) skipped for v1** per user instruction. The excipient encoder will use structural branch + missing-flag + learned placeholder only. The gate mechanism (`g = sigmoid(W·m + b)`) simplifies to a direct structural encoding with the missing-flag concatenated. The architecture is designed so the categorical branch can be added later without restructuring.

> [!IMPORTANT]
> **Lookup-table module (Section 6) skipped** — no functional-group/reaction-type table file found in the data folder. The model runs cleanly without it, as specified in the architecture.

> [!WARNING]
> **mol2vec `model_300dim.pkl` download required.** The pretrained 300-dim mol2vec model (~100MB) must be downloaded from the mol2vec GitHub repo. The script will auto-download it on first run if not present. Requires `gensim<4.0` to load correctly.

## Proposed Changes

### Project Structure

```
Api-Excipient-Compatibility/
├── data/                          # existing — train.csv, val.csv, test.csv, start_dataset.csv
├── src/
│   ├── __init__.py
│   ├── config.py                  # [NEW] all hyperparameters in a dataclass
│   ├── featurization.py           # [NEW] mol2vec pipeline (fragment extraction, token sets, UNK embedding)
│   ├── dataset.py                 # [NEW] PyTorch Dataset + DataLoader factory
│   ├── model.py                   # [NEW] full model architecture (Sections 4-5)
│   ├── loss.py                    # [NEW] Asymmetric Loss (ASL) on raw logits
│   ├── train.py                   # [NEW] training loop with early stopping, LR scheduling
│   ├── evaluate.py                # [NEW] metrics (PR-AUC, F1, MCC, recall), threshold sweep, confusion matrix
│   └── utils.py                   # [NEW] logging, checkpoint saving, misc helpers
├── main.py                        # [NEW] entry point — orchestrates full training + evaluation pipeline
├── sanity_check.py                # [NEW] naive random-split comparison run (Section 9)
├── requirements.txt               # [NEW] pinned dependencies
├── ARCHITECTURE.md                # existing
├── AGENT_PROMPT.md                # existing
```

---

### 1. Config (`src/config.py`) — [NEW]

A `@dataclass` exposing every hyperparameter from Section 8 so they can be swept without editing model code:

| Parameter | Default | Notes |
|---|---|---|
| `data_dir` | `"data"` | Path to train/val/test CSVs |
| `mol2vec_model_path` | `"models/model_300dim.pkl"` | Auto-download if missing |
| `mol2vec_dim` | `300` | Frozen pretrained dim |
| `proj_dim` | `128` | Token projection output dim |
| `num_heads` | `4` | Cross-attention heads |
| `attn_dropout` | `0.1` | Attention dropout |
| `clf_dropout_1` | `0.3` | MLP head first dropout |
| `clf_dropout_2` | `0.2` | MLP head second dropout |
| `lr` | `3e-4` | AdamW learning rate |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `batch_size` | `64` | Training batch size |
| `max_epochs` | `150` | Maximum training epochs |
| `early_stop_patience` | `20` | Early stopping on val PR-AUC |
| `lr_patience` | `8` | ReduceLROnPlateau patience |
| `lr_factor` | `0.5` | ReduceLROnPlateau factor |
| `modality_dropout_rate` | `0.2` | Excipient structural branch dropout (15-30%) |
| `asl_gamma_neg` | `4` | ASL negative focusing |
| `asl_gamma_pos` | `1` | ASL positive focusing |
| `asl_clip` | `0.05` | ASL probability clipping |
| `threshold_step` | `0.001` | F1-max threshold sweep granularity |
| `seed` | `42` | Random seed |
| `device` | `auto` | CUDA if available, else CPU |
| `checkpoint_dir` | `"checkpoints"` | Model checkpoint save directory |

---

### 2. Featurization (`src/featurization.py`) — [NEW]

**mol2vec pipeline (Section 3):**

- **Download helper**: Auto-download `model_300dim.pkl` from the mol2vec GitHub repo if not present locally. Save to `models/` directory.
- **Load**: `Word2Vec.load(path)` (not `KeyedVectors.load` — per spec). Pin `gensim<4.0`.
- **Fragment extraction**: For each SMILES string:
  1. `Chem.MolFromSmiles(smiles)` → RDKit mol
  2. `mol2alt_sentence(mol, radius=1)` → list of Morgan identifier strings (2 per atom: radius-0 + radius-1)
  3. For each identifier: lookup in mol2vec vocabulary → 300-dim vector. Unknown identifiers → map to a learned `UNK_fragment` embedding (trainable parameter, initialized ~N(0, 0.01)).
  4. Output **token set** `T = {t_1, ..., t_n}` (variable length, n = 2 × atom_count) — kept as separate vectors, NOT collapsed.
  5. Also compute **pooled global vector** `g = Σ t_i` (300-dim, sum of all token vectors).
- **Frozen embeddings**: mol2vec fragment vectors are lookup-only (frozen). Only the UNK embedding is trainable.
- **Caching**: Pre-featurize all unique SMILES at startup and cache in a dict to avoid redundant RDKit/mol2vec calls during training.

**Key implementation detail**: The featurizer returns `(token_matrix: Tensor[n, 300], global_vec: Tensor[300], num_tokens: int)` per molecule. Batching uses padding + mask.

---

### 3. Dataset (`src/dataset.py`) — [NEW]

**`CompatibilityDataset(torch.utils.data.Dataset)`:**
- Loads CSV, extracts `API_Smiles`, `Excipient_Smiles`, `Outcome1`
- Calls featurizer to get token sets + global vectors for both API and excipient
- Returns per sample:
  - `api_tokens` (padded), `api_global`, `api_mask`, `api_num_tokens`
  - `exc_tokens` (padded), `exc_global`, `exc_mask`, `exc_num_tokens`
  - `exc_smiles_available` (missing-flag: always 1 for training data, but 0 when modality dropout fires)
  - `label` (Outcome1)

**Collate function**: Custom collate to pad variable-length token sets within a batch to the max length in that batch, produce attention masks.

**Modality dropout**: Applied in `__getitem__` during training — with probability `modality_dropout_rate`, replace excipient tokens with the learned placeholder embedding and set `exc_smiles_available=0`.

**DataLoader factory**: Returns train/val/test DataLoaders with appropriate shuffle/no-shuffle.

---

### 4. Model Architecture (`src/model.py`) — [NEW]

#### 4a. Excipient Encoder (Section 4, simplified for v1)

- **Structural branch**: mol2vec token set (from featurizer). If SMILES missing → single learned placeholder embedding (trainable vector, 300-dim).
- **Missing-flag**: Binary scalar `m ∈ {0, 1}`, 1 = SMILES available. Concatenated as a feature.
- No categorical branch or gate for v1 (deferred per user decision).

#### 4b. Token Projection (Stage 1)

```
APITokenProjector:    Linear(300 → 128) → LayerNorm → GELU
ExcTokenProjector:    Linear(300 → 128) → LayerNorm → GELU
```
Separate weights for API vs excipient (not shared). No positional encoding.

#### 4c. Bidirectional Cross-Attention (Stage 2)

- `nn.MultiheadAttention(d_model=128, num_heads=4, dropout=0.1, batch_first=True)`
- **Direction 1**: Q=exc_tokens, K/V=api_tokens → refined exc_tokens
- **Direction 2**: Q=api_tokens, K/V=exc_tokens → refined api_tokens
- Each direction: Add & LayerNorm (residual connection)
- **1 transformer layer** only (dataset too small for more)

#### 4d. Attention Pooling

- Single learned query vector (128-dim) attends over each refined token set
- `attn_score = softmax((query · tokens^T) / √d)` → weighted sum → `h_api_attn`, `h_exc_attn` ∈ R^128

#### 4e. Global Vector Merge (Stage 3)

```
g_api_proj  = Linear(300 → 128)(g_api)
g_exc_proj  = Linear(300 → 128)(g_exc)
h_api = [h_api_attn ; g_api_proj]   → 256-dim
h_exc = [h_exc_attn ; g_exc_proj]   → 256-dim
```

#### 4f. Pair Interaction (Stage 4)

```
pair_vec = [h_api ; h_exc ; h_api ⊙ h_exc ; |h_api − h_exc|]  → 1024-dim
```

#### 4g. Classifier Head (Stage 5)

```
Linear(1024 → 256) → GELU → Dropout(0.3) → LayerNorm
Linear(256 → 64)   → GELU → Dropout(0.2)
Linear(64 → 1)     # raw logit, no sigmoid
```

**Note on missing-flag**: The missing-flag scalar is concatenated into `h_exc` before the pair interaction layer, making `pair_vec` slightly wider (1026-dim → adjusted head accordingly). Alternatively, it can be concatenated after `g_exc_proj` making `h_exc` 257-dim, resulting in `pair_vec` = 1028-dim. I'll concatenate it into the excipient representation at the global merge stage.

---

### 5. Loss Function (`src/loss.py`) — [NEW]

**Asymmetric Loss (ASL)** operating on raw logits:

```python
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        ...
    def forward(self, logits, targets):
        # Apply sigmoid internally
        p = torch.sigmoid(logits)
        # Clip negative probability (probability shifting)
        p_neg = (p + clip).clamp(max=1.0)
        # Compute loss terms
        loss_pos = -targets * (1 - p)**gamma_pos * torch.log(p + eps)
        loss_neg = -(1 - targets) * p_neg**gamma_neg * torch.log(1 - p_neg + eps)
        return (loss_pos + loss_neg).mean()
```

No SMOTE, no class weighting, no resampling — ASL alone handles the 9.7% imbalance.

---

### 6. Training Loop (`src/train.py`) — [NEW]

- **Optimizer**: `AdamW(lr=3e-4, weight_decay=1e-5)`
- **Scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=8)` monitoring val PR-AUC
- **Early stopping**: Patience=20 on val PR-AUC (best model checkpoint saved)
- **Per-epoch logging**: train loss, val loss, val PR-AUC, val F1, val MCC, current LR
- **Checkpoint**: Save best model (by val PR-AUC) to `checkpoints/best_model.pt`

---

### 7. Evaluation (`src/evaluate.py`) — [NEW]

- **Metrics**: PR-AUC, F1, MCC, recall on positive class (incompatible)
- **F1-max threshold sweep**: Sweep threshold ∈ (0, 1) step 0.001 on val predictions, select threshold maximizing F1
- **Final evaluation**: Apply selected threshold to test set — report PR-AUC, F1, MCC, recall, confusion matrix
- **Run summary**: Print complete summary (final val metrics, test metrics, threshold used, confusion matrix, training time)

---

### 8. Entry Point (`main.py`) — [NEW]

Orchestrates the full pipeline:
1. Parse config (CLI args or defaults)
2. Load mol2vec model (download if needed)
3. Pre-featurize all unique SMILES
4. Build datasets + dataloaders
5. Initialize model, loss, optimizer, scheduler
6. Training loop with early stopping
7. Load best checkpoint
8. Threshold sweep on val
9. Final evaluation on test
10. Print full run summary

---

### 9. Sanity Check (`sanity_check.py`) — [NEW]

Section 9 naive random-split comparison:
- Re-split `start_dataset.csv` with naive `train_test_split` (60/20/20, no grouping, no stratification or simple stratification)
- Train identical architecture on the naive split
- Report same metrics (PR-AUC, F1, MCC, recall)
- Compare against grouped-split results to quantify API-identity leakage inflation

---

### 10. Dependencies (`requirements.txt`) — [NEW]

```
torch>=2.0
gensim<4.0
rdkit-pypi
mol2vec
numpy
pandas
scikit-learn
```

---

## Verification Plan

### Automated Tests
1. **Smoke test**: Run `python main.py` end-to-end with `max_epochs=2` to verify the full pipeline runs without errors
2. **Shape assertions**: Print tensor shapes at each stage during first forward pass to verify dimensional correctness
3. **Data integrity**: Verify train/val/test row counts match split log (2030/842/672), no overlap in API_CIDs
4. **Loss sanity**: Verify ASL loss decreases over first few epochs

### Manual Verification
1. Review per-epoch training logs for expected behavior (decreasing loss, improving val PR-AUC)
2. Review final run summary (val/test metrics, confusion matrix, threshold)
3. Compare grouped-split vs naive-split metrics in sanity check output
4. Verify mol2vec featurization produces expected token counts (2 × atom count per molecule)
