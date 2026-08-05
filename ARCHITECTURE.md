# API–Excipient Compatibility Prediction Model — Architecture Specification

## 1. Problem Statement

Binary classification: given an API (drug) SMILES and an excipient SMILES, predict
whether the pair is **compatible (0)** or **incompatible (1)**.

- **Training/validation/test source:** `start_dataset.csv` — 3544 API-excipient pairs
  (3200 compatible, 344 incompatible → 9.7% positive rate), 533 unique APIs, 336 unique
  excipients. Columns: `API_CID, Excipient_CID, Outcome1, API_Smiles, Excipient_Smiles`.
- **Deployment/inference target:** `oral_formulations_final.csv` (API + inactive
  ingredient lists per formulation) joined against `excipients_in_oral_formulations.csv`
  (648 unique excipients, of which **258 (40%) have no SMILES** — see Section 6).
- Class imbalance (9.7% positive) and missing-SMILES excipients (40% on the deployment
  side) are first-class design constraints, not edge cases — both are handled inside the
  architecture and training procedure, not patched on afterward.

---

## 2. Data Split (already generated — do not re-split)

Files `train.csv` (2133 rows), `val.csv` (705 rows), `test.csv` (706 rows) were produced
by `split_data.py` from `start_dataset.csv` using a **leakage-safe grouped stratified
split**:

- Grouping key: Butina clusters of API Morgan fingerprints at Tanimoto similarity ≥ 0.85
  (catches near-duplicate APIs — salt forms, close analogs — that share a raw `API_CID`
  group would miss). 41/533 APIs (7.7%) merged into 20 multi-member clusters.
- `StratifiedGroupKFold` (5-fold for test carve-out, 4-fold on the remainder for
  train/val), stratified on `Outcome1`, grouped on the cluster ID.
- **No API (or near-duplicate API cluster) appears in more than one split.**
  Excipients are intentionally *not* grouped — excipient overlap across splits is
  expected and correct, since the deployment excipient vocabulary is recurring and
  largely fixed.
- Verified: 0 group overlap, 0 `API_CID` overlap across train/val/test; class ratio
  preserved at ~9.5–9.8% positive in every split.
- **Agent instructions: load `train.csv`, `val.csv`, `test.csv` directly. Do not
  re-split `start_dataset.csv` with `train_test_split` or any ungrouped/unstratified
  method — that reintroduces API-identity leakage.**

---

## 3. Molecular Featurization — mol2vec

- Pretrained model: `model_300dim.pkl` (300-dim, skip-gram, radius 1, trained on
  ~19.9M PubChem compounds). Source:
  `https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl`.
- Load with `gensim.models.word2vec.Word2Vec.load(...)` — **not** `KeyedVectors.load`,
  which raises `AttributeError` on this file. Pin `gensim<4.0` if load errors occur.
- Fragment extraction per molecule: `mol2alt_sentence(mol, radius=1)` (RDKit) → 2
  Morgan identifiers per atom (radius 0 and radius 1).
- **Do not collapse to the standard single pooled 300-dim vector.** Keep the per-atom
  fragment vectors as a **variable-length token set** `T = {t_1, ..., t_n}` per molecule
  (n = 2 × atom count), so the fusion layer (Section 5) has something to attend over.
  Also compute the standard pooled vector `g = Σ t_i` (300-dim) and retain it separately
  as a global residual feature (Section 5, Stage 3).
- Unknown fragment identifiers (not in the pretrained vocabulary — expect this for
  excipients containing metal ions, e.g. Mg/Ca/Al salts) → map to a single **learned
  "UNK fragment" embedding** (trainable, not zero-filled), so the model isn't blind to
  chemistry mol2vec's original training set didn't cover.
- mol2vec fragment embeddings themselves are **frozen** during training (pretrained,
  not fine-tuned). Only the projection/attention/head layers built on top are trained.

---

## 4. Excipient Encoder — Dual Branch (handles missing SMILES)

This is the critical piece for deployment-time generalization. It must be trained in
from the start, not added after training on `start_dataset.csv` — see rationale below.

**Why:** `start_dataset.csv` has 100% SMILES coverage. If the model is trained with
only a structural (mol2vec) input path, it never learns to interpret a missing-SMILES
case. Bolting on a fallback at inference time on the deployment data lands in a region
of embedding space the classifier was never trained to handle — undefined behavior, not
graceful degradation. The fallback must be trained via **modality dropout**.

**Per-excipient representation, three components:**

1. **Structural branch** `h_struct`: mol2vec token set (Section 3) → projected via
   attention (Section 5). If SMILES is missing, `h_struct` = a single **learned
   placeholder embedding** (trainable "unknown" vector, analogous to an `<UNK>` token),
   not zeros.
2. **Categorical branch** `h_cat`: functional-category embedding. Every excipient
   (including all 336 in `start_dataset.csv`) must be mapped to a functional category
   (diluent, binder, lubricant, disintegrant, etc. — reuse the existing 90-category HPE
   taxonomy already built for this project; multi-label if an excipient serves multiple
   roles) via UNII/name matching. This tag indexes a learned embedding table. Populated
   regardless of SMILES availability.
3. **Missing-flag** `m ∈ {0, 1}`: explicit binary scalar, 1 = SMILES available. Fed in
   alongside `h_struct`/`h_cat` so the model doesn't have to infer missingness
   implicitly from a placeholder value.

**Combination:** learned gate `g = sigmoid(W·m + b)` weighting `h_struct` vs `h_cat`
contribution to the final excipient embedding (preferred over plain concatenation — lets
the model learn "rely on category when structure is absent" as explicit behavior).

**Training-time modality dropout:** even though `start_dataset.csv` has full SMILES
coverage, during training randomly zero out the structural branch (replace with the
placeholder embedding, set `m=0`) for **15–30% of training examples** (tune as a
hyperparameter), forcing the classifier head to learn both regimes: (structure present)
and (structure absent, category-only). Vary the dropped fraction across epochs/batches
rather than fixing one static subset.

**API side is unaffected** — all APIs (training and deployment) have SMILES; only the
excipient encoder needs the dual-branch design.

---

## 5. Fusion & Classifier Architecture

**Stage 1 — Token projection**
- `Linear(300 → 128) → LayerNorm → GELU`, separate weights for API-token-projector and
  excipient-token-projector (not shared — API and excipient chemical spaces differ).
- No positional encoding (fragment set is unordered — bag of substructures).

**Stage 2 — Bidirectional cross-attention fusion**
- Multi-head attention: `d_model=128`, `num_heads=4`, `dropout=0.1`.
- Direction 1: Query = excipient tokens, Key/Value = API tokens.
- Direction 2: Query = API tokens, Key/Value = excipient tokens.
- Add & LayerNorm (residual) after each direction. **1 transformer layer only** —
  dataset is small (2133 training rows), do not stack more than 1–2 layers.
- Pool each refined token set via **attention pooling** (single learned query vector
  attends over the token set) rather than mean pooling.
- Output: `h_api_attn`, `h_exc_attn` ∈ R^128. (`h_exc_attn` is the output of the
  Section 4 dual-branch process, not raw mol2vec tokens directly.)

**Stage 3 — Merge with global mol2vec vector**
- Project pooled `g_api`, `g_exc` (300-dim) → 128-dim via separate linear layers.
- Concatenate: `h_api = [h_api_attn ; g_api_proj]` (256-dim), same pattern for
  excipient → `h_exc` (256-dim, incorporating the dual-branch output from Section 4).

**Stage 4 — Pair interaction layer**
```
pair_vec = [h_api ; h_exc ; h_api ⊙ h_exc ; |h_api − h_exc|]   → 1024-dim
```

**Stage 5 — Classifier head**
```
Linear(1024 → 256) → GELU → Dropout(0.3) → LayerNorm
Linear(256 → 64)   → GELU → Dropout(0.2)
Linear(64 → 1)                              # raw logit, no sigmoid
```

---

## 6. Lookup-Table Module (optional — implement only if a functional-group /
reaction-type table is provided; skip cleanly otherwise)

If available, integrate via all three of the following (not as a post-hoc override on
the model's output — that regresses to a rule-based system):

1. **Weak-label expansion**: auto-label additional API-excipient pairs (structures
   available, no ground-truth label) using known functional-group/reaction-type rules.
   Add to training data with a lower per-sample loss weight than the 344 real labeled
   positives (rule-derived labels are noisier).
2. **Engineered features**: per pair, compute `has_known_functional_group_pair` and
   `known_reaction_type` (Maillard / acid-base / complexation / redox / none) as
   categorical/binary features, concatenate into `pair_vec` (Stage 4) before the
   classifier head.
3. **Auxiliary task** (only if a reasonable volume of reaction-type labels exists):
   second output head predicting reaction mechanism type, joint loss with the main
   compatibility head. Not needed at inference time.

**Do not** use table lookups to gate/override the model's final prediction.

---

## 7. Loss Function — Asymmetric Loss (ASL), no SMOTE

- **Decision: ASL alone. Do not combine with SMOTE.** Both address the same 9.7%
  imbalance at different points (loss-level vs. data-level); stacking them requires
  detuning ASL back toward plain BCE to compensate, which defeats the purpose of using
  ASL at all.
- ASL hyperparameters: start at `gamma_neg=4, gamma_pos=1, clip=0.05` (standard
  defaults). Grid-search on validation PR-AUC: `gamma_neg ∈ {2,3,4}`,
  `gamma_pos ∈ {0,1}`, `clip ∈ {0, 0.05, 0.1}`.
- ASL operates on raw logits — classifier head (Stage 5) outputs a logit, no sigmoid
  applied before the loss.
- No class weighting, no resampling layered on top for the primary model.

---

## 8. Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-5 |
| LR schedule | ReduceLROnPlateau on val PR-AUC, factor 0.5, patience 8 |
| Batch size | 64 |
| Max epochs | 150, early stop patience 20 on val PR-AUC |
| Modality dropout rate (excipient structural branch) | 15–30%, tune on val |
| Trainable params | Token projectors, attention block, global projectors, MLP head, UNK fragment embedding, excipient placeholder embedding, functional-category embedding table, gate — mol2vec fragment embeddings frozen |

---

## 9. Evaluation

- Metrics: **PR-AUC, F1, MCC, recall on the positive (incompatible) class** — not
  accuracy (meaningless at 9.7% imbalance).
- Report on `val.csv` for model selection / threshold tuning, `test.csv` for final
  reported numbers, evaluated once.
- **Post-hoc threshold tuning** (independent of loss/architecture choice): sweep
  threshold ∈ (0, 1), step 0.001, maximize F1 on val; select and fix the threshold
  before evaluating on test.
- Sanity check: also report performance under a naive random 60/20/20 split (no
  grouping) on the same architecture, to quantify how much the paper's-style split
  inflates metrics via API-identity leakage vs. the grouped split used here.

---

## 10. Out of Scope for This Spec (future extensions, not required for v1)

- GNN (AttentiveFP) or chemical-language-model (ChemBERTa/MolFormer) encoders as
  alternatives to mol2vec — separate ladder experiment, not part of this build.
- Early-fusion (joint single-encoder) architecture as an alternative to the dual-encoder
  + cross-attention design specified here.
- SMOTE-based comparison row (SVM-SMOTE on pooled mol2vec vector + weighted BCE) — build
  as a separate script for comparison, not merged into this pipeline.
