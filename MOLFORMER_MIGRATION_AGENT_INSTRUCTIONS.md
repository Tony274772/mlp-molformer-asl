# MoLFormer-XL Migration — Agent Instructions

## Status quo (verified against the uploaded project)

- `src/config.py`, `src/model.py`, `src/dataset.py`, `src/featurization.py` are still the
  **original mol2vec version** — nothing has been ported yet. This document is the full spec
  for that port.
- MoLFormer-XL-both-10pct is already downloaded, but in the wrong place:
  `MoLFormer-XL/models/molformer-xl-both-10pct/` (a stray top-level folder), not under the
  project's `models/` directory where `config.py` conventions expect pretrained assets
  (`models/model_300dim.pkl`, `models/descriptor_norm_stats.json`).
- Two duplicate copies of `download_molformer.py` exist (`models/download_molformer.py` and
  `MoLFormer-XL/download_molformer.py`).
- Data is unchanged and confirmed clean: `data/train.csv` (2030 rows, 191 positive),
  `data/val.csv` (842 rows, 76 positive), `data/test.csv` (672 rows, 77 positive). **Zero
  missing/empty `Excipient_Smiles` or `API_Smiles` in any split.** A separate `data/naive/`
  split exists (random, not grouped) — out of scope, do not touch.
- `FIXES.md` / `OVERFITTING_FIX.md` document a **confirmed, already-diagnosed overfitting
  problem** on the mol2vec model: 567,769 trainable params vs. ~190 positive training rows,
  val PR-AUC peaking at epoch 3 then degrading while train loss keeps falling. **This is
  critical context for Section 4 below** — several of the changes requested here
  (wider projections, more nonlinearity) *increase* capacity at exactly the point where
  capacity was already the diagnosed problem. Do not implement Section 3/4 without also
  implementing the regularization changes in Section 5 — they are not optional add-ons, they
  are required to not reproduce the same failure mode at a larger scale.
- This document **supersedes** `ARCHITECTURE.md` Sections 3–5 (mol2vec featurization,
  excipient structural encoder, fusion) and `AGENT_PROMPT.md` steps 2–4, which describe the
  mol2vec pipeline being replaced. Sections 1 (problem statement), 2 (data split), 6
  (missing-SMILES/lookup table — still not applicable, skip), 7 (loss), 8 (training loop),
  and 9 (evaluation protocol) of `ARCHITECTURE.md` are **unchanged and still authoritative**.

---

## 0. Cleanup and verified install (do this first)

1. Move the model into the correct location:
   ```
   mv MoLFormer-XL/models/molformer-xl-both-10pct models/molformer-xl-both-10pct
   ```
2. Delete the now-empty `MoLFormer-XL/` folder and the duplicate `MoLFormer-XL/download_molformer.py`.
   Keep one canonical copy at `models/download_molformer.py`.
3. Verify the moved folder contains exactly these files (this repo does **not** ship
   `tokenization_molformer.py`/`tokenization_molformer_fast.py` — the tokenizer loads
   directly from `tokenizer.json`/`tokenizer_config.json`, do not add a check for those
   files if reusing/adapting the download script):
   `config.json`, `configuration_molformer.py`, `modeling_molformer.py`, `model.safetensors`
   (~187MB), `tokenizer.json`, `tokenizer_config.json`. `.gitattributes`/`README.md` are
   harmless extras, fine to keep or drop.
4. Run a smoke test from the new path before writing any other code:
   ```python
   from transformers import AutoModel, AutoTokenizer
   import torch
   m = AutoModel.from_pretrained("models/molformer-xl-both-10pct", deterministic_eval=True, trust_remote_code=True)
   t = AutoTokenizer.from_pretrained("models/molformer-xl-both-10pct", trust_remote_code=True)
   out = t(["Cn1c(=O)c2c(ncn2C)n(C)c1=O"], return_tensors="pt")
   with torch.no_grad():
       r = m(**out)
   print(r.pooler_output.shape, r.last_hidden_state.shape)   # expect [1,768] and [1,L,768]
   ```
   Do not proceed past this step until it passes. If `pooler_output` is absent or the model
   loads via a different output class, inspect `modeling_molformer.py` directly and adapt
   Section 2 below to whatever the actual output object provides.
5. **Also verify what "CLS-equivalent" token the tokenizer actually produces** before
   implementing Section 2 — check `t.cls_token`, `t.bos_token`, and print
   `t(["CCO"])["input_ids"]` to see whether a fixed special token id is prepended to every
   sequence. Section 2 depends on knowing whether such a token genuinely exists in
   `last_hidden_state[:, 0, :]` or whether `pooler_output` is a separately-computed summary
   (e.g. mean-pool + dense + tanh) with no corresponding sequence position. Both cases are
   handled below, but which code path to use depends on this check.
6. Freeze the entire MoLFormer backbone — do not fine-tune it in this version:
   ```python
   for p in molformer_model.parameters():
       p.requires_grad = False
   molformer_model.eval()
   ```
   Rationale: 44M+ backbone params against 2030 training rows is not viable to fine-tune,
   and it isn't what was asked for — this migration is about replacing frozen mol2vec
   features with better frozen MoLFormer features, not about fine-tuning a transformer on a
   dataset two orders of magnitude too small for it.
7. **Delete** the old mol2vec checkpoints and model files outright — this folder is a
   dedicated copy for the MoLFormer version, not the original project, so there's no
   rollback reason to keep them around, and they will not load into the new model anyway
   (state dict shapes change completely):
   ```
   rm -rf checkpoints checkpoints_asl checkpoints_naive
   rm -rf __pycache__ src/__pycache__ src/.ipynb_checkpoints
   rm models/model_300dim.pkl
   ```
   Keep `metrics/run_metrics.json` and `metrics/cv_metrics.json` — these are numbers, not
   model artifacts, and they're the mol2vec baseline needed to judge whether the new
   architecture is actually better (Section 6). Write new run metrics to a new path
   (`metrics/molformer/`) rather than overwriting these.
8. Delete the old mol2vec code paths now, not later — same reasoning as above, no rollback
   need in this copied folder:
   - Delete `src/featurization.py` (mol2vec featurizer — fully superseded by
     `src/molformer_featurization.py` in Section 1).
   - Remove `gensim` and `mol2vec` from `requirements.txt`; add `transformers` and
     `huggingface_hub`.
   - Remove `mol2vec_model_path` and `mol2vec_dim` from `config.py` entirely (Section 1
     originally said to leave these as harmless dead fields — that guidance is superseded by
     this section: delete them, since this folder has no reason to support both pipelines).
   - Search the repo for any remaining `import` of `Mol2VecFeaturizer` or
     `src.featurization` (`dataset.py`, `main.py`, `cross_validate.py`, `smoke_test.py`,
     `sanity_check.py` are the likely locations) and update each to the new
     `MolFormerFeaturizer` import — a leftover import of a deleted module will crash on the
     first run, so this must be done in the same pass as the deletions above, not after.

---

## 1. Featurizer replacement

Create `src/molformer_featurization.py`. `src/featurization.py` (the old mol2vec
featurizer) is deleted per Section 0.8 — build the new file standalone rather than editing
the old one in place, since the internals (tokenizer-based vs. fragment-vocab-based) don't
share meaningful structure.

Replace `Mol2VecFeaturizer` with a `MolFormerFeaturizer(nn.Module)` that:

- Loads the model + tokenizer from `config.molformer_model_path` once, freezes all backbone
  params (Section 0.6).
- `forward(smiles_list)` tokenizes the batch (`padding=True, return_tensors="pt"`), runs the
  frozen backbone under `torch.no_grad()`, and returns:
  - `last_hidden_state` — `[B, L, 768]` (replaces `padded_tokens`)
  - `pooler_output` — `[B, 768]` (replaces the old sum-pooled `global` vector — **this comes
    from the model directly, do not hand-compute a global vector by pooling yourself**)
  - `key_padding_mask` — `[B, L]` bool, **True = padding** (this is the inverse of HF's
    `attention_mask`, which is 1=real/0=pad — invert it: `mask = attention_mask == 0`, to
    match the existing `key_padding_mask` convention already used in `model.py`'s
    `nn.MultiheadAttention` calls)
  - `num_tokens` — `[B]` real token counts (sum of `attention_mask` per row)
- Keep the same SMILES-string caching pattern the mol2vec featurizer used
  (`self.cache[smiles] = ...`) — cache the tokenized+forwarded result per unique SMILES
  string, since the same excipient SMILES recurs across many rows.
- Empty-string handling: unlike mol2vec, an empty SMILES fed to a tokenizer will not
  cleanly produce a zero-token result — do not feed `""` to the tokenizer at all. Since
  Section 4 sets modality dropout to 0 and there is no missing-SMILES data, the empty-string
  path should never actually be exercised in this version, but keep the guard: if `exc_smi`
  is empty/NaN, short-circuit and return the learned placeholder path in `model.py`
  (Section 2) without calling the tokenizer, exactly as the missing-flag mechanism already
  does structurally — just don't rely on the featurizer to handle `""` gracefully.
- Delete the mol2vec-specific OOV/UNK vocabulary logic entirely — subword tokenization has
  no OOV-fragment problem, there's nothing to port here.
- **No manual pooling code should exist in this file at all.** Any place in the old
  `featurization.py` that did `global_vecs[b] += known_embs.sum(dim=0)` (sum-pooling) has no
  equivalent here — `pooler_output` replaces it entirely. If you find yourself writing a
  `.mean(dim=1)` or `.sum(dim=1)` anywhere in this file, stop — that pooling belongs in
  `model.py`'s cross-attention pooling (Section 2), not in the featurizer.

Update `config.py`:
```python
molformer_model_path: str = "models/molformer-xl-both-10pct"
molformer_dim: int = 768          # replaces mol2vec_dim for this pipeline
```
Remove `mol2vec_model_path`/`mol2vec_dim` from the dataclass in this same pass, per
Section 0.8 — this folder has no rollback need, so there's no reason to keep dead config
fields around.

---

## 2. Model architecture rewrite (`src/model.py`)

This is the major change. Three requested changes, addressed together since they interact:

### 2a. CLS-before-cross-attention (not global-after-attention)

**Old design:** cross-attention ran over structural tokens only; a separate "global" vector
(sum-pooled mol2vec embedding → plain `Linear(300,12)`) was computed independently and
concatenated in *after* attention pooling, bypassing attention entirely (Stage 3 in the old
code, disconnected from Stage 2).

**New design:** the global/CLS representation must **participate in cross-attention itself**,
not bypass it. Concretely:

1. Take MoLFormer's `pooler_output [B, 768]` for each molecule and project it through the
   *same* token-projection layer used for the sequence tokens (Section 2b) — so it lives in
   the same 768→proj_dim space as every other token.
2. Prepend this projected CLS vector as position 0 of the token sequence for each molecule:
   `sequence = cat([cls_token.unsqueeze(1), token_sequence], dim=1)` → `[B, L+1, proj_dim]`.
3. Extend the padding mask with a `False` (never-masked) entry at position 0 for the CLS slot.
4. Run the existing bidirectional cross-attention (`attn_exc_to_api`, `attn_api_to_exc`) over
   these extended sequences — so the CLS token attends to and is attended by the other
   molecule's actual structural tokens, and ordinary tokens can also attend to the other
   molecule's CLS summary. This is the actual difference from the old design: global context
   now flows through attention instead of being concatenated in afterward.
5. Take the model's final per-molecule representation from **position 0 of the
   post-attention sequence** (the CLS position), instead of the old learned-query attention
   pooling (`api_pool_query`/`exc_pool_query`/`_attention_pool`). Remove the separate pooling
   query mechanism entirely — CLS-token pooling replaces it; keeping both would be redundant
   and would reintroduce the same "compute global separately, glue it on" pattern this change
   is meant to eliminate.

Regarding Section 0.5's check: if the tokenizer truly has no BOS/CLS token and
`last_hidden_state[:, 0, :]` is just an ordinary first fragment token (not a special
summary), use `pooler_output` as computed above regardless — it's still a valid learned
whole-molecule summary from the model's own pooling head, and prepending it as an extra
"synthetic" sequence position for cross-attention purposes is still the correct
architecture; it does not require the position to have been a real CLS token during
MoLFormer's own pretraining.

### 2b. Fix the 300→12 bottleneck; no sum pooling anywhere

- `mol2vec_dim=300` → `molformer_dim=768`. The old `proj_dim=12` (a ~64x compression from
  300, worse now at ~64x from 768) was too aggressive. Replace it with:
  - `proj_dim: int = 128`
  - Token projection becomes a genuine two-layer nonlinear projection, not a single Linear:
    ```
    Linear(768, 256) → LayerNorm(256) → GELU → Dropout(proj_dropout) →
    Linear(256, 128) → LayerNorm(128) → GELU
    ```
    Apply this identically to `api_proj` and `exc_proj` (separate weights, same shape), and
    reuse the exact same module for projecting the CLS/`pooler_output` vector in Section 2a
    (same weights — the CLS token and the structural tokens must live in the same learned
    space, that's the point of sharing the projection).
  - `num_heads: int = 8` (128/8 = 16 per head — check this divides cleanly, adjust `proj_dim`
    if you change `num_heads`).
- Confirm no `.sum(dim=...)` pooling survives anywhere in `model.py` or the new featurizer.
  The only pooling operation in the new architecture is the CLS-position extraction in 2a
  (which is attention-based, not a sum/mean at all) — there should be no separate mean-pool
  step needed since `pooler_output` already comes pre-pooled from MoLFormer. If you add any
  additional pooling for some other purpose, it must be mean or attention-weighted, never sum.

### 2c. Descriptors — use them properly, not as an afterthought bolt-on

- `desc_proj_dim: int = 8` → `24` (proportional to the `proj_dim` increase; still deliberately
  smaller than the structural branch since the structural branch is the primary signal and
  descriptors are meant to be a compact supplementary signal, not overshadow it).
- Descriptor projection: `Linear(21, 32) → LayerNorm(32) → GELU → Dropout(desc_dropout) →
  Linear(32, 24) → LayerNorm(24) → GELU` — same two-layer-with-nonlinearity treatment as the
  structural projection, for consistency and because a single Linear on 21 raw RDKit
  descriptors is exactly the same "too-linear" problem being fixed elsewhere.
- Concatenate descriptor projection into `h_api`/`h_exc` exactly as before (after the
  structural/CLS representation), just with the new dims.

### 2d. Pair combination — use a richer, standard pairwise-fusion form

- Old: `pair_vec = [h_api, h_exc, h_api * h_exc]`.
- New: add the elementwise absolute difference, a standard and well-validated way to combine
  two vectors for a pairwise relationship task (compatible/incompatible is fundamentally a
  relationship between two embeddings, not a simple concatenation problem):
  ```
  h_exc_core = h_exc[:, :dim_api]   # drop the missing-flag bit, same as current code
  interaction = h_api * h_exc_core
  difference  = torch.abs(h_api - h_exc_core)
  pair_vec = torch.cat([h_api, h_exc, interaction, difference], dim=-1)
  ```
- Resulting dims with the numbers above: `h_api` = 128(CLS/structural) + 24(desc) = 152.
  `h_exc` = 152 + 1(avail flag) = 153. `interaction` = 152, `difference` = 152.
  `pair_dim` = 152 + 153 + 152 + 152 = **609**.

### 2e. Classifier head — more nonlinearity, sized to the new pair_dim

- Old: `Linear(97,16) → GELU → Dropout(0.4) → Linear(16,1)` — a single hidden layer.
- New: two hidden layers, both with GELU + dropout, since a single 16-unit layer cannot
  meaningfully process a 609-dim input:
  ```
  Linear(609, 128) → GELU → Dropout(clf_dropout_1) →
  Linear(128, 64)  → GELU → Dropout(clf_dropout_2) →
  Linear(64, 1)
  ```
  Add `clf_hidden_dim_2: int = 64` and `clf_dropout_2: float = 0.4` to `config.py` alongside
  the existing `clf_hidden_dim` (repurposed as the first hidden layer, set to 128) and
  `clf_dropout_1`.

---

## 3. Config changes — full before/after table

| Parameter | Old (mol2vec) | New (MoLFormer) | Note |
|---|---|---|---|
| `mol2vec_dim` | 300 | — | kept, unused, see Section 1 |
| `molformer_dim` | — | 768 | new |
| `molformer_model_path` | — | `"models/molformer-xl-both-10pct"` | new |
| `proj_dim` | 12 | 128 | Section 2b |
| `num_heads` | 2 | 8 | must divide `proj_dim` |
| `attn_dropout` | 0.1 | 0.15 | Section 5 |
| `proj_dropout` | — | 0.15 | new, Section 2b |
| `desc_proj_dim` | 8 | 24 | Section 2c |
| `desc_dropout` | — | 0.15 | new, Section 2c |
| `clf_hidden_dim` | 16 | 128 | Section 2e, first hidden layer |
| `clf_hidden_dim_2` | — | 64 | new, second hidden layer |
| `clf_dropout_1` | 0.4 | 0.5 | Section 5 |
| `clf_dropout_2` | — | 0.4 | new |
| `modality_dropout_rate` | 0.2 | **0.0** | Section 4 |
| `lr` | 3e-4 | 1.5e-4 | Section 5 |
| `weight_decay` | 1e-4 | 8e-4 | Section 5 |
| `early_stop_patience` | 8 | 6 | Section 5 |
| `num_descriptors` | 21 | 21 | unchanged |

---

## 4. Modality dropout

Set `modality_dropout_rate: float = 0.0` in `config.py`. Confirmed rationale: `train.csv`,
`val.csv`, and `test.csv` all have 0 missing/empty `Excipient_Smiles` and `API_Smiles`
rows — there is nothing for the modality-dropout mechanism to simulate robustness against
that reflects a real deployment scenario for this dataset, and training with it on wastes
~20% of each epoch's structural signal for no corresponding benefit at eval time (dropout is
already disabled at eval via `is_train=False` in `dataset.py`, so it only ever fires during
training).

**Do not delete the mechanism.** `dataset.py`'s modality-dropout branch and `model.py`'s
missing-flag/placeholder logic (`exc_placeholder`, `exc_global_placeholder`, `exc_available`
flag) must stay in the code, just inert at `rate=0.0` — this keeps the architecture ready for
a future version where missing-SMILES excipients genuinely appear (e.g. if this model is
later pointed at `oral_formulations_final.csv`, which per `ARCHITECTURE.md` Section 1 does
have ~40% missing-SMILES excipients). Note: `exc_global_placeholder` was 300-dim
(`mol2vec_dim`), must be resized to `proj_dim` now that it's injected post-projection/CLS
rather than pre-projection — check where in the forward pass the placeholder swap-in happens
relative to the new CLS-prepend step in Section 2a and make sure the dimensions still line up.

---

## 5. Nonlinearity, dropout, and LR — why these specific numbers

This section exists because Sections 2b/2c/2e substantially increase trainable parameter
count (single Linear layers → two-layer nonlinear MLPs throughout, proj_dim 12→128,
classifier 97→609 input dim with an added hidden layer) at the same time
`FIXES.md`/`OVERFITTING_FIX.md` already diagnosed overfitting as the dominant failure mode of
the *smaller* mol2vec model (567K params vs. ~190 positive training rows). More capacity
without more regularization on the same ~2030-row, ~9.4%-positive dataset will very likely
reproduce the same epoch-3-peak-then-degrade pattern documented in those files, just worse.
Apply all of the following together, not selectively:

- **Dropout raised everywhere a new layer was added**: `attn_dropout` 0.1→0.15,
  `clf_dropout_1` 0.4→0.5, new `clf_dropout_2`=0.4, new `proj_dropout`=0.15, new
  `desc_dropout`=0.15. Every added Linear layer in Sections 2b/2c/2e must have a dropout
  immediately after its activation, not just the final classifier layer as before.
- **Weight decay raised**: 1e-4 → 8e-4. AdamW weight decay is the other primary lever against
  the diagnosed overfitting pattern, per `FIXES.md`'s own recommendation to increase it as a
  non-deviating, already-sanctioned sweep parameter.
- **LR lowered slightly**: 3e-4 → 1.5e-4. With the backbone frozen (Section 0.6), this LR
  only touches the projection/attention/classifier layers — but those layers are now
  substantially larger, and a lower LR paired with `ReduceLROnPlateau` (unchanged,
  patience=8, factor=0.5) gives the optimizer more room to find a stable minimum before the
  scheduler starts cutting it, rather than overshooting into the sharp overfit regime seen
  before.
- **Early-stop patience lowered**: 8 → 6. The prior run's best epoch was epoch 3 out of a run
  that continued to epoch 23+ before stopping — patience=8 let it run needlessly long past
  the peak. With more capacity now, expect the same early-peak pattern to still apply; don't
  give it more room to overfit than before just because the architecture changed.
- These are starting values, not final ones — after the first full retrain, run
  `cross_validate.py` (already fold-aware and grouped by API_CID, no changes needed there
  beyond it picking up the new `Config` and `APIExcipientModel`) and compare fold-to-fold
  PR-AUC variance against the old `metrics/cv_metrics.json` baseline (fold PR-AUCs
  0.42–0.63, std 0.077). If variance is still high or val/test PR-AUC still diverges early,
  the next lever to pull is `proj_dim` (128→64) before touching dropout/weight_decay
  further — a smaller-but-still-nonlinear projection may be more appropriate than a wide one
  given the dataset size, and that's a cheaper, more diagnostic change to test first than
  further blind regularization increases.

---

## 6. Verification checklist (in order)

1. Section 0 smoke test passes (model loads, forward pass produces `[B,768]`/`[B,L,768]`).
2. `MolFormerFeaturizer` unit-tested standalone on a handful of SMILES from `data/train.csv`
   — including at least one ionic/inorganic excipient (e.g. `[OH-].[Na+]`,
   `O=[Al]O[Si](=O)O[Si](=O)O[Al]=O`) — confirm no tokenizer exceptions and no all-zero
   `attention_mask` rows.
3. `APIExcipientModel(config)` forward pass on one real batch from `train.csv` runs without
   shape errors end-to-end (this will surface any CLS-prepend/mask-dimension mismatches from
   Section 2a immediately).
4. `count_parameters(model)` (already in `utils.py`) — record the new trainable param count
   and compare it explicitly against the old 567,769 in `FIXES.md`, in whatever run log or
   PR description accompanies this change. This number should be reported, not just silently
   changed.
5. Full run via `main.py`, then `cross_validate.py`. Save outputs to `metrics/molformer/`
   (new path, per Section 0.7) — do not overwrite the existing mol2vec baseline files.
6. Compare new val/test PR-AUC, F1, MCC, and the epoch-of-best-checkpoint against the
   mol2vec baseline (`metrics/run_metrics.json`, `metrics/cv_metrics.json`) and against the
   overfitting symptoms described in `FIXES.md`/`OVERFITTING_FIX.md` (train/val loss
   divergence curve, best-epoch position). Report all four together — a higher PR-AUC with
   the same early-epoch-peak-then-degrade shape is not a fixed problem, just a bigger number
   on a still-broken training curve.
