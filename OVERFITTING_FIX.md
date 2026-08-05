# Model Improvement Spec — Fixing Overfitting

## Diagnosis (confirmed from training logs, not speculative)

The model overfits. Evidence from the actual run:

| | Epoch 3 (best) | Epoch 24 (early-stopped) |
|---|---|---|
| Train Loss | 0.0361 | 0.0069 |
| Val Loss | 0.0359 | 0.1365 |
| Val PR-AUC | 0.6091 | 0.4368 |

Train loss falls monotonically the entire run. Val loss falls only through epoch 2–3,
then rises every subsequent epoch (~4x by epoch 24). Val PR-AUC peaks at epoch 2–3 and
never recovers. Classic train/val divergence.

**Root causes:**
1. Model capacity is large relative to data: 567,769 trainable params vs. ~2,030–2,133
   training rows (~280 params per example), and only ~190 positive training examples to
   generalize the harder class from.
2. The train/val/test split is API-grouped (zero API overlap across splits, by design,
   to prevent leakage) — val genuinely contains chemistry never seen in training, so the
   generalization gap is real, not an artifact of a leaky split hiding it.
3. Early stopping patience (20) combined with EMA-smoothed PR-AUC as the tracked metric
   delays detecting the epoch-3 peak — the model trained 21 extra epochs past its best
   point before stopping.

---

## Required changes

### 1. Reduce model capacity

- Attention block `d_model`: **128 → 64**.
- Classifier head: change from
  `Linear(1024→256) → GELU → Dropout(0.3) → LayerNorm → Linear(256→64) → GELU → Dropout(0.2) → Linear(64→1)`
  to a single hidden layer:
  `Linear(new_input_dim→128) → GELU → Dropout(0.4) → Linear(128→1)`
- Pair interaction layer: drop the `|h_api − h_exc|` term. Change
  `pair_vec = [h_api ; h_exc ; h_api ⊙ h_exc ; |h_api − h_exc|]`
  to
  `pair_vec = [h_api ; h_exc ; h_api ⊙ h_exc]`
  (roughly 25% narrower input to the classifier head, on top of the d_model reduction
  above which already shrinks `h_api`/`h_exc` themselves).
- Recompute and log the new total trainable parameter count after these changes — expect
  a meaningful drop from 567,769; report the exact new number.

### 2. Fix early stopping

- Change early stopping to track **raw val PR-AUC**, not the EMA-smoothed value. Smoothing
  is appropriate for noisy metrics but here it delayed detecting a real, sustained
  degradation trend, not noise.
- Reduce `early_stop_patience` from 20 to **8**.
- Add a `min_delta` requirement (e.g. `0.002`) so trivial noise-level fluctuations don't
  reset the patience counter — an improvement only counts if it exceeds best-so-far by
  more than `min_delta`.
- Keep checkpointing on whichever metric early stopping now tracks (raw val PR-AUC), for
  consistency between what triggers the stop and what model gets saved/restored.

### 3. Increase regularization

- Dropout in the classifier head: increase per item 1's new single dropout layer to
  **0.4** (already specified above).
- Weight decay (AdamW): **1e-5 → 1e-4**.
- Add dropout inside the attention block on the attention weights themselves (not only
  after pooling), if not already present — same `dropout=0.1` used elsewhere in the
  attention block is fine as a starting value; this is a smaller lever than the other
  changes, apply it but don't over-index on tuning it first.

### 4. Multi-fold model selection (validation methodology change, not architecture)

- Instead of relying on a single fixed train/val split for model selection and early
  stopping decisions, run `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
  (same convention as the existing split script) on `train.csv` only — grouped on
  `split_group` / `API_CID` exactly as before, so no fold leaks an API across its own
  train/val division — train the same configuration on each of the 5 folds, and average
  val PR-AUC across all 5 before deciding "is this configuration better."
- Use 5 folds, not more or fewer: `train.csv` has ~191 positive examples, so at k=5 each
  held-out fold gets ~38 positives — enough for a PR-AUC estimate that isn't dominated by
  a handful of examples. k=10 would roughly halve that per-fold positive count (~19),
  trading estimate stability for more folds to average, a net wash at higher compute cost.
  k=3 keeps more positives per fold (~64) but only yields 3 estimates, less protection
  against one unlucky fold skewing the comparison.
- This does not replace the fixed `train.csv`/`val.csv`/`test.csv` files already in use
  for the final reported run — it's an additional check specifically for comparing
  configurations (e.g. "did shrinking the model actually help, or did I get lucky with
  which APIs landed in val this fold") given how few positive examples (~190) exist per
  fold. Implement as a separate `cross_validate_config()` utility, callable on demand, not
  as a change to the main training entrypoint.

---

## What NOT to change here

- Do not touch the mol2vec featurization pipeline, the dual-branch excipient encoder, the
  ASL loss, or the data split — none of those are implicated by this diagnosis. This is
  strictly an architecture-capacity and training-procedure fix.
- Do not add SMOTE or additional resampling as part of this fix — unrelated to
  overfitting, already decided against elsewhere.
- Do not increase `max_epochs` or loosen early stopping further — the direction of the
  fix is tighter regularization and earlier stopping, not more training.

---

## After implementing: what to check

Re-run training and confirm:
1. New trainable parameter count is meaningfully lower than 567,769 (report the number).
2. Val loss should track train loss more closely for longer before diverging (don't
   expect zero divergence — some gap is normal — but the epoch-3-then-degrade pattern
   should be less severe).
3. Early stopping should trigger closer to the actual best epoch, not ~20 epochs after it.
4. Report final val/test PR-AUC, F1, MCC, recall-on-positive-class, and the epoch at
   which the best model was checkpointed, exactly as the existing run summary format
   already does — no format changes needed, just confirm the numbers reflect a smaller,
   better-regularized model.

If val PR-AUC still peaks very early (epoch 2–4) and decays even after these changes,
that's a signal the ceiling is now bounded by data volume (~190 positive training
examples) rather than by model capacity or regularization — the next lever at that point
is mining more real positive training examples (separate, larger effort, not part of this
spec).
