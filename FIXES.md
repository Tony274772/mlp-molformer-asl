# Overfitting Fixes — Agent Instructions

Context: training run shows val PR-AUC peaking at epoch 3 (0.6093) and degrading
steadily afterward while train loss keeps falling (0.048 → 0.0067 by epoch 23).
Test PR-AUC (0.5499) and F1 (0.4569) are noticeably below val (0.6093 / 0.5397).
This is overfitting driven by model capacity vs. dataset size (~207 positive
training rows against 567K trainable params), not a bug in the checkpointing —
`train.py` already saves/restores the best-val-PR-AUC checkpoint correctly, so
the epoch-3 model is what gets evaluated. The functional-category branch
(Section 4 of ARCHITECTURE.md) is intentionally **out of scope** — do not add
it as part of these fixes.

Apply the following changes. Where a change deviates from the literal spec in
ARCHITECTURE.md or AGENT_PROMPT.md, it's flagged explicitly below — surface
that flag in a code comment at the point of the change, per AGENT_PROMPT.md's
instruction to flag deviations rather than silently picking one.

---

## 1. Increase weight decay (no spec deviation — already a Section 8 sweep param)

In `src/config.py`:

```python
weight_decay: float = 1e-3   # was 1e-5 — too weak for ~207 positive training rows
```

`1e-5` is negligible regularization at this dataset size. This is the
highest-priority, lowest-risk fix.

---

## 2. Add gradient clipping (no spec deviation — `train.py` left this unimplemented)

In `src/train.py`, in the training loop, right after `loss.backward()`:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`max_norm=1.0` is a reasonable starting point; expose it as a config field
(`grad_clip_norm: float = 1.0`) so it can be swept like other hyperparameters.

---

## 3. Smooth the early-stopping / checkpoint signal (⚠ deviation — flag in code)

Val PR-AUC is computed over 705 rows with ~68 positives — a noisy metric where
single-epoch spikes can look like real peaks. Currently `train_model` compares
the raw per-epoch val PR-AUC against the running best. Change it to compare a
short moving average instead:

```python
# DEVIATION FROM SPEC: Section 8 says "early stopping on val PR-AUC" using the
# raw per-epoch value. We smooth over a 3-epoch window before comparing, since
# val PR-AUC is noisy at this sample size (~68 positives) and single-epoch
# spikes were triggering premature "best model" saves. Flagging per
# AGENT_PROMPT.md's instruction to surface spec deviations explicitly.
pr_auc_history.append(val_pr_auc)
smoothed_pr_auc = sum(pr_auc_history[-3:]) / len(pr_auc_history[-3:])
if smoothed_pr_auc > best_val_pr_auc:
    best_val_pr_auc = smoothed_pr_auc
    ...
```

Keep logging the raw per-epoch val PR-AUC/F1/MCC as before — only the
checkpoint/early-stopping decision uses the smoothed value.

---

## 4. Run the ASL hyperparameter grid (already specified, just not run yet)

Section 7 of `ARCHITECTURE.md` specifies a grid that hasn't been swept —
currently running only the defaults (`gamma_neg=4, gamma_pos=1, clip=0.05`):

```
gamma_neg ∈ {2, 3, 4}
gamma_pos ∈ {0, 1}
clip      ∈ {0, 0.05, 0.1}
```

Add a small sweep script (or extend `main.py` with a `--sweep-asl` flag) that
loops over this grid, trains with fixes #1–#3 applied, and reports val PR-AUC
per combination so the best setting can be selected before final test
evaluation. `gamma_neg=4` (aggressive negative suppression) may itself be
contributing to early overfitting at this sample size — don't assume the
defaults are correct.

---

## 5. (Optional, try only if #1–#4 don't fix it) Shrink the first classifier layer (⚠ deviation — flag in code)

`model.py`'s `classifier[0]` is `Linear(1025 → 256)` — roughly 262K of the
model's 567K trainable params sit in this single layer, and `pair_vec`'s
multiply/abs-diff terms are nonlinear recombinations of the same underlying
256+257 values, so the layer has more capacity than the input's real
information content justifies. If val PR-AUC still peaks within the first few
epochs after #1–#4, try:

```python
# DEVIATION FROM SPEC: Section 5 specifies Linear(1024 → 256) as the first
# classifier layer. Reduced output width to 128 to cut classifier-head
# capacity, since ~262K params in this single layer relative to ~207 positive
# training examples was identified as a likely overfitting driver. Flagging
# per AGENT_PROMPT.md.
nn.Linear(pair_dim, 128),
nn.GELU(),
nn.Dropout(config.clf_dropout_1),
nn.LayerNorm(128),

nn.Linear(128, 64),
...
```

Only apply this if the earlier, non-architectural fixes don't move the
val-PR-AUC-peaks-too-early symptom — it's a genuine spec deviation and should
be tried last, not first.

---

## Verification after applying

- Re-run `python main.py` and check that val PR-AUC now improves for more than
  3–5 epochs before plateauing/declining, and that the train/val loss gap at
  the best epoch is smaller than in the original run.
- Confirm test PR-AUC/F1 move closer to val PR-AUC/F1 (a persistent gap here
  is expected given the leakage-safe grouped split — Section 2 — but it
  shouldn't be as large as the current 0.61 → 0.55 / 0.54 → 0.46 drop).
- Re-run `sanity_check.py` afterward to confirm the naive-split vs.
  grouped-split comparison still behaves as expected (naive split inflated,
  grouped split lower but now less overfit).
