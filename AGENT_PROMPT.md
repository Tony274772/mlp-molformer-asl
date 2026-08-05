You are building a PyTorch deep learning model for API-excipient pharmaceutical
compatibility prediction (binary classification: compatible vs incompatible).

The full architecture, data handling, loss function, training procedure, and
evaluation protocol are specified in the attached `ARCHITECTURE.md`. Follow it exactly
— every design choice in it (encoder, fusion mechanism, loss function, split usage,
imbalance handling, missing-SMILES handling) was already decided and validated against
the actual dataset; do not substitute your own defaults or "improve" on it without
flagging the change explicitly first.

Build, in order:

1. **Data loading**: load `train.csv`, `val.csv`, `test.csv` as given (Section 2) —
   do not re-split or shuffle across files.
2. **mol2vec featurization pipeline** (Section 3): download/load `model_300dim.pkl`,
   implement fragment extraction producing a variable-length token set per molecule
   (not a single pooled vector), plus the separate pooled global vector. Implement the
   learned UNK-fragment embedding for out-of-vocabulary fragments.
3. **Excipient dual-branch encoder** (Section 4): structural branch (mol2vec tokens or
   learned placeholder), categorical branch (functional-category embedding table),
   missing-flag, gated combination. Implement modality dropout during training as
   specified. Flag clearly if functional-category labels for the 336 `start_dataset.csv`
   excipients aren't available yet — this branch needs that mapping to be meaningful,
   don't fake it with random labels.
4. **Fusion + classifier architecture** (Section 5): token projection, bidirectional
   cross-attention with attention pooling, global-vector merge, pair interaction layer,
   MLP classifier head, exactly as specified layer-by-layer.
5. **Lookup-table module** (Section 6): implement as a toggleable/optional component.
   If no lookup table file is provided, skip it cleanly (the model must run correctly
   with this module absent — don't make it a hard dependency).
6. **Loss** (Section 7): Asymmetric Loss (ASL) implementation operating on raw logits,
   with the specified hyperparameter grid available for tuning. Do not add SMOTE or
   any other resampling into this training path.
7. **Training loop** (Section 8): AdamW, ReduceLROnPlateau, early stopping on val
   PR-AUC, exact hyperparameters as listed. Log train/val loss and val PR-AUC/F1/MCC
   per epoch.
8. **Evaluation** (Section 9): PR-AUC, F1, MCC, recall-on-positive-class on val (for
   threshold tuning) and test (final numbers, evaluate once). Implement the F1-max
   threshold sweep. Also implement the naive-random-split sanity check described in
   Section 9 as a separate, clearly-labeled comparison run.

Deliverables:
- Modular, runnable code (not a single notebook cell dump) — separate files/functions
  for featurization, model definition, training loop, evaluation.
- A config object/file exposing every hyperparameter in Section 8 so they can be swept
  without editing model code.
- Print/log a full run summary at the end (final val and test metrics, threshold used,
  confusion matrix) in the same spirit as the split script's diagnostic output — I want
  to see exactly what happened, not just a final number.

If anything in `ARCHITECTURE.md` is ambiguous or you need to make an implementation
decision it doesn't cover, state the assumption explicitly in a comment at that point in
the code rather than silently picking one.
