You're working on this copied version of the API-Excipient Compatibility project, whose job
is to migrate the model from mol2vec embeddings to MoLFormer-XL (IBM) embeddings, with
several deliberate architecture changes on top of the swap.

Full, authoritative instructions are in `MOLFORMER_MIGRATION_AGENT_INSTRUCTIONS.md` — read it
completely before writing any code. It covers, in order: verifying/relocating the already-
downloaded MoLFormer-XL model, deleting the old mol2vec checkpoints/model file/featurizer
code (this is a dedicated copy of the project for this migration — no rollback path is
needed, delete rather than archive), the new featurizer, the new model architecture (CLS
token fused through cross-attention instead of a bypassed post-attention global vector, no
sum-pooling anywhere, wider nonlinear projections instead of the old 300→12 linear
bottleneck, descriptors properly sized and projected instead of bolted on, and a richer
pairwise fusion combining API/excipient representations), the full config parameter
before/after table, modality dropout set to 0 (kept in code, just inert — dataset has zero
missing SMILES), and the dropout/weight-decay/LR changes required to offset the added model
capacity given the already-documented overfitting problem in `FIXES.md`/`OVERFITTING_FIX.md`.

That document supersedes `ARCHITECTURE.md` Sections 3–5 and `AGENT_PROMPT.md` steps 2–4 (the
mol2vec-specific parts). Everything else in `ARCHITECTURE.md` — problem statement, data
split, loss function, training loop, evaluation protocol — is unchanged and still applies.

Follow the instructions file's section order exactly: cleanup and verified install first
(Section 0), then featurizer (Section 1), then model (Section 2), then config (Section 3),
then modality dropout (Section 4), then the regularization changes (Section 5) — Sections
2 and 5 are not independent; do not implement the capacity increases in Section 2 without
also implementing the regularization changes in Section 5 in the same pass, since Section 5
exists specifically to offset Section 2's added capacity against a dataset with only ~190
positive training examples.

Run the full verification checklist in Section 6 before reporting this as done — that
includes a standalone featurizer test on at least one ionic/inorganic excipient SMILES, a
full forward pass shape check, a reported before/after trainable parameter count, and a full
train + cross-validation run compared against the existing `metrics/run_metrics.json` /
`metrics/cv_metrics.json` mol2vec baseline (kept on disk specifically for this comparison —
do not delete or overwrite them). If anything in the instructions file is ambiguous or
conflicts with what you find in the actual code, flag it explicitly rather than silently
picking a default, per this project's existing convention in `AGENT_PROMPT.md`.
