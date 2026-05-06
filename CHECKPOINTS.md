# Checkpoint Files (Google Drive)

Model checkpoint files are intentionally excluded from this repository.

Place downloaded files in:
backend/app/models/checkpoints/

Expected files:
stage1_ner_best.pt
stage2_interaction_best.pt
stage3_severity_best.pt

Optional legacy fallback (runtime only if present):
best_model_3heads.pt