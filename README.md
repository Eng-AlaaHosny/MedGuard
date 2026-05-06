## MedGuard Project Report

This file explains what I built, how I trained it, and what the current results are.

## Quick Summary

- The model is trained in 3 stages: NER, interaction type, then severity.
- Stage 3 uses **DDInter curated severity labels** (not DrugBank keyword labels).
- Entity linking uses a deterministic 3-layer resolver.
- Stage 3 uses **Balanced Softmax** to handle class imbalance.
- Runtime uses Stage 3 as the main model and Stage 2 as a dedicated interaction model.

## Project Goal

MedGuard predicts:

- Drug entities (NER head)
- DDI interaction type (interaction head)
- DDI severity (severity head)

## Model Architecture (Short Version)

- Backbone: `emilyalsentzer/Bio_ClinicalBERT`
- Heads:
  - NER head (token classification: `O`, `B-DRUG`, `I-DRUG`)
  - Interaction head (`false`, `mechanism`, `effect`, `advise`, `int`)
  - Severity head (`safe`, `caution`, `warning`, `danger`)
- Extra features:
  - KG embedding per drug (128-dim)
  - Lipinski features per drug (5 values)
- Fusion:
  - For each drug, model fuses BERT span + KG + Lipinski
  - Then predicts interaction and severity from pair representation

## Training Design

### Stage 1 (NER)
- Trains the NER head for drug entities.
- Best checkpoint: `backend/app/models/checkpoints/stage1_ner_best.pt`

### Stage 2 (Interaction)
- Trains interaction type (`false`, `mechanism`, `effect`, `advise`, `int`).
- Loss in current code: `CrossEntropyLoss(weight=ddi_w)`.
- Best checkpoint: `backend/app/models/checkpoints/stage2_interaction_best.pt`

### Stage 3 (Severity)
- Trains severity (`safe`, `caution`, `warning`, `danger`) using DDInter-curated labels.
- Uses deterministic name linking + coverage reporting.
- Uses Balanced Softmax for long-tail labels.
- Best checkpoint: `backend/app/models/checkpoints/stage3_severity_best.pt`

## Training Defaults (from current code)

- Stage 1 defaults:
  - epochs: 5
  - batch size: 16
  - lr: 2e-5
  - val split: 0.15
- Stage 2 defaults:
  - epochs: 10
  - batch size: 4
  - lr: 1e-4
  - gradient accumulation: 4
  - val split: 0.15
- Stage 3 defaults:
  - epochs: 10
  - batch size: 4
  - lr: 1e-4
  - gradient accumulation: 4
  - val split: 0.15

## Runtime Checkpoint Policy (matches code)

From `backend/main.py`:

- Main model checkpoints (priority):
  1. `backend/app/models/checkpoints/stage3_severity_best.pt`
  2. `backend/app/models/checkpoints/best_model_3heads.pt` (legacy fallback)
- Interaction model:
  - `backend/app/models/checkpoints/stage2_interaction_best.pt`
  - if missing, interaction falls back to main model
- `stage1_ner_best.pt` is training-only (used to initialize Stage 2 training).

## Inference Behavior (important)

- Main endpoint takes explicit drug names: `drug_a`, `drug_b`.
- Optional text can be provided (`text` field).
- If text is empty, backend creates a short template sentence for inference.
- No training-corpus sentence substitution is used at inference.

## Data Used

- DDI Corpus (XML)
- DrugBank XML (`full database.xml`)
- DrugBank SQLite (`drugbank.db`)
- DDInter CSV files (`ddinter_code_*.csv`)
- KG file (`knowledge_graph.pkl`)
- Lipinski file (`DB_compounds_lipinski.csv`)

## Data / Build Pipeline

1. Parse DrugBank XML and build SQLite DB:
   - `python -m app.data.drugbank_processor`
   - Produces `backend/app/data/drugbank.db`
2. Build KG (if needed):
   - `python -m app.knowledge_graph.kg_builder_full`
   - Produces `knowledge_graph.pkl` and `kg_embeddings.pkl`
3. Keep DDInter CSV files in `backend/app/data/` as `ddinter_code_*.csv`
4. Run training stages

## Latest Training Results

Training command:

```bash
python -m app.models.trainer --stage all
```

Results:

- Stage 1 (NER) best macro-F1: **0.9517**
- Stage 2 (interaction) best macro-F1: **0.4190**
- Stage 3 (severity) best macro-F1: **0.3602**
- Stage 3 DDInter coverage: **795 / 3411 = 23.31%**
- Stage 3 skipped class mentions: **539**
- Linking snapshot run id: `854063fe93f2b781`

Note on focal loss:

- A focal-loss test was done earlier outside the current committed trainer path.
- The current repository keeps CE for Stage 2 and that is the canonical training path.

## How To Run

From `backend/`:

```bash
.\venv\Scripts\Activate.ps1
python -m app.data.drugbank_processor
python -m app.models.trainer --stage all
python main.py
```

Optional stage-by-stage training:

```bash
python -m app.models.trainer --stage 1
python -m app.models.trainer --stage 2
python -m app.models.trainer --stage 3
```

## API Endpoints

Base prefix: `/api`

- `POST /api/analyze`
  - Request body:
    - `drug_a` (string, required)
    - `drug_b` (string, required)
    - `text` (string, optional)
  - Returns:
    - predicted interaction type + confidence
    - predicted severity + confidence
    - detected entities
    - KG context
    - Lipinski context
    - modality coverage flags

- `GET /api/health`
  - Quick status of model/tokenizer/KG/Lipinski loading and assistant key config

- `GET /api/drugs?limit=50`
  - Sample of drug names currently available in KG

## Known Limitations

- Stage 3 severity is trained only on pairs that can be linked to DDInter.
- Some corpus entities are drug classes, not specific ingredients.
- KG coverage is incomplete; missing drugs use zero vectors.
- `caution` class is still low-support.
- Probability calibration is not fully studied yet (no full ECE/Brier analysis in this repo).
- Automated tests are currently lightweight (more smoke/integration tests are planned).

### Data Split Note (for academic review)

- Current runs used sentence-level train/validation split while building and stabilizing the full pipeline.
- This can overestimate validation when related pairs appear across splits.
- Official corpus test evaluation should be treated as more important.
- Planned upgrade: retrain with strict pair-level or document-level split.

## Reproducibility Notes

- Linking decisions are stored in `backend/app/data/linking_snapshots.sqlite`.
- Each run has a `run_id` and method tags (`ddinter_surface`, `ddinter_from_drugbank_synonym`, etc.).

## Demo + Assistant Layer

- API app: `backend/main.py`
- Demo page: `backend/app/static/demo.html`
- Assistant routes: `backend/app/api/assistant_routes.py`
- LLM dependency: `backend/requirements-llm.txt`

Assistant setup:

- Install: `pip install -r backend/requirements-llm.txt`
- Set env var: `ANTHROPIC_API_KEY`
- Optional env var: `ANTHROPIC_MODEL` (default `claude-3-5-haiku-20241022`)
- Endpoints:
  - `GET /api/assistant/status`
  - `POST /api/assistant/chat`

## Checkpoint / Artifact Policy

- Large model/data artifacts are intentionally excluded from git.
- Required checkpoints are listed in `CHECKPOINTS.md`.
- Runtime-critical checkpoints are:
  - `stage3_severity_best.pt`
  - `stage2_interaction_best.pt`
- `stage1_ner_best.pt` is needed for stage-wise training workflow.
- Optional legacy fallback (if present): `best_model_3heads.pt`

## Safety Note

This project is for academic/research demonstration.  
It is not a clinical decision-support system and must not replace licensed medical advice.

Repository: [MedGuard](https://github.com/Eng-AlaaHosny/MedGuard)

