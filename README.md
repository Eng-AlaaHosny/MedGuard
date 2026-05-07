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

## Model Architecture 

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

## Artifact Availability (Important)

The repository intentionally excludes large runtime/training artifacts.  
If these files are missing locally, training/inference will run with fallbacks or fail where the artifact is required.

- Usually external or generated artifacts:
  - `backend/app/models/checkpoints/*.pt`
  - `backend/app/data/drugbank.db`
  - `backend/app/data/knowledge_graph.pkl`
  - `backend/app/data/kg_embeddings.pkl`
  - `backend/app/data/linking_snapshots.sqlite` (created after Stage 3/linking runs)
- Committed in this repo:
  - `backend/app/data/DDICorpus/**`
  - `backend/app/data/ddinter_code_*.csv`
  - `backend/app/data/DB_compounds_lipinski.csv`

## Training Defaults ( current status)

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

## Runtime Checkpoint Policy (current)

From `backend/main.py`:

- Main model checkpoints (priority):
  1. `backend/app/models/checkpoints/stage3_severity_best.pt`
  2. `backend/app/models/checkpoints/best_model_3heads.pt` (legacy fallback)
- Interaction model:
  - `backend/app/models/checkpoints/stage2_interaction_best.pt`
  - if missing, interaction falls back to main model
- `stage1_ner_best.pt` is training-only (used to initialize Stage 2 training).

## Inference Behavior 

- Main endpoint takes explicit drug names: `drug_a`, `drug_b`.
- Optional text can be provided (`text` field).
- If text is empty, backend creates a short template sentence for inference.
- No training-corpus sentence substitution is used at inference.

## Data Used

- DDI Corpus (XML)
  - What it is: the main NLP dataset with sentences, drug entities, and DDI relation labels.
  - Used for: Stage 1 NER training and Stage 2 interaction-type training.

- DrugBank XML (`full database.xml`)
  - What it is: structured drug knowledge source (drugs, descriptions, interactions, synonyms).
  - Used for: building local `drugbank.db` and creating KG/linking resources.

- DrugBank SQLite (`drugbank.db`)
  - What it is: local processed DB built from DrugBank XML.
  - Used for: fast lookup of synonyms/IDs/interactions during linking and inference context building.

- DDInter CSV files (`ddinter_code_*.csv`)
  - What it is: curated interaction risk-level tables from DDInter.
  - Used for: Stage 3 severity labels after name linking to DDInter vocabulary.

- KG file (`knowledge_graph.pkl`)
  - What it is: prebuilt drug knowledge graph + node embeddings mapping.
  - Used for: KG feature vectors and known-interaction context at inference.

- Lipinski file (`DB_compounds_lipinski.csv`)
  - What it is: physicochemical descriptors per DrugBank compound (MW, HBA, HBD, logP, Ro5).
  - Used for: extra numerical features fused with BERT drug representations.

## Data / Build Pipeline

1. Put DrugBank XML at:
   - `backend/app/data/drugbank_full.xml/full database.xml`
2. Parse DrugBank XML and build SQLite DB:
   - `python -m app.data.drugbank_processor`
   - Produces `backend/app/data/drugbank.db`
3. Build KG (if needed):
   - `python -m app.knowledge_graph.kg_builder_full`
   - Produces `knowledge_graph.pkl` and `kg_embeddings.pkl`
4. Keep DDInter CSV files in `backend/app/data/` as `ddinter_code_*.csv`
5. Run training stages

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

Note:
- These values are from the referenced training run and are not auto-updated from code.
- Re-running training can produce different numbers depending on data/artifact state and environment.

checkpoints files : https://drive.google.com/drive/folders/1qBovw44ooOrlT1yP_onUIVjUtQX2CEAq?usp=sharing

Note on focal loss:

- A focal-loss test was done earlier outside the current committed trainer path.
- The current repository keeps CE for Stage 2 and that is the canonical training path.

## How To Run

From `backend/`:

```bash
.\.venv\Scripts\Activate.ps1
python -m app.data.drugbank_processor
python -m app.models.trainer --stage all
python main.py
```

Before running, confirm required data/artifacts are present (or generate/download them via the pipeline above).

Optional stage-by-stage training:

```bash
python -m app.models.trainer --stage 1
python -m app.models.trainer --stage 2
python -m app.models.trainer --stage 3
```

### Run Demo (Web UI)

1. Start the API:

```bash
python main.py
```

2. Open in browser:
   - `http://127.0.0.1:8000`

3. Use the page:
   - Add 2+ drugs and click **Analyze Interactions**
   - Optional: use assistant panel after setting `ANTHROPIC_API_KEY`

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

## Known Limitations (currently)

- Stage 3 severity is trained only on pairs that can be linked to DDInter.
  - What we did: built a deterministic 3-layer linker (DrugBank synonyms + DDInter vocabulary + passthrough), logged run-level linking snapshots, and reported coverage in training output.
  - Why this limitation remains: DDInter does not cover all DDI Corpus pairs, and some mentions are too ambiguous to map reliably without adding noisy labels.

- Some corpus entities are drug classes, not specific ingredients.
  - What we did: added explicit class routing (`class_routing`) so these cases are identified and not force-mapped to wrong ingredient names.
  - Why this limitation remains: many KBs are ingredient-focused, so class-level mentions are not always directly linkable to pair labels.

- KG coverage is incomplete; missing drugs use zero vectors.
  - What we did: designed graceful fallback to zero KG/Lipinski vectors and exposed modality coverage flags in API output.
  - Why this limitation remains: the KG is a practical subgraph for this project scope, not full DrugBank graph coverage at all times.

- `caution` class is still low-support.
  - What we did: used class-aware losses (weighted CE in Stage 2, Balanced Softmax in Stage 3) and reported class distributions.
  - Why this limitation remains: real label distribution is long-tail and caution examples are limited.

- Probability calibration is not fully studied yet (no full ECE/Brier analysis in this repo).
  - What we did: return full class probability vectors for transparency instead of only top labels.
  - Why this limitation remains: calibration study was out of current project time scope.

- Automated tests are currently lightweight.
  - What we did: focused on pipeline correctness, endpoint health checks, and reproducible training/inference behavior first.
  - Why this limitation remains: comprehensive unit/integration test suite is planned as the next engineering hardening step.

### Data Split Note 

- Current runs used sentence-level train/validation split while building and stabilizing the full pipeline.
- This can overestimate validation when related pairs appear across splits.
- Official corpus test evaluation should be treated as more important.
- Why we moved forward with it: the priority was to first complete and validate the full end-to-end system (3-stage training, linker, KG/Lipinski fusion, API, assistant layer) with a consistent protocol.
- Planned upgrade: retrain with strict pair-level or document-level split as the next methodological improvement.

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
- Optional env var: `ANTHROPIC_MODEL` (if not set, backend uses the default configured in `backend/app/api/assistant_routes.py`; check that file for the latest value)
- Endpoints:
  - `GET /api/assistant/status`
  - `POST /api/assistant/chat`

## Checkpoint 
https://drive.google.com/drive/folders/1qBovw44ooOrlT1yP_onUIVjUtQX2CEAq?usp=sharing

- Large model/data artifacts are intentionally excluded from git.
- Required checkpoints are listed in `CHECKPOINTS.md`.
- Runtime-critical checkpoints are:
  - `stage3_severity_best.pt`
  - `stage2_interaction_best.pt`
- `stage1_ner_best.pt` is needed for stage-wise training workflow.
- Optional legacy fallback (if present): `best_model_3heads.pt`

## Safety Note

This project is for academic/research demonstration ONLYYY
It is NOT a clinical decision-support system and must not replace licensed medical advice.

## Project Structure 

```text
MedGuard-clean/
├─ README.md
├─ CHECKPOINTS.md
└─ backend/
   ├─ main.py
   ├─ requirements-llm.txt
   └─ app/
      ├─ api/
      │  ├─ routes.py
      │  └─ assistant_routes.py
      ├─ models/
      │  ├─ medguard_model.py
      │  ├─ trainer.py
      │  └─ checkpoints/
      ├─ data/
      │  ├─ preprocessor.py
      │  ├─ drugbank_processor.py
      │  ├─ entity_linker.py
      │  ├─ lipinski_processor.py
      │  └─ ddinter_code_*.csv
      ├─ knowledge_graph/
      │  ├─ graph_builder.py
      │  └─ kg_builder_full.py
      └─ static/
         └─ demo.html
```

Repository: [MedGuard](https://github.com/Eng-AlaaHosny/MedGuard)

