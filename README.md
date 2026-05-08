## MedGuard Project Report

This file explains what I built, how I trained it, and what the current results are.

## Quick Summary

- The model is trained in 3 stages: NER, interaction type, then severity.
- Stage 3 uses **DDInter curated severity labels** (not DrugBank keyword labels).
- Entity linking uses a deterministic 3-layer resolver.
- Stage 3 uses **Balanced Softmax** to handle class imbalance.
- At inference, **severity** comes from the **main** model (Stage 3 weights when loaded); **interaction type** uses the **Stage 2** checkpoint when `stage2_interaction_best.pt` is present, otherwise the main model handles both heads (`routes.py`).

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


## Training Defaults (current)

CLI overrides: `python -m app.models.trainer` accepts `--epochs1`, `--epochs2`, `--epochs3`, and `--model` (default backbone name matches `medguard_model.py`).

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

## Runtime Checkpoint Policy 

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
- If text is empty, backend builds a short template sentence (`routes.run_pair_inference`).
- No training-corpus sentence substitution is used at inference.
- **Which checkpoint does what** (`routes.py`): **interaction type** logits come from `interaction_model` when `stage2_interaction_best.pt` loaded; otherwise the main model is used for both heads. **Severity** logits always come from the **main** model (Stage 3 checkpoint when present).
- NER entities are decoded from the main model’s `ner_logits` on the inference text.

## Data Used

- DDI Corpus (XML)
  - What it is: the main NLP dataset with sentences, drug entities, and DDI relation labels.
  - Used for: Stage 1 NER and Stage 2 interaction-type training; Stage 3 uses the same corpus sentences with DDInter-linked severity labels.

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
  - What it is: serialized graph, DrugBank-derived interaction edges, **node2vec embeddings**, and name/id maps (`DrugKnowledgeGraph.save` format).
  - Used for: KG feature vectors during training (via `trainer.load_kg_embeddings`) and known-interaction context at inference (`routes.build_kg_context`).

- Lipinski file (`DB_compounds_lipinski.csv`)
  - What it is: physicochemical descriptors per DrugBank compound (MW, HBA, HBD, logP, Ro5).
  - Used for: extra numerical features fused with BERT drug representations.

## Data / Build Pipeline

1. Obtain **DDI Corpus** XML and place under `backend/app/data/DDICorpus/` with the folder layout expected by `preprocessor.load_ddi_corpus` (`Train/`, `Test/Test for DDI Extraction task/`).
2. Put DrugBank XML at:
   - `backend/app/data/drugbank_full.xml/full database.xml`
3. Parse DrugBank XML and build SQLite DB:
   - `python -m app.data.drugbank_processor`
   - Produces `backend/app/data/drugbank.db`
4. Build KG (recommended before stages 2–3):

```bash
python -m app.knowledge_graph.kg_builder_full
```

Produces **`backend/app/data/knowledge_graph.pkl`** (graph + node2vec embeddings + maps) and **`backend/app/data/kg_embeddings.pkl`** (standalone embedding dict — auxiliary; trainer uses `knowledge_graph.pkl`).
5. Keep DDInter CSV files in `backend/app/data/` (`ddinter_code_*.csv`; this repo includes `A,B,D,H,L,P,R,V`).
6. Run training stages. Stages 2–3 load KG vectors via **`knowledge_graph.pkl`** (`trainer.py`: `kg_path = …/knowledge_graph.pkl` → `load_kg_embeddings`). If that file is missing, training continues with **zero** KG vectors (see trainer warnings).

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

Note on focal loss:

- A focal-loss test was done earlier outside the current committed trainer path.
- The current repository keeps CE for Stage 2 and that is the canonical training path.

## How To Run



### Windows setup (quick)

In **PowerShell**, from the repository root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```



For a **demo without retraining**, place checkpoints from [Checkpoints](#checkpoints-google-drive), keep `DB_compounds_lipinski.csv`, and run `python main.py` — KG will fall back to the built-in demo graph if `knowledge_graph.pkl` is missing.

Retraining from raw corpora needs extra data not shipped with `git clone` 

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

   - Add 2+ drugs and click **Analyze Interactions**
   - Optional: use assistant panel after setting `ANTHROPIC_API_KEY`

## API Endpoints

Mounted in `backend/main.py`: routers use prefix **`/api`**. FastAPI docs: **`/docs`** when the server is running.

- **`POST /api/analyze`** — defined in `app/api/routes.py` (`DDIResponse`)
  - Body (`DDIRequest`): `drug_a`, `drug_b` (required); `text` (optional).
  - Response highlights: `inference_text`, `detected_entities` (NER spans), `interaction_type` / `interaction_type_idx` / `interaction_source`, `severity_label` / `severity_level` / `severity_color` / `severity_source`, `interaction_reason`, `evidence_text` / `evidence_source`, `plain_guidance`, `kg_context`, `lipinski_context`, `confidence` (per-class probs for interaction + severity), `modality_coverage` (whether KG/Lipinski/token spans were available per drug).

- **`GET /api/health`** — loading flags for main model, Stage 2 interaction model, tokenizer, KG, Lipinski, and whether `ANTHROPIC_API_KEY` is set.

- **`GET /api/drugs?limit=50`** — sample drug names from the loaded KG (`limit` optional).

Assistant endpoints (optional): **`GET /api/assistant/status`**, **`POST /api/assistant/chat`** — see [Demo + Assistant Layer](#demo--assistant-layer).

Static: **`/`** serves `backend/app/static/demo.html`; assets under **`/static/`** (see `main.py`).

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

- There is **no** automated test suite in this repository (`pytest`/CI-style tests are absent).
  - What we did: focused on pipeline correctness, manual health checks (`/api/health`), and reproducible training/inference behavior first.
  - Why this limitation remains: comprehensive unit/integration tests were out of scope for the current submission.

### Data Split Note 

- Current runs used sentence-level train/validation split while building and stabilizing the full pipeline.
- This can overestimate validation when related pairs appear across splits.
- Official corpus test evaluation should be treated as more important.
- Why we moved forward with it: the priority was to first complete and validate the full end-to-end system (3-stage training, linker, KG/Lipinski fusion, API, assistant layer) with a consistent protocol.
- Planned upgrade: retrain with strict pair-level or document-level split as the next methodological improvement.


## Checkpoints (Google Drive)

Download folder: [Google Drive — MedGuard checkpoints](https://drive.google.com/drive/folders/1qBovw44ooOrlT1yP_onUIVjUtQX2CEAq?usp=sharing)

Same list as `CHECKPOINTS.md`:

- **Training / runtime:** `stage1_ner_best.pt`, `stage2_interaction_best.pt`, `stage3_severity_best.pt` → place under `backend/app/models/checkpoints/` (directory exists; files are gitignored).


## Safety Note

This project is for **academic and research demonstration only**. It is **not** a clinical decision-support system and must not replace licensed medical advice.

## Project Structure 

```text
MedGuard-clean/
├─ README.md
├─ CHECKPOINTS.md
├─ .gitignore
└─ backend/
   ├─ main.py                 # FastAPI app, lifespan loads models/KG/Lipinski, port selection
   ├─ requirements.txt       # Core: torch, transformers, fastapi, pandas, etc.
   ├─ requirements-llm.txt    # Optional: anthropic SDK for assistant only
   └─ app/
      ├─ __init__.py
      ├─ api/
      │  ├─ __init__.py
      │  ├─ routes.py             # /analyze, /health, /drugs
      │  └─ assistant_routes.py   # /assistant/* (prefix included in router)
      ├─ models/
      │  ├─ __init__.py
      │  ├─ medguard_model.py     # MedGuardModel, labels, load_model/load_tokenizer
      │  ├─ trainer.py            # Stages 1–3, BalancedSoftmax severity loss
      │  └─ checkpoints/          # .pt files (ignored by git; content from Drive)
      ├─ data/
      │  ├─ preprocessor.py       # DDICorpus XML → sentences
      │  ├─ drugbank_processor.py # DrugBank XML → drugbank.db
      │  ├─ entity_linker.py      # DDInter linking + snapshots sqlite
      │  ├─ lipinski_processor.py
      │  ├─ kb_normalization.py
      │  ├─ ddinter_vocabulary.py
      │  ├─ corpus_inspector.py
      │  ├─ ddinter_code_{A,B,D,H,L,P,R,V}.csv
      │  └─ DB_compounds_lipinski.csv
      ├─ knowledge_graph/
      │  ├─ __init__.py
      │  ├─ graph_builder.py      # DrugKnowledgeGraph, demo graph
      │  └─ kg_builder_full.py      # Full KG build script
      ├─ utils/
      │  └─ __init__.py
      └─ static/
         └─ demo.html
```

Repository: [MedGuard](https://github.com/Eng-AlaaHosny/MedGuard)

