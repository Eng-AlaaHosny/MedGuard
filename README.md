## MedGuard Training Report (Current Project State)

This README documents the latest training pipeline, data linking strategy, and final training outcomes for academic review.

## Teacher Quick View

- The final system trains in 3 stages: NER, interaction type, then severity.
- Stage 3 severity training uses **DDInter curated labels**, not the broken DrugBank keyword heuristic labels.
- Entity linking was upgraded to a **deterministic 3-layer resolver** using DrugBank synonym expansion, DDInter vocabulary anchoring, and snapshot logging for reproducibility.
- Stage 1 final metric is captured: **Best NER macro-F1 = 0.9517**.
- Final full-run results reached **DDI macro-F1 = 0.4190** (Stage 2) and **Severity macro-F1 = 0.3602** (Stage 3).
- Stage 3 severity coverage improved to **23.31%** of positive DDI pairs after the final linking redesign.
- A controlled Stage 2 **Focal Loss** experiment was run and rejected because it underperformed baseline (**0.2442** vs **0.4190**), so baseline CE was kept.

## Project Goal

MedGuard predicts:

- Drug entities (NER head)
- DDI interaction type (interaction head)
- DDI severity (severity head)

Training is done in 3 stages:

1. Stage 1: NER
2. Stage 2: Interaction
3. Stage 3: Severity

---

## Why The Pipeline Was Updated

Three issues were identified and fixed:

- DrugBank heuristic severity labels were highly skewed and not usable for severity training.
- Stage 3 severity coverage collapsed when string linking failed between DDI Corpus and DDInter.
- Stage 2 KG coverage is incomplete (handled with zero vectors and reported as limitation).

---

## The current Design 

### 1) Severity Labels Source

- Stage 3 uses **DDInter curated labels** only.
- It does **not** train from DrugBank heuristic severity labels.

### 2) Deterministic Entity Linking (for Stage 3)

A 3-layer resolver is used:

1. DrugBank synonym expansion (`drugbank_synonyms` table in `drugbank.db`)
2. DDInter vocabulary anchoring (`ddinter_drug_names` table in `drugbank.db`)
3. Passthrough when unresolved

Drug-class mentions are explicitly routed (`class_routing`) and reported, not force-mapped to ingredient names.

### 3) Reproducibility / Auditability

- Every linking decision is logged to `backend/app/data/linking_snapshots.sqlite`
- Each run has a `run_id`
- Matching methods are tracked (e.g., `ddinter_surface`, `ddinter_from_drugbank_synonym`, `passthrough`)

### 4) Stage 3 Imbalance Handling

- Stage 3 uses **Balanced Softmax (logit adjustment)** with class priors from training labels.

---

## Data Assets Used

- DDI Corpus (train/test XML)
- DrugBank XML (`full database.xml`)
- DrugBank SQLite (`drugbank.db`)
- DDInter CSV files (`ddinter_code_*.csv`)
- KG embeddings (`knowledge_graph.pkl`)
- Lipinski features (`DB_compounds_lipinski.csv`)

---

## Final Full Training Results (Latest Run)

Command used:

```bash
python -m app.models.trainer --stage all
```

### Stage 1 (NER)

- Best NER macro-F1: **0.9517**
- Output checkpoint: `backend/app/models/checkpoints/stage1_ner_best.pt`
- Interpretation: Stage 1 learns token-level drug mention boundaries (B-DRUG / I-DRUG), and its checkpoint is used as the initialization source for Stage 2.

### Stage 2 (Interaction)

- Best DDI macro-F1: **0.4190**
- Output checkpoint: `backend/app/models/checkpoints/stage2_interaction_best.pt`
- Interpretation: interaction-type learning is stable across minority classes (`mechanism`, `effect`, `advise`, `int`) with a usable macro-F1 for downstream stage transfer.

Controlled rare-class experiment (teacher-facing):

- Tested one principled variant: `FocalLoss(gamma=2.0)` in Stage 2.
- Result: **Best DDI macro-F1 = 0.2442** (worse than baseline **0.4190**).
- Decision rule: keep only if it wins.
- Final decision: revert Focal Loss and keep baseline `CrossEntropyLoss(weight=ddi_w)`.

### Stage 3 (Severity)

- Coverage (DDI positives linked to DDInter): **795 / 3411 = 23.31%**
- Skipped class mentions: **539**
- Best Severity macro-F1: **0.3602**
- Best checkpoint: `backend/app/models/checkpoints/stage3_severity_best.pt`
- Linking snapshot run id: `854063fe93f2b781`
- Interpretation: after deterministic linking + Balanced Softmax, Stage 3 no longer collapses to a single class and reaches the strongest severity result obtained in this project state.

---

## Commands (Reproducible Workflow)

From `backend/`:

```bash
.\venv\Scripts\Activate.ps1
python -m app.data.drugbank_processor
python -m app.models.trainer --stage all
```

Stage-by-stage (optional):

```bash
python -m app.models.trainer --stage 1
python -m app.models.trainer --stage 2
python -m app.models.trainer --stage 3
```

Run demo API:

```bash
python main.py
```

---

## Known Limitations 

- Stage 3 severity supervision is on the linked subset only (coverage reported each run).
- Some DDI mentions are drug classes and are excluded from direct DDInter pair matching.
- KG coverage is partial; unmatched mentions use zero KG vectors by design.
- `caution` class remains low-support and difficult.

---

## 

Entity linking was performed via a deterministic three-layer resolver:  
(1) DrugBank synonym expansion,  
(2) DDInter vocabulary anchoring,  
(3) passthrough with documented coverage gaps.  
All linking decisions were logged with method provenance and run identifiers for full reproducibility.

---

## Addendum (after README freeze for teacher review — appended only)

**Repository:** [MedGuard](https://github.com/Eng-AlaaHosny/MedGuard)

This block documents **runtime / demo** updates only. All sections above are unchanged.

### Demo UI & API entry

- **`backend/main.py`** — FastAPI app; after startup prints a bare `http://127.0.0.1:8000` line for easier terminal link detection.
- **`backend/app/static/demo.html`** — Browser demo: common-medication quick grid, typed drug list + **Analyze Interactions** (same inference path as `POST /api/analyze`).



### LLM orchestration (Anthropic + MedGuard tool)

- **Install:** `pip install -r backend/requirements-llm.txt`
- **Env:** `ANTHROPIC_API_KEY` (required). Optional: `ANTHROPIC_MODEL` (default `claude-3-5-haiku-20241022`).
- **Endpoints:** `GET /api/assistant/status`, `POST /api/assistant/chat` (body: `{ "messages": [ {"role":"user"|"assistant","content":"..."} ] }`).
- **Tool:** `medguard_analyze_pair` → server runs `run_pair_inference` and returns **structured JSON**; system instructions require the model **not to contradict** tool outputs on severity / interaction type.
- **Files:** `backend/app/api/assistant_routes.py`, `backend/requirements-llm.txt`; demo includes an **LLM assistant** panel calling these routes (keys stay server-side only).

