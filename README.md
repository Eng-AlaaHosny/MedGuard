# MedGuard

**Drug–drug interaction (DDI) detection** with a multi-task clinical NLP model, DrugBank knowledge-graph enrichment, Lipinski physicochemical features, and a FastAPI demo with an optional Claude assistant.

| | |
|---|---|
| **Course** | SE 4003 — Multidisciplinary Engineering Projects |
| **Institution** | Muğla Sıtkı Koçman University — Software Engineering |
| **Semester** | 2025–2026 Spring |
| **Author** | Alaa Hosny Saber Hassouba (220717702) |

> **Safety — read first**  
> MedGuard is for **academic demonstration only**. It must **not** be used for clinical decision-making. All outputs require verification by a licensed healthcare professional.

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Training from scratch](#training-from-scratch)
- [Three-stage curriculum](#three-stage-curriculum)
- [Entity linking](#entity-linking)
- [Training vs inference](#training-vs-inference)
- [Project structure](#project-structure)
- [Data and artefacts](#data-and-artefacts)
- [Known limitations](#known-limitations)
- [Documentation](#documentation)
- [References](#references)
- [Author](#author)

---

## Overview

MedGuard accepts two drug names (and optional clinical context text) and predicts:

| Head | Output | Labels |
|------|--------|--------|
| **NER** | Drug mentions in text | `O`, `B-DRUG`, `I-DRUG` |
| **Interaction type** | DDI linguistic category | `false`, `mechanism`, `effect`, `advise`, `int` |
| **Severity** | Clinical risk level | `safe`, `caution`, `warning`, `danger` |

Each drug is represented by fusing three signals:

- **Clinical text** — Bio_ClinicalBERT span pooling (768-d)
- **Knowledge graph** — DrugBank subgraph + node2vec embeddings (128-d)
- **Chemistry** — Lipinski descriptors: MW, HBA, HBD, logP, Ro5 (5-d)

Concatenated per drug → **901-d** → `drug_fusion` MLP → 768-d.  
Pair vector → concat(drug_a, drug_b, CLS) → **2304-d** → `pair_projection` → classification heads.

Training follows a **three-stage curriculum** (NER → interaction → severity) so easier tasks do not dominate gradients. At runtime, **two forward passes** use separate best checkpoints for interaction vs NER/severity.

**Deliverables in this repo:** trained-model inference API, browser demo (`demo.html`), offline training pipeline, entity-linking audit log, and optional conversational layer (Anthropic Claude with tool-calling into the same inference function).

---

## Results

All metrics are **macro-F1** on a **15% validation split** (single training run; no test-set tuning).

| Stage | Task | Macro-F1 | Notes |
|-------|------|----------|-------|
| 1 | NER (B-DRUG + I-DRUG only) | **0.9517** | Class `O` excluded from metric |
| 2 | Interaction type (5-class) | **0.4190** | DDI Corpus linguistic labels |
| 3 | Severity (4-class) | **0.3602** | Limited by DDInter name coverage |

**Stage 3 labelling:** **795 / 3,411** sentence pairs (23.31%) received a resolved DDInter severity label; **539** mentions were skipped as drug-class phrases (`class_routing`). The low severity F1 is primarily a **data-linking constraint**, not a modelling failure in isolation.

**Checkpoints** (not in git): [Google Drive folder](https://drive.google.com/drive/folders/1qBovw44ooOrlT1yP_onUIVjUtQX2CEAq?usp=sharing) — see [CHECKPOINTS.md](CHECKPOINTS.md).

---

## Architecture

```text
OFFLINE                          ONLINE
───────                          ──────
DDI Corpus ──► preprocessor.py
DrugBank XML ─► drugbank_processor.py ──► drugbank.db
              ─► kg_builder_full.py ──► knowledge_graph.pkl
DDInter CSVs ─► ddinter_vocabulary.py
Lipinski CSV ─► (shipped in repo)
              ─► trainer.py ──► stage{1,2,3}_*.pt
                                        │
                                        ▼
                              main.py (FastAPI)
                              ├── POST /api/analyze  ◄── demo.html
                              └── POST /api/assistant/chat (optional)
```

| Component | Details |
|-----------|---------|
| **Encoder** | [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) (768-d hidden) |
| **KG** | NetworkX graph from DrugBank interactions; node2vec 128-d (`KG_DIM`) |
| **Max tokens** | 128 (`MAX_LENGTH`) |
| **Runtime checkpoints** | `stage3_severity_best.pt` (NER + severity); `stage2_interaction_best.pt` (interaction head) |


---

## Quick start

Run the **inference demo** only — no training required if you download checkpoints.

### Requirements

- **Python 3.11** recommended (see `backend/requirements.txt`)
- **~2 GB** disk for checkpoints + dependencies
- **GPU** optional (CPU inference works, slower)

### 1. Clone and install

```powershell
git clone https://github.com/Eng-AlaaHosny/MedGuard.git
cd MedGuard/backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux / macOS:

```bash
git clone https://github.com/Eng-AlaaHosny/MedGuard.git
cd MedGuard/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



### 2. Download checkpoints

Place these files in `backend/app/models/checkpoints/`:

| File | Runtime role |
|------|----------------|
| `stage2_interaction_best.pt` | Interaction-type head |
| `stage3_severity_best.pt` | NER + severity heads |

`stage1_ner_best.pt` is used only inside the training chain — **not** loaded by the API.

Download: [Google Drive](https://drive.google.com/drive/folders/1qBovw44ooOrlT1yP_onUIVjUtQX2CEAq?usp=sharing) · [CHECKPOINTS.md](CHECKPOINTS.md)

### 3. Start the server

```bash
cd backend
python main.py
```

- Default URL: `http://127.0.0.1:8000` (scans ports **8000–8009** if busy; override with `PORT`)
- **Demo UI:** `/` → `demo.html`
- **Swagger:** `/docs`
- **Health:** `GET /api/health`

**Notes**

- If `knowledge_graph.pkl` is missing, a small **demo graph** is built automatically; predictions are still meaningful with checkpoints + Lipinski CSV.
- Lipinski CSV (`DB_compounds_lipinski.csv`) is included in the repo.
- Without checkpoints, the server starts but inference will fail until weights are present.



## API reference

**Request body**

| Field | Required | Description |
|-------|----------|-------------|
| `drug_a` | Yes | First drug name |
| `drug_b` | Yes | Second drug name |
| `text` | No | Clinical context; if empty, a synthetic sentence is built |

**Response highlights:** `interaction_type`, `severity_label`, `severity_color`, `confidence`, `detected_entities`, `kg_context`, `lipinski_context`, `modality_coverage`, `plain_guidance`, `interaction_reason`.

**Inference flow**

1. Tokenise text (max 128 tokens); locate drug spans via offset mapping.
2. Look up KG embeddings (128-d) and Lipinski features (5-d); missing → zeros + `modality_coverage` flags.
3. Forward pass on **stage3** model → NER + severity logits.
4. Forward pass on **stage2** model (if present) → interaction logits; else interaction from stage3.
5. Softmax → labels; attach DrugBank edge text as `kg_context` when a direct KG edge exists.

### `POST /api/assistant/chat`

Claude (`claude-3-5-haiku-20241022` by default) calls `medguard_analyze_pair`, which invokes the same `run_pair_inference()` as `/api/analyze`. Responses include a `tool_trace` for transparency. Requires `ANTHROPIC_API_KEY`.

---

## Training from scratch

All commands run from **`backend/`**. Large raw datasets are **not** included in git.

### Prerequisites

| Asset | Expected path | Purpose |
|-------|---------------|---------|
| DDI Corpus XML | `app/data/DDICorpus/` | Stages 1–3 sentences |
| DrugBank 5.0 XML | `app/data/drugbank_full.xml/full database.xml` | `drugbank.db` + KG seed |
| DDInter CSVs | `app/data/ddinter_code_*.csv` | Stage 3 severity labels (partial set in repo) |
| Lipinski CSV | `app/data/DB_compounds_lipinski.csv` | Shipped in repo |

### Build and train

```bash
cd backend

# 1. DrugBank → SQLite
python -m app.data.drugbank_processor

# 2. Knowledge graph + node2vec (recommended before Stages 2–3)
python -m app.knowledge_graph.kg_builder_full

# 3. Three-stage curriculum
python -m app.models.trainer --stage all
```

**Outputs**

- `app/models/checkpoints/stage1_ner_best.pt`
- `app/models/checkpoints/stage2_interaction_best.pt`
- `app/models/checkpoints/stage3_severity_best.pt`
- `app/data/linking_snapshots.sqlite` (entity-linker audit log)

**CLI options:** `--stage {1,2,3,all}`, `--epochs1`, `--epochs2`, `--epochs3`, `--model` (default: Bio_ClinicalBERT).

### Default hyperparameters

| Setting | Stage 1 (NER) | Stage 2 (Interaction) | Stage 3 (Severity) |
|---------|---------------|------------------------|---------------------|
| Epochs | 5 | 10 | 10 |
| Batch size | 16 | 4 | 4 |
| Learning rate | 2e-5 | 1e-4 | 1e-4 |
| Gradient accumulation | — | 4 | 4 |
| Validation split | 0.15 | 0.15 | 0.15 |
| Max length | 128 | 128 | 128 |
| Loss | Weighted CE | Weighted CE | Balanced Softmax |
| Validation metric | Macro-F1 (B+I) | Macro-F1 | Macro-F1 |

**KG build (node2vec):** 200 walks × length 30, window 10, **128-d** output.

---

## Three-stage curriculum

### Stage 1 — NER

- Trains Bio_ClinicalBERT + NER head on DDI Corpus token labels.
- Weighted cross-entropy downweights class `O`.
- Checkpoint: `stage1_ner_best.pt`.

### Stage 2 — Interaction type

- Loads Stage 1 weights; **freezes** encoder + NER head.
- Trains `drug_fusion`, `pair_projection`, `interaction_head`.
- Labels from DDI Corpus interaction annotations.
- Checkpoint: `stage2_interaction_best.pt`.

### Stage 3 — Severity

- Loads Stage 2 weights; **freezes** interaction head.
- Severity labels from DDInter via `DictionaryFirstLinker`.
- Balanced Softmax (logit adjustment) for sparse `caution` class.
- Checkpoint: `stage3_severity_best.pt`.

### DDInter level → severity label

| DDInter `level` | Model label |
|-----------------|-------------|
| *(no match / safe negative)* | `safe` |
| `minor` | `caution` |
| `moderate` | `warning` |
| `major`, `contraindicated` | `danger` |

### Runtime checkpoint roles

| File | Loaded by API | Provides |
|------|---------------|----------|
| `stage3_severity_best.pt` | Yes (required) | NER + severity |
| `stage2_interaction_best.pt` | Yes (recommended) | Interaction type |
| `stage1_ner_best.pt` | No | Training chain only |
| `best_model_3heads.pt` | Legacy fallback | All heads if stage files absent |

---

## Entity linking

`DictionaryFirstLinker` (`entity_linker.py`) resolves DDI Corpus drug strings to DDInter names through a **deterministic six-step chain** (first match wins):

| Step | Strategy | Effect |
|------|----------|--------|
| 1 | `class_routing` | Drug class phrases (e.g. “ACE inhibitor”) — **excluded** from severity training |
| 2 | `ddinter_surface` | Normalised name in `ddinter_drug_names` |
| 3 | DrugBank synonym | Synonym → DrugBank ID → best DDInter match |
| 4 | KG maps | `knowledge_graph.pkl` name maps → DDInter |
| 5 | `drugbank_only` | In DrugBank but not DDInter — **excluded** from Stage 3 |
| 6 | `passthrough` | Normalised raw string (lookup often misses) |

Resolutions are logged to `linking_snapshots.sqlite` with a unique `run_id` (reference run: `854063fe93f2b781`).

Text normalisation for all KB lookups: `normalize_kb_text()` in `kb_normalization.py` (lowercase, strip parentheses/punctuation, remove salt forms).

---

## Training vs inference

| Phase | How drug text is pooled |
|-------|-------------------------|
| Stages 2–3 **training** | **[CLS] token** (entity spans not passed to `forward`) |
| **Inference** | **Token span mean-pooling** on `drug_a` / `drug_b` via `find_token_span()` |

During training, sentence context plus KG/Lipinski side features carry drug identity. At inference, users supply explicit drug names, so span pooling is used. See [Known limitations](#known-limitations).

---

## Project structure

```text
MedGuard/
├── README.md                     # This file
├── CHECKPOINTS.md                # Google Drive download links
├── PRESENTATION_GUIDE.md         # Demo script and examiner Q&A
├── ARCHITECTURE_GRAPHS_CN.md     # Mermaid architecture diagrams
└── backend/
    ├── main.py                   # FastAPI entry (lifespan loads models + KG)
    ├── requirements.txt          # Core: torch, transformers, fastapi, …
    ├── requirements-llm.txt      # Optional: anthropic SDK
    └── app/
        ├── api/
        │   ├── routes.py             # /api/analyze, /health, /drugs
        │   └── assistant_routes.py     # /api/assistant/chat
        ├── models/
        │   ├── medguard_model.py       # MedGuardModel architecture
        │   ├── trainer.py              # Three-stage training CLI
        │   └── checkpoints/            # .pt weights (gitignored)
        ├── data/
        │   ├── preprocessor.py
        │   ├── drugbank_processor.py
        │   ├── entity_linker.py
        │   ├── ddinter_vocabulary.py
        │   ├── kb_normalization.py
        │   ├── lipinski_processor.py
        │   ├── DB_compounds_lipinski.csv
        │   └── ddinter_code_*.csv
        ├── knowledge_graph/
        │   ├── graph_builder.py        # DrugKnowledgeGraph
        │   └── kg_builder_full.py      # Builds knowledge_graph.pkl
        └── static/
            └── demo.html               # Browser UI
```

---

## Data and artefacts

| File | Role |
|------|------|
| `drugbank.db` | Drugs, synonyms, interactions; DDInter vocab |
| `knowledge_graph.pkl` | NetworkX graph + node2vec embeddings + name maps |
| `DB_compounds_lipinski.csv` | Five physicochemical features per compound |
| `linking_snapshots.sqlite` | Entity-linker resolution audit trail |

**Primary external sources:** DDI Corpus (SemEval-2013 Task 9), DrugBank 5.0, DDInter, Lipinski Rule of Five descriptors.

---

## Known limitations

- **DDInter coverage (~23%)** — only 795 / 3,411 pairs carry severity labels; constrains Stage 3 F1.
- **Drug-class mentions** — 539 `class_routing` exclusions; no class→ingredient expansion yet.
- **Train/inference span mismatch** — CLS pooling in Stages 2–3 vs span pooling at inference.
- **Sentence-level validation split** — may be optimistic; pair- or document-level split would be stricter.
- **No probability calibration** — confidences are raw softmax, not ECE-corrected.
- **No automated test suite** — pytest / CI not included.
- **Incomplete KG/Lipinski coverage** — zero vectors + `modality_coverage` flags in API responses.
- **No modality ablation** — contribution of text vs KG vs Lipinski not isolated in experiments.

Future work: expand DDInter CSV coverage and fuzzy matching, span-consistent training, calibration metrics, pharmacist evaluation, full patient-facing React app.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview, setup, API, training (this file) |
| [CHECKPOINTS.md](CHECKPOINTS.md) | Checkpoint download instructions |

---

## References

- Alsentzer, E., et al. (2019). Publicly available clinical BERT embeddings. *2nd Clinical NLP Workshop*.
- Herrero-Zazo, M., et al. (2013). The DDI corpus. *Journal of Biomedical Informatics*, 46(5), 914–920.
- Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. *ACM SIGKDD*.
- Lipinski, C.A., et al. (1997). Experimental and computational approaches to solubility/permeability. *Adv. Drug Deliv. Rev.*, 23, 3–25.
- Menon, A.K., et al. (2021). Long-tail learning via logit adjustment. *ICLR*.
- Wishart, D.S., et al. (2018). DrugBank 5.0. *Nucleic Acids Research*, 46(D1), D1074–D1082.
- Xiong, G., et al. (2022). DDInter: An online drug–drug interaction database. *Nucleic Acids Research*, 50(D1), D1200–D1207.

---

## Author

**Alaa Hosny Saber Hassouba**  
Student ID: **220717702**  
Instructor: Doç.Dr. Selim Yılmaz  
Course: **SE 4003** — Multidisciplinary Engineering Projects  
Institution: Muğla Sıtkı Koçman University — Faculty of Engineering, Software Engineering

**Repository:** [github.com/Eng-AlaaHosny/MedGuard](https://github.com/Eng-AlaaHosny/MedGuard)


