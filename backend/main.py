"""
MedGuard — FastAPI entry point
==============================
Startup sequence (lifespan):
  1. Load Bio_ClinicalBERT tokenizer + MedGuardModel
  2. Load pre-built DrugKnowledgeGraph (knowledge_graph.pkl)
  3. Load LipinskiProcessor (DB_compounds_lipinski.csv)
  4. Inject all singletons into routes module

Project layout (main.py lives inside backend/):
  backend/
    main.py
    app/
      static/demo.html
      api/routes.py
      models/checkpoints/best_model_3heads.pt
      data/
        knowledge_graph.pkl
        kg_embeddings.pkl
        DB_compounds_lipinski.csv
        drugbank.db
        DDICorpus/

API prefix:  /api
Demo UI:     http://127.0.0.1:8000/
"""

import os
import sys
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api import routes as routes_module
from app.api.assistant_routes import router as assistant_router
from app.models.medguard_model import load_model, load_tokenizer
from app.knowledge_graph.graph_builder import DrugKnowledgeGraph, build_demo_graph
from app.data.lipinski_processor import LipinskiProcessor

# Ensure predictable console output on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
APP_DIR         = os.path.join(BASE_DIR, "app")
DATA_DIR        = os.path.join(APP_DIR, "data")          # ← app/data/
STATIC_DIR      = os.path.join(APP_DIR, "static")
MODELS_DIR      = os.path.join(APP_DIR, "models", "checkpoints")

KG_PATH         = os.path.join(DATA_DIR, "knowledge_graph.pkl")
LIPINSKI_PATH   = os.path.join(DATA_DIR, "DB_compounds_lipinski.csv")
CHECKPOINT_CANDIDATES = [
    os.path.join(MODELS_DIR, "stage3_severity_best.pt"),
    os.path.join(MODELS_DIR, "best_model_3heads.pt"),
]
INTERACTION_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "stage2_interaction_best.pt")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("MedGuard - starting up")
    print("=" * 60)
    print(f"BASE_DIR        : {BASE_DIR}")
    print(f"APP_DIR         : {APP_DIR}")
    print(f"DATA_DIR        : {DATA_DIR}")
    print(f"STATIC_DIR      : {STATIC_DIR}   (exists={os.path.isdir(STATIC_DIR)})")
    resolved_checkpoint = next((p for p in CHECKPOINT_CANDIDATES if os.path.exists(p)), None)
    print(f"CHECKPOINT_PATH : {resolved_checkpoint or CHECKPOINT_CANDIDATES[0]}   (exists={resolved_checkpoint is not None})")
    print(f"STAGE2_PATH     : {INTERACTION_CHECKPOINT_PATH}   (exists={os.path.exists(INTERACTION_CHECKPOINT_PATH)})")
    print(f"KG_PATH         : {KG_PATH}   (exists={os.path.exists(KG_PATH)})")
    print(f"LIPINSKI_PATH   : {LIPINSKI_PATH}   (exists={os.path.exists(LIPINSKI_PATH)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device          : {device}")

    # 1. Tokenizer
    print("\n[1/4] Loading Bio_ClinicalBERT tokenizer...")
    try:
        tokenizer = load_tokenizer(MODEL_NAME)
        routes_module.tokenizer = tokenizer
        print("      Tokenizer ready.")
    except Exception as exc:
        print(f"      ERROR: {exc}")
        raise

    # 2. Model
    print("\n[2/4] Loading MedGuard model...")
    try:
        model = load_model(MODEL_NAME)
        if resolved_checkpoint is not None:
            print(f"      Found checkpoint: {resolved_checkpoint}")
            state = torch.load(resolved_checkpoint, map_location=device)
            model.load_state_dict(state)
            print("      [OK] Fine-tuned weights loaded successfully.")
        else:
            print("      [WARN] No checkpoint found - using pretrained weights only.")
        model.to(device)
        model.eval()
        routes_module.model = model
        print("      Model ready.")
    except Exception as exc:
        print(f"      ERROR: {exc}")
        raise

    # 2b. Dedicated interaction model (Stage 2 checkpoint) to preserve
    # interaction behavior after Stage 3 fine-tuning.
    print("\n[2b/4] Loading dedicated interaction model...")
    try:
        if os.path.exists(INTERACTION_CHECKPOINT_PATH):
            interaction_model = load_model(MODEL_NAME)
            interaction_state = torch.load(INTERACTION_CHECKPOINT_PATH, map_location=device)
            interaction_model.load_state_dict(interaction_state)
            interaction_model.to(device)
            interaction_model.eval()
            routes_module.interaction_model = interaction_model
            print(f"      [OK] Stage 2 interaction model loaded: {INTERACTION_CHECKPOINT_PATH}")
        else:
            routes_module.interaction_model = None
            print("      [WARN] Stage 2 checkpoint not found - using main model for interaction.")
    except Exception as exc:
        print(f"      WARNING: interaction model load failed ({exc}). Falling back to main model.")
        routes_module.interaction_model = None

    # 3. Knowledge Graph
    print("\n[3/4] Loading Knowledge Graph...")
    try:
        kg = DrugKnowledgeGraph()
        if os.path.exists(KG_PATH):
            kg.load(KG_PATH)
            print(f"      [OK] Full KG loaded - {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges.")
        else:
            print("      [WARN] knowledge_graph.pkl not found - using 10-node demo graph.")
            kg = build_demo_graph()
        routes_module.kg = kg
    except Exception as exc:
        print(f"      WARNING: KG load failed ({exc}) - continuing without KG.")
        routes_module.kg = None

    # 4. Lipinski
    print("\n[4/4] Loading Lipinski physicochemical data...")
    try:
        if os.path.exists(LIPINSKI_PATH):
            lip = LipinskiProcessor()
            lip.load(LIPINSKI_PATH)
            routes_module.lipinski = lip
            print(f"      [OK] Lipinski ready - {len(lip.drug_id_to_features)} compounds.")
        else:
            print("      [WARN] DB_compounds_lipinski.csv not found.")
            routes_module.lipinski = None
    except Exception as exc:
        print(f"      WARNING: Lipinski load failed ({exc}).")
        routes_module.lipinski = None

    print("\n" + "=" * 60)
    print("MedGuard startup complete - API is ready.")
    # Bare URL on its own line — some terminals only linkify URLs when the line is mostly the URL.
    print("http://127.0.0.1:8000")
    print("=" * 60 + "\n")

    yield

    print("MedGuard shutting down.")


app = FastAPI(
    title="MedGuard DDI Detection API",
    description=(
        "Drug-Drug Interaction detection using Bio_ClinicalBERT with "
        "multi-task learning (NER + DDI classification + Severity prediction) "
        "enriched by a DrugBank Knowledge Graph."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_module.router, prefix="/api")
app.include_router(assistant_router, prefix="/api")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_demo():
        demo_path = os.path.join(STATIC_DIR, "demo.html")
        if os.path.exists(demo_path):
            return FileResponse(demo_path)
        return {"message": f"demo.html not found in {STATIC_DIR}"}
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {
            "message": f"static/ not found - expected: {STATIC_DIR}",
            "docs": "/docs",
            "health": "/api/health",
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)