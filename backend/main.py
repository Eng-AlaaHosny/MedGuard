# this file starts the api server, loads models and data, and connects all routes
import os
import socket
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
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, 'app')
DATA_DIR = os.path.join(APP_DIR, 'data')
STATIC_DIR = os.path.join(APP_DIR, 'static')
MODELS_DIR = os.path.join(APP_DIR, 'models', 'checkpoints')
KG_PATH = os.path.join(DATA_DIR, 'knowledge_graph.pkl')
LIPINSKI_PATH = os.path.join(DATA_DIR, 'DB_compounds_lipinski.csv')
CHECKPOINT_CANDIDATES = [os.path.join(MODELS_DIR, 'stage3_severity_best.pt'), os.path.join(MODELS_DIR, 'best_model_3heads.pt')]
INTERACTION_CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'stage2_interaction_best.pt')
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
LISTEN_HOST = '0.0.0.0'


def pick_listen_port(host: str = LISTEN_HOST, start: int = 8000, span: int = 10) -> int:
    """Choose a TCP port for uvicorn. Honors PORT env; otherwise first free port in range."""
    env = os.environ.get('PORT')
    candidates = [int(env)] if env is not None else list(range(start, start + span))
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError as exc:
                if env is not None:
                    raise RuntimeError(
                        f'PORT={port} is already in use (another server or stale python.exe). '
                        f'Stop that process, unset PORT, or set PORT to a free port. Original error: {exc}'
                    ) from exc
                continue
        return port
    raise RuntimeError(
        f'No free TCP port on {host} in range {start}-{start + span - 1}. '
        'Stop whatever is using those ports or set PORT to a free port.'
    )

@asynccontextmanager
# this async function is used to handle lifespan
async def lifespan(app: FastAPI):
    print('=' * 60)
    print('MedGuard - starting up')
    print('=' * 60)
    print(f'BASE_DIR        : {BASE_DIR}')
    print(f'APP_DIR         : {APP_DIR}')
    print(f'DATA_DIR        : {DATA_DIR}')
    print(f'STATIC_DIR      : {STATIC_DIR}   (exists={os.path.isdir(STATIC_DIR)})')
    resolved_checkpoint = next((p for p in CHECKPOINT_CANDIDATES if os.path.exists(p)), None)
    print(f'CHECKPOINT_PATH : {resolved_checkpoint or CHECKPOINT_CANDIDATES[0]}   (exists={resolved_checkpoint is not None})')
    print(f'STAGE2_PATH     : {INTERACTION_CHECKPOINT_PATH}   (exists={os.path.exists(INTERACTION_CHECKPOINT_PATH)})')
    print(f'KG_PATH         : {KG_PATH}   (exists={os.path.exists(KG_PATH)})')
    print(f'LIPINSKI_PATH   : {LIPINSKI_PATH}   (exists={os.path.exists(LIPINSKI_PATH)})')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device          : {device}')
    print('\n[1/4] Loading Bio_ClinicalBERT tokenizer...')
    try:
        tokenizer = load_tokenizer(MODEL_NAME)
        routes_module.tokenizer = tokenizer
        print('      Tokenizer ready.')
    except Exception as exc:
        print(f'      ERROR: {exc}')
        raise
    print('\n[2/4] Loading MedGuard model...')
    try:
        model = load_model(MODEL_NAME)
        if resolved_checkpoint is not None:
            print(f'      Found checkpoint: {resolved_checkpoint}')
            state = torch.load(resolved_checkpoint, map_location=device)
            model.load_state_dict(state)
            print('      [OK] Fine-tuned weights loaded successfully.')
        else:
            print('      [WARN] No checkpoint found - using pretrained weights only.')
        model.to(device)
        model.eval()
        routes_module.model = model
        print('      Model ready.')
    except Exception as exc:
        print(f'      ERROR: {exc}')
        raise
    print('\n[2b/4] Loading dedicated interaction model...')
    try:
        if os.path.exists(INTERACTION_CHECKPOINT_PATH):
            interaction_model = load_model(MODEL_NAME)
            interaction_state = torch.load(INTERACTION_CHECKPOINT_PATH, map_location=device)
            interaction_model.load_state_dict(interaction_state)
            interaction_model.to(device)
            interaction_model.eval()
            routes_module.interaction_model = interaction_model
            print(f'      [OK] Stage 2 interaction model loaded: {INTERACTION_CHECKPOINT_PATH}')
        else:
            routes_module.interaction_model = None
            print('      [WARN] Stage 2 checkpoint not found - using main model for interaction.')
    except Exception as exc:
        print(f'      WARNING: interaction model load failed ({exc}). Falling back to main model.')
        routes_module.interaction_model = None
    print('\n[3/4] Loading Knowledge Graph...')
    try:
        kg = DrugKnowledgeGraph()
        if os.path.exists(KG_PATH):
            kg.load(KG_PATH)
            print(f'      [OK] Full KG loaded - {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges.')
        else:
            print('      [WARN] knowledge_graph.pkl not found - using 10-node demo graph.')
            kg = build_demo_graph()
        routes_module.kg = kg
    except Exception as exc:
        print(f'      WARNING: KG load failed ({exc}) - continuing without KG.')
        routes_module.kg = None
    print('\n[4/4] Loading Lipinski physicochemical data...')
    try:
        if os.path.exists(LIPINSKI_PATH):
            lip = LipinskiProcessor()
            lip.load(LIPINSKI_PATH)
            routes_module.lipinski = lip
            print(f'      [OK] Lipinski ready - {len(lip.drug_id_to_features)} compounds.')
        else:
            print('      [WARN] DB_compounds_lipinski.csv not found.')
            routes_module.lipinski = None
    except Exception as exc:
        print(f'      WARNING: Lipinski load failed ({exc}).')
        routes_module.lipinski = None
    listen_port = int(os.environ.get('MEDGUARD_LISTEN_PORT', '8000'))
    print('\n' + '=' * 60)
    print('MedGuard startup complete - API is ready.')
    print(f'http://127.0.0.1:{listen_port}')
    if listen_port != 8000:
        print(
            f'Note: serving on port {listen_port} — open the URL above for the demo. '
            f'If you use demo.html from disk, add ?api=http://127.0.0.1:{listen_port} or serve this app from /.'
        )
    print('=' * 60 + '\n')
    yield
    print('MedGuard shutting down.')
app = FastAPI(title='MedGuard DDI Detection API', description='Drug-Drug Interaction detection using Bio_ClinicalBERT with multi-task learning (NER + DDI classification + Severity prediction) enriched by a DrugBank Knowledge Graph.', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(routes_module.router, prefix='/api')
app.include_router(assistant_router, prefix='/api')
if os.path.isdir(STATIC_DIR):
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

    @app.get('/', include_in_schema=False)
    # this function is used to handle serve demo
    def serve_demo():
        demo_path = os.path.join(STATIC_DIR, 'demo.html')
        if os.path.exists(demo_path):
            return FileResponse(demo_path)
        return {'message': f'demo.html not found in {STATIC_DIR}'}
else:

    @app.get('/', include_in_schema=False)
    # this function is used to handle root
    def root():
        return {'message': f'static/ not found - expected: {STATIC_DIR}', 'docs': '/docs', 'health': '/api/health'}
if __name__ == '__main__':
    import uvicorn
    port = pick_listen_port()
    os.environ['MEDGUARD_LISTEN_PORT'] = str(port)
    uvicorn.run('main:app', host=LISTEN_HOST, port=port, reload=False)
