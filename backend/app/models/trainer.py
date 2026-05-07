# this file trains stage 1, stage 2, and stage 3 models and saves checkpoints
import os
import argparse
import pickle
import sqlite3
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List, Dict, Optional, Callable, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import numpy as np
from app.models.medguard_model import MedGuardModel, load_tokenizer, STAGE_NER, STAGE_INTERACTION, STAGE_SEVERITY, DDI_LABELS, KG_DIM, LIPINSKI_DIM
from app.data.preprocessor import load_ddi_corpus, DDISentence
from app.data.entity_linker import build_stage3_resolver, looks_like_drug_class, normalize_text
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
LABEL2IDX = {'false': 0, 'mechanism': 1, 'effect': 2, 'advise': 3, 'int': 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
NER_PAD_LABEL = -100
SEVERITY_LABELS_MAP = {0: 'safe', 1: 'caution', 2: 'warning', 3: 'danger'}
DDINTER_LEVEL_TO_SEVERITY = {'minor': 1, 'moderate': 2, 'major': 3, 'contraindicated': 3}
_SALT_WORDS = {'hydrochloride', 'hcl', 'sodium', 'potassium', 'calcium', 'magnesium', 'acetate', 'phosphate', 'sulfate', 'sulphate', 'nitrate', 'chloride', 'bromide', 'iodide', 'tartrate', 'citrate', 'succinate', 'fumarate', 'gluconate', 'mesylate', 'besylate', 'tosylate', 'maleate'}
_ALIAS_MAP = {'toradol': 'ketorolac', 'ultram': 'tramadol', 'celebrex': 'celecoxib'}

# this function is used to normalize drug name
def normalize_drug_name(name: str) -> str:
    if not name:
        return ''
    s = name.strip().lower()
    if '(' in s and ')' in s:
        import re
        s = re.sub('\\([^)]*\\)', ' ', s)
    for ch in [',', '.', ';', ':', "'", '"', '/', '\\', '+', '-', '_']:
        s = s.replace(ch, ' ')
    s = ' '.join(s.split())
    parts = [p for p in s.split() if p not in _SALT_WORDS]
    s2 = ' '.join(parts).strip()
    out = s2 if s2 else s
    return _ALIAS_MAP.get(out, out)

# this function is used to load drugbank canonical map
def load_drugbank_canonical_map(db_path: str) -> Dict[str, str]:
    canon: Dict[str, str] = {}
    if not os.path.exists(db_path):
        return canon
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT name FROM drugs')
        for name, in c.fetchall():
            n = normalize_drug_name(name or '')
            if not n:
                continue
            canon.setdefault(n, n)
        conn.close()
    except Exception:
        return canon
    return canon

# this function is used to handle make name resolver
def make_name_resolver(*args, **kwargs):
    raise RuntimeError('Deprecated. Use build_stage3_resolver().')

# this function is used to load kg embeddings
def load_kg_embeddings(kg_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(kg_path):
        print(f'  ⚠  KG not found at {kg_path} — training without KG embeddings')
        return {}
    try:
        with open(kg_path, 'rb') as f:
            data = pickle.load(f)
        embeddings = data.get('embeddings', {})
        name_to_id = data.get('drug_name_to_id', {})
        name_to_emb = {name: embeddings[did] for name, did in name_to_id.items() if did in embeddings}
        print(f'  ✓  KG loaded — {len(name_to_emb)} drug embeddings')
        return name_to_emb
    except Exception as e:
        print(f'  ⚠  KG load error: {e}')
        return {}

# this function is used to load kg name maps
def load_kg_name_maps(kg_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not os.path.exists(kg_path):
        return ({}, {})
    try:
        with open(kg_path, 'rb') as f:
            data = pickle.load(f)
        name_to_id = data.get('drug_name_to_id', {}) or {}
        id_to_name = data.get('drug_id_to_name', {}) or {}
        norm_map: Dict[str, str] = {}
        for k, v in name_to_id.items():
            if not k or not v:
                continue
            raw = str(k).lower()
            norm_map[raw] = str(v)
            nk = normalize_drug_name(raw)
            if nk:
                norm_map.setdefault(nk, str(v))
        name_to_id = norm_map
        id_to_name = {str(k): str(v) for k, v in id_to_name.items() if k and v}
        return (name_to_id, id_to_name)
    except Exception:
        return ({}, {})

# this function is used to load lipinski features
def load_lipinski_features(csv_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(csv_path):
        print(f'  ⚠  Lipinski CSV not found at {csv_path} — training without Lipinski features')
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        feat_cols = ['molecular_weight', 'n_hba', 'n_hbd', 'logp', 'ro5_fulfilled']
        for col in ['molecular_weight', 'n_hba', 'n_hbd', 'logp']:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
        df['ro5_fulfilled'] = df['ro5_fulfilled'].astype(float)
        id_col = 'ID'
        result = {}
        for _, row in df.iterrows():
            drug_id = str(row[id_col])
            features = row[feat_cols].values.astype(np.float32)
            result[drug_id] = features
        print(f'  ✓  Lipinski loaded — {len(result)} drug feature vectors')
        return result
    except Exception as e:
        print(f'  ⚠  Lipinski load error: {e}')
        return {}

# this function is used to load severity from db
def load_severity_from_db(db_path: str) -> Dict:
    lookup = {}
    if not os.path.exists(db_path):
        print(f'  ⚠  drugbank.db not found — Stage 3 severity will be zero-only')
        return lookup
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('\n            SELECT LOWER(da.name), LOWER(db.name), i.severity\n            FROM interactions i\n            JOIN drugs da ON i.drug_a_id = da.id\n            JOIN drugs db ON i.drug_b_id = db.id\n        ')
        for drug_a, drug_b, severity in c.fetchall():
            lookup[drug_a, drug_b] = severity
            lookup[drug_b, drug_a] = severity
        conn.close()
        dist = Counter(lookup.values())
        print(f'  ✓  DrugBank severity loaded — {len(lookup) // 2} pairs | dist={dict(sorted(dist.items()))}')
    except Exception as e:
        print(f'  ⚠  Severity DB load error: {e}')
    return lookup

# this function is used to load severity from ddinter
def load_severity_from_ddinter(csv_paths: List[str]) -> Dict[tuple, int]:
    import csv
    lookup: Dict[tuple, int] = {}
    for path in csv_paths:
        if not os.path.exists(path):
            print(f'  ⚠  DDInter CSV not found: {path}')
            continue
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in reader.fieldnames or []}
            drug_a_col = cols.get('drug_a') or cols.get('drug a')
            drug_b_col = cols.get('drug_b') or cols.get('drug b')
            level_col = cols.get('level') or cols.get('risk level') or cols.get('risk')
            if not (drug_a_col and drug_b_col and level_col):
                print(f'  ⚠  DDInter CSV has unexpected columns: {reader.fieldnames}')
                continue
            for row in reader:
                a = normalize_drug_name((row.get(drug_a_col) or '').strip())
                b = normalize_drug_name((row.get(drug_b_col) or '').strip())
                lvl = (row.get(level_col) or '').strip().lower()
                if not a or not b:
                    continue
                sev = DDINTER_LEVEL_TO_SEVERITY.get(lvl)
                if sev is None:
                    continue
                lookup[a, b] = sev
                lookup[b, a] = sev
    pair_count = len(lookup) // 2
    dist_pairs = Counter()
    seen = set()
    for (a, b), v in lookup.items():
        if (b, a) in seen:
            continue
        seen.add((a, b))
        dist_pairs[v] += 1
    print(f'  ✓  DDInter severity loaded — {pair_count} pairs | dist={dict(sorted(dist_pairs.items()))}')
    return lookup

# this function is used to load name to id map
def load_name_to_id_map(db_path: str) -> Dict[str, str]:
    name_to_id: Dict[str, str] = {}
    if not os.path.exists(db_path):
        print(f'  ⚠  drugbank.db not found — Lipinski lookup will be unavailable during training')
        return name_to_id
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT id, LOWER(name) FROM drugs')
        for drug_id, name in c.fetchall():
            name_to_id[name] = drug_id
        conn.close()
        print(f'  ✓  Name→ID map loaded — {len(name_to_id)} drugs')
    except Exception as e:
        print(f'  ⚠  Name→ID map load error: {e}')
    return name_to_id

# this function is used to handle drug name to kg tensor
def drug_name_to_kg_tensor(name: str, kg_embeddings: Dict[str, np.ndarray], device: str) -> Optional[torch.Tensor]:
    emb = kg_embeddings.get(name.lower())
    if emb is None:
        return None
    return torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

# this function is used to handle drug id to lipinski tensor
def drug_id_to_lipinski_tensor(drug_id: str, lipinski: Dict[str, np.ndarray], device: str) -> Optional[torch.Tensor]:
    feats = lipinski.get(drug_id)
    if feats is None:
        return None
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

# this function is used to handle kg or zero
def kg_or_zero(name: str, kg_embeddings: Dict, device: str) -> torch.Tensor:
    t = drug_name_to_kg_tensor(name, kg_embeddings, device)
    return t if t is not None else torch.zeros(1, KG_DIM, device=device)

# this function is used to handle lipinski or zero
def lipinski_or_zero(drug_id: str, lipinski: Dict, device: str) -> torch.Tensor:
    t = drug_id_to_lipinski_tensor(drug_id, lipinski, device)
    return t if t is not None else torch.zeros(1, LIPINSKI_DIM, device=device)

# this function is used to handle lip by name or zero
def lip_by_name_or_zero(name: str, name_to_id: Dict[str, str], lipinski: Dict[str, np.ndarray], device: str) -> torch.Tensor:
    drug_id = name_to_id.get(name.lower())
    if drug_id is None:
        return torch.zeros(1, LIPINSKI_DIM, device=device)
    return lipinski_or_zero(drug_id, lipinski, device)

# this class groups logic for NERDataset
class NERDataset(Dataset):

    # this function is used to set up initial values for this object
    def __init__(self, sentences: List[DDISentence], tokenizer, max_length: int=128):
        self.samples = []
        self._build(sentences, tokenizer, max_length)

    # this function is used to handle build
    def _build(self, sentences, tokenizer, max_length):
        for sent in sentences:
            encoding = tokenizer(sent.text, max_length=max_length, truncation=True, padding='max_length', return_offsets_mapping=True, return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            offsets = encoding['offset_mapping'].squeeze(0).tolist()
            ner_labels = [NER_PAD_LABEL] * max_length
            for idx, (ts, te) in enumerate(offsets):
                if not (ts == 0 and te == 0):
                    ner_labels[idx] = 0
            for entity in sent.entities:
                first = True
                for idx, (ts, te) in enumerate(offsets):
                    if ts == 0 and te == 0:
                        continue
                    if ts >= entity.start and te <= entity.end + 1:
                        ner_labels[idx] = 1 if first else 2
                        first = False
            self.samples.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'ner_labels': torch.tensor(ner_labels, dtype=torch.long), 'text': sent.text})

    # this function is used to handle len
    def __len__(self):
        return len(self.samples)

    # this function is used to handle getitem
    def __getitem__(self, i):
        return self.samples[i]

# this class groups logic for DDIDataset
class DDIDataset(Dataset):

    # this function is used to set up initial values for this object
    def __init__(self, sentences: List[DDISentence], tokenizer, max_length: int=128):
        self.samples = []
        self._build(sentences, tokenizer, max_length)

    # this function is used to handle build
    def _build(self, sentences, tokenizer, max_length):
        for sent in sentences:
            if not sent.interactions:
                continue
            encoding = tokenizer(sent.text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            for interaction in sent.interactions:
                ddi_type = interaction.get('type', 'false')
                ddi_label = LABEL2IDX.get(ddi_type, 0)
                e1_id = interaction.get('e1', '')
                e2_id = interaction.get('e2', '')
                drug_a = next((e.text for e in sent.entities if e.id == e1_id), '')
                drug_b = next((e.text for e in sent.entities if e.id == e2_id), '')
                self.samples.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'label': torch.tensor(ddi_label, dtype=torch.long), 'drug_a': drug_a, 'drug_b': drug_b, 'text': sent.text})

    # this function is used to handle len
    def __len__(self):
        return len(self.samples)

    # this function is used to handle getitem
    def __getitem__(self, i):
        return self.samples[i]

# this class groups logic for SeverityDataset
class SeverityDataset(Dataset):

    # this function is used to set up initial values for this object
    def __init__(self, sentences: List[DDISentence], tokenizer, severity_lookup: Dict, name_resolver: Optional[Callable[[str], str]]=None, max_length: int=128, neg_to_pos_ratio: float=1.0, seed: int=42):
        self.samples = []
        self._build(sentences, tokenizer, severity_lookup, name_resolver, max_length, neg_to_pos_ratio, seed)

    # this function is used to handle build
    def _build(self, sentences, tokenizer, severity_lookup, name_resolver, max_length, neg_to_pos_ratio, seed):
        import random
        rng = random.Random(seed)
        pos_samples = []
        neg_samples = []
        for sent in sentences:
            if not sent.interactions:
                continue
            encoding = tokenizer(sent.text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            for interaction in sent.interactions:
                e1_id = interaction.get('e1', '')
                e2_id = interaction.get('e2', '')
                drug_a = next((e.text for e in sent.entities if e.id == e1_id), '')
                drug_b = next((e.text for e in sent.entities if e.id == e2_id), '')
                a_key = name_resolver(drug_a) if name_resolver else normalize_drug_name(drug_a)
                b_key = name_resolver(drug_b) if name_resolver else normalize_drug_name(drug_b)
                ddi_type = (interaction.get('type', 'false') or 'false').lower()
                is_ddi = bool(interaction.get('ddi', False))
                if not is_ddi or ddi_type == 'false':
                    sev_label = 0
                else:
                    sev_label = severity_lookup.get((a_key, b_key))
                    if sev_label is None:
                        continue
                sample = {'input_ids': input_ids, 'attention_mask': attention_mask, 'severity_label': torch.tensor(sev_label, dtype=torch.long), 'drug_a': drug_a, 'drug_b': drug_b, 'text': sent.text}
                if sev_label == 0:
                    neg_samples.append(sample)
                else:
                    pos_samples.append(sample)
        if neg_to_pos_ratio <= 0:
            chosen_negs = []
        else:
            k = int(round(len(pos_samples) * float(neg_to_pos_ratio)))
            k = min(k, len(neg_samples))
            chosen_negs = rng.sample(neg_samples, k) if k and len(neg_samples) >= k else neg_samples
        self.samples = pos_samples + chosen_negs
        rng.shuffle(self.samples)

    # this function is used to handle len
    def __len__(self):
        return len(self.samples)

    # this function is used to handle getitem
    def __getitem__(self, i):
        return self.samples[i]

# this function is used to handle collate ner
def collate_ner(batch):
    return {'input_ids': torch.stack([b['input_ids'] for b in batch]), 'attention_mask': torch.stack([b['attention_mask'] for b in batch]), 'ner_labels': torch.stack([b['ner_labels'] for b in batch]), 'text': [b['text'] for b in batch]}

# this function is used to handle collate ddi
def collate_ddi(batch):
    return {'input_ids': torch.stack([b['input_ids'] for b in batch]), 'attention_mask': torch.stack([b['attention_mask'] for b in batch]), 'label': torch.stack([b['label'] for b in batch]), 'drug_a': [b['drug_a'] for b in batch], 'drug_b': [b['drug_b'] for b in batch], 'text': [b['text'] for b in batch]}

# this function is used to handle collate severity
def collate_severity(batch):
    return {'input_ids': torch.stack([b['input_ids'] for b in batch]), 'attention_mask': torch.stack([b['attention_mask'] for b in batch]), 'severity_label': torch.stack([b['severity_label'] for b in batch]), 'drug_a': [b['drug_a'] for b in batch], 'drug_b': [b['drug_b'] for b in batch], 'text': [b['text'] for b in batch]}

# this function is used to compute weights
def compute_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    arr = np.array(labels)
    classes = np.unique(arr)
    weights = compute_class_weight('balanced', classes=classes, y=arr)
    t = torch.ones(num_classes, dtype=torch.float32)
    for c, w in zip(classes, weights):
        t[int(c)] = float(w)
    return t

# this class groups logic for BalancedSoftmaxLoss
class BalancedSoftmaxLoss(nn.Module):

    # this function is used to set up initial values for this object
    def __init__(self, class_counts: List[int], eps: float=1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=0.0) + float(eps)
        self.register_buffer('log_prior', torch.log(counts))

    # this function is used to handle forward
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj = logits + self.log_prior.to(logits.device)
        return nn.functional.cross_entropy(adj, targets)

# this function is used to report ner
def report_ner(true, pred):
    names = ['O', 'B-DRUG', 'I-DRUG']
    print(classification_report(true, pred, target_names=names, labels=[0, 1, 2], zero_division=0))
    return f1_score(true, pred, average='macro', zero_division=0, labels=[1, 2])

# this function is used to report ddi
def report_ddi(true, pred):
    names = ['false', 'mechanism', 'effect', 'advise', 'int']
    print(classification_report(true, pred, target_names=names, labels=list(range(5)), zero_division=0))
    return f1_score(true, pred, average='macro', zero_division=0)

# this function is used to report severity
def report_severity(true, pred):
    names = ['safe', 'caution', 'warning', 'danger']
    print(classification_report(true, pred, target_names=names, labels=list(range(4)), zero_division=0))
    return f1_score(true, pred, average='macro', zero_division=0)

# this function is used to train stage1 ner
def train_stage1_ner(data_dir: str, output_dir: str, model_name: str='emilyalsentzer/Bio_ClinicalBERT', num_epochs: int=5, batch_size: int=16, lr: float=2e-05, val_split: float=0.15):
    print('\n' + '=' * 60)
    print('STAGE 1 — NER HEAD TRAINING')
    print('=' * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    corpus_dir = os.path.join(data_dir, 'DDICorpus')
    train_all, test_sents = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(train_all, test_size=val_split, random_state=42)
    print(f'Train: {len(train_sents)} | Val: {len(val_sents)} | Test: {len(test_sents)}')
    train_ds = NERDataset(train_sents, tokenizer)
    val_ds = NERDataset(val_sents, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ner, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ner, num_workers=0)
    all_labels = [l for s in train_ds.samples for l in s['ner_labels'].tolist() if l != NER_PAD_LABEL]
    ner_w = compute_weights(all_labels, num_classes=3).to(device)
    print(f'NER class weights: {ner_w.tolist()}')
    model = MedGuardModel(model_name=model_name).to(device)
    model.set_stage(STAGE_NER)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=ner_w, ignore_index=NER_PAD_LABEL)
    best_f1 = 0.0
    save_path = os.path.join(output_dir, 'stage1_ner_best.pt')
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['ner_labels'].to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            B, T, C = outputs['ner_logits'].shape
            loss = criterion(outputs['ner_logits'].view(-1, C), labs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f'  [train] epoch={epoch + 1} step={step + 1}/{len(train_loader)} loss={loss.item():.4f}')
        avg_loss = total_loss / max(1, len(train_loader))
        model.eval()
        all_true, all_pred = ([], [])
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labs = batch['ner_labels'].to(device)
                logits = model(input_ids=ids, attention_mask=mask)['ner_logits']
                preds = logits.argmax(-1).view(-1).cpu().tolist()
                true = labs.view(-1).cpu().tolist()
                for p, t in zip(preds, true):
                    if t != NER_PAD_LABEL:
                        all_pred.append(p)
                        all_true.append(t)
        print(f'\nEpoch {epoch + 1}/{num_epochs}  |  loss={avg_loss:.4f}')
        macro_f1 = report_ner(all_true, all_pred)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            print(f'  ✓  Stage 1 best saved (NER macro-F1={best_f1:.4f}) → {save_path}')
    print(f'\nStage 1 complete. Best NER macro-F1 (B-DRUG+I-DRUG): {best_f1:.4f}')
    return save_path

# this function is used to train stage2 interaction
def train_stage2_interaction(data_dir: str, output_dir: str, stage1_ckpt: str, model_name: str='emilyalsentzer/Bio_ClinicalBERT', num_epochs: int=10, batch_size: int=4, lr: float=0.0001, accumulation_steps: int=4, val_split: float=0.15):
    print('\n' + '=' * 60)
    print('STAGE 2 — INTERACTION HEAD TRAINING')
    print('=' * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    corpus_dir = os.path.join(data_dir, 'DDICorpus')
    kg_path = os.path.join(data_dir, 'knowledge_graph.pkl')
    lipinski_path = os.path.join(data_dir, 'DB_compounds_lipinski.csv')
    db_path = os.path.join(data_dir, 'drugbank.db')
    train_all, test_sents = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(train_all, test_size=val_split, random_state=42)
    kg_embeddings = load_kg_embeddings(kg_path)
    lipinski = load_lipinski_features(lipinski_path)
    name_to_id = load_name_to_id_map(db_path)
    train_ds = DDIDataset(train_sents, tokenizer)
    val_ds = DDIDataset(val_sents, tokenizer)
    print(f'Train samples: {len(train_ds)} | Val samples: {len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ddi, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ddi, num_workers=0)
    ddi_labels_list = [s['label'].item() for s in train_ds.samples]
    ddi_w = compute_weights(ddi_labels_list, num_classes=5).to(device)
    print(f'DDI class weights: {[round(x, 3) for x in ddi_w.tolist()]}')
    print(f'DDI class dist: {dict(sorted(Counter(ddi_labels_list).items()))}')
    model = MedGuardModel(model_name=model_name).to(device)
    if os.path.exists(stage1_ckpt):
        model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print(f'  ✓  Loaded Stage 1 weights from {stage1_ckpt}')
    else:
        print(f'  ⚠  Stage 1 checkpoint not found — training from scratch (not recommended)')
    model.set_stage(STAGE_INTERACTION)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=ddi_w)
    best_f1 = 0.0
    save_path = os.path.join(output_dir, 'stage2_interaction_best.pt')
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['label'].to(device)
            kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_a']], dim=0)
            kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_b']], dim=0)
            lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_a']], dim=0)
            lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_b']], dim=0)
            outputs = model(input_ids=ids, attention_mask=mask, kg_embedding_a=kg_a, kg_embedding_b=kg_b, lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
            loss = criterion(outputs['interaction_logits'], labs)
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % 10 == 0:
                print(f'  [train] epoch={epoch + 1} step={step + 1}/{len(train_loader)} loss={loss.item() * accumulation_steps:.4f}')
        avg_loss = total_loss / max(1, len(train_loader))
        model.eval()
        all_true, all_pred = ([], [])
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labs = batch['label'].to(device)
                kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_a']], dim=0)
                kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_b']], dim=0)
                lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_a']], dim=0)
                lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_b']], dim=0)
                out = model(input_ids=ids, attention_mask=mask, kg_embedding_a=kg_a, kg_embedding_b=kg_b, lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
                all_pred.extend(out['interaction_logits'].argmax(-1).cpu().tolist())
                all_true.extend(labs.cpu().tolist())
        print(f'\nEpoch {epoch + 1}/{num_epochs}  |  loss={avg_loss:.4f}')
        macro_f1 = report_ddi(all_true, all_pred)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            print(f'  ✓  Stage 2 best saved (DDI macro-F1={best_f1:.4f}) → {save_path}')
    print(f'\nStage 2 complete. Best DDI macro-F1: {best_f1:.4f}')
    return save_path

# this function is used to train stage3 severity
def train_stage3_severity(data_dir: str, output_dir: str, stage2_ckpt: str, model_name: str='emilyalsentzer/Bio_ClinicalBERT', num_epochs: int=10, batch_size: int=4, lr: float=0.0001, accumulation_steps: int=4, val_split: float=0.15):
    print('\n' + '=' * 60)
    print('STAGE 3 — SEVERITY HEAD TRAINING')
    print('  (Labels: curated DDInter risk levels)')
    print('=' * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    corpus_dir = os.path.join(data_dir, 'DDICorpus')
    kg_path = os.path.join(data_dir, 'knowledge_graph.pkl')
    ddinter_dir = data_dir
    lipinski_path = os.path.join(data_dir, 'DB_compounds_lipinski.csv')
    train_all, _ = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(train_all, test_size=val_split, random_state=42)
    kg_embeddings = load_kg_embeddings(kg_path)
    kg_name_to_id, kg_id_to_name = load_kg_name_maps(kg_path)
    lipinski = load_lipinski_features(lipinski_path)
    db_path = os.path.join(data_dir, 'drugbank.db')
    name_to_id = load_name_to_id_map(db_path)
    drugbank_canon = load_drugbank_canonical_map(db_path)
    ddinter_paths = [os.path.join(ddinter_dir, f) for f in os.listdir(ddinter_dir) if f.lower().startswith('ddinter_code_') and f.lower().endswith('.csv')]
    severity_lookup = load_severity_from_ddinter(ddinter_paths)
    ddinter_vocab = sorted({a for a, _ in severity_lookup.keys()})
    snapshot_db = os.path.join(data_dir, 'linking_snapshots.sqlite')
    resolve_name, resolve_stats = build_stage3_resolver(data_dir=data_dir, kg_name_to_id=kg_name_to_id, kg_id_to_name=kg_id_to_name, snapshot_db_path=snapshot_db, config_extra={'stage': '3', 'resolver_vocab': 'ddinter'})
    _flush_linker = resolve_stats.get('_flush')
    pos_total = 0
    pos_matched = 0
    pos_skipped_class = 0
    unmatched_examples = []
    for sent in train_sents:
        for ia in sent.interactions or []:
            ddi_type = (ia.get('type', 'false') or 'false').lower()
            is_ddi = bool(ia.get('ddi', False))
            if not is_ddi or ddi_type == 'false':
                continue
            e1_id = ia.get('e1', '')
            e2_id = ia.get('e2', '')
            drug_a = next((e.text for e in sent.entities if e.id == e1_id), '')
            drug_b = next((e.text for e in sent.entities if e.id == e2_id), '')
            pos_total += 1
            a_key = resolve_name(drug_a)
            b_key = resolve_name(drug_b)
            if looks_like_drug_class(normalize_text(drug_a)) or looks_like_drug_class(normalize_text(drug_b)):
                pos_skipped_class += 1
                continue
            if severity_lookup.get((a_key, b_key)) is not None:
                pos_matched += 1
            elif len(unmatched_examples) < 10:
                unmatched_examples.append((drug_a, drug_b, a_key, b_key))
    cov = pos_matched / pos_total if pos_total else 0.0
    print('\n── Severity label coverage (DDI Corpus → DDInter) ──')
    print(f'Positive interaction pairs in corpus : {pos_total:,}')
    print(f'Matched to DDInter risk levels       : {pos_matched:,}')
    print(f'Coverage                             : {cov:.2%}')
    print(f'Skipped (drug-class mentions)        : {pos_skipped_class:,}')
    print(f'Linker snapshot run_id               : {resolve_stats.get('run_id')} (db={snapshot_db})')
    if unmatched_examples:
        print('Unmatched examples (raw -> normalized keys):')
        for ra, rb, ka, kb in unmatched_examples:
            print(f'  - {ra} / {rb}  ->  {ka} / {kb}')
    train_ds = SeverityDataset(train_sents, tokenizer, severity_lookup, name_resolver=resolve_name, neg_to_pos_ratio=1.0)
    val_ds = SeverityDataset(val_sents, tokenizer, severity_lookup, name_resolver=resolve_name, neg_to_pos_ratio=1.0)
    if callable(_flush_linker):
        _flush_linker()
    print(f'Train samples: {len(train_ds)} | Val samples: {len(val_ds)}')
    sev_labels = [s['severity_label'].item() for s in train_ds.samples]
    sev_dist = dict(sorted(Counter(sev_labels).items()))
    print(f'Severity label dist (DDInter curated): {sev_dist}')
    print('Note: unlabeled pairs (not in DDInter) are excluded from Stage 3 training.')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_severity, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_severity, num_workers=0)
    counts = Counter(sev_labels)
    class_counts = [int(counts.get(i, 0)) for i in range(4)]
    print(f'Severity class counts (train): {class_counts}')
    criterion = BalancedSoftmaxLoss(class_counts=class_counts).to(device)
    print('Severity loss: BalancedSoftmax (logit adjustment) enabled')
    model = MedGuardModel(model_name=model_name).to(device)
    if os.path.exists(stage2_ckpt):
        model.load_state_dict(torch.load(stage2_ckpt, map_location=device))
        print(f'  ✓  Loaded Stage 2 weights from {stage2_ckpt}')
    else:
        print(f'  ⚠  Stage 2 checkpoint not found — not recommended')
    model.set_stage(STAGE_SEVERITY)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    best_f1 = 0.0
    save_path = os.path.join(output_dir, 'stage3_severity_best.pt')
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['severity_label'].to(device)
            kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_a']], dim=0)
            kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_b']], dim=0)
            lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_a']], dim=0)
            lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_b']], dim=0)
            outputs = model(input_ids=ids, attention_mask=mask, kg_embedding_a=kg_a, kg_embedding_b=kg_b, lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
            loss = criterion(outputs['severity_logits'], labs) / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % 10 == 0:
                print(f'  [train] epoch={epoch + 1} step={step + 1}/{len(train_loader)} loss={loss.item() * accumulation_steps:.4f}')
        avg_loss = total_loss / max(1, len(train_loader))
        model.eval()
        all_true, all_pred = ([], [])
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labs = batch['severity_label'].to(device)
                kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_a']], dim=0)
                kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device) for n in batch['drug_b']], dim=0)
                lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_a']], dim=0)
                lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device) for n in batch['drug_b']], dim=0)
                out = model(input_ids=ids, attention_mask=mask, kg_embedding_a=kg_a, kg_embedding_b=kg_b, lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
                all_pred.extend(out['severity_logits'].argmax(-1).cpu().tolist())
                all_true.extend(labs.cpu().tolist())
        print(f'\nEpoch {epoch + 1}/{num_epochs}  |  loss={avg_loss:.4f}')
        macro_f1 = report_severity(all_true, all_pred)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            print(f'  ✓  Stage 3 best saved (Severity macro-F1={best_f1:.4f}) → {save_path}')
    print(f'\nStage 3 complete. Best Severity macro-F1: {best_f1:.4f}')
    print('(Severity labels: curated DDInter KB labels; evaluated on linked subset.)')
    return save_path
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MedGuard Trainer')
    parser.add_argument('--stage', type=str, default='all', choices=['1', '2', '3', 'all'], help='Which stage to train: 1 (NER), 2 (Interaction), 3 (Severity), all')
    parser.add_argument('--epochs1', type=int, default=5)
    parser.add_argument('--epochs2', type=int, default=10)
    parser.add_argument('--epochs3', type=int, default=10)
    parser.add_argument('--model', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    args = parser.parse_args()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
    ckpt1 = os.path.join(OUTPUT_DIR, 'stage1_ner_best.pt')
    ckpt2 = os.path.join(OUTPUT_DIR, 'stage2_interaction_best.pt')
    if args.stage in ('1', 'all'):
        train_stage1_ner(DATA_DIR, OUTPUT_DIR, model_name=args.model, num_epochs=args.epochs1)
    if args.stage in ('2', 'all'):
        if args.stage == '2' and (not os.path.exists(ckpt1)):
            raise SystemExit(f'Stage 2 requires Stage 1 checkpoint at: {ckpt1}\nRun: python -m app.models.trainer --stage 1')
        train_stage2_interaction(DATA_DIR, OUTPUT_DIR, ckpt1, model_name=args.model, num_epochs=args.epochs2)
    if args.stage in ('3', 'all'):
        if args.stage == '3' and (not os.path.exists(ckpt2)):
            raise SystemExit(f'Stage 3 requires Stage 2 checkpoint at: {ckpt2}\nRun: python -m app.models.trainer --stage 2 (after stage 1), or --stage all')
        train_stage3_severity(DATA_DIR, OUTPUT_DIR, ckpt2, model_name=args.model, num_epochs=args.epochs3)
