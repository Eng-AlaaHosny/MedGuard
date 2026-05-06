"""
MedGuard — Three-Stage Separate Trainer
========================================
Implements Option B: truly separate training for each head.
Each stage saves its own checkpoint. The encoder is progressively frozen.

Stage 1 — NER Head
  - All parameters trainable (encoder learns NER-specific representations).
  - Loss: CrossEntropyLoss on token labels (ignore_index=-100 for padding/CLS/SEP).
  - Class weights: balanced (O tokens dominate — >90% of all tokens).
  - Saves: checkpoints/stage1_ner_best.pt

Stage 2 — Interaction Head
  - Loads Stage 1 checkpoint (encoder already tuned for clinical NER).
  - Encoder FROZEN — preserves NER representations.
  - KG embeddings (128-dim) + Lipinski features (5-dim) fused with BERT repr.
  - Loss: CrossEntropyLoss with class weights (DDI corpus is class-imbalanced:
    ~63% false, ~20% mechanism, ~13% effect, ~3% advise, ~1% int).
  - Saves: checkpoints/stage2_interaction_best.pt

Stage 3 — Severity Head
  - Loads Stage 2 checkpoint.
  - Encoder FROZEN, Interaction head FROZEN.
  - Severity labels: curated DDInter risk levels mapped into
    safe/caution/warning/danger classes.
  - Unlabeled positive pairs (not linkable to DDInter) are excluded and
    coverage is reported explicitly.
  - Loss: Balanced Softmax (logit adjustment) for long-tail imbalance.
  - Saves: checkpoints/stage3_severity_best.pt

Run from backend/:
    python -m app.models.trainer --stage 1   (NER)
    python -m app.models.trainer --stage 2   (Interaction)
    python -m app.models.trainer --stage 3   (Severity)
    python -m app.models.trainer --stage all (run all 3 sequentially)
"""

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

from app.models.medguard_model import (
    MedGuardModel, load_tokenizer,
    STAGE_NER, STAGE_INTERACTION, STAGE_SEVERITY,
    DDI_LABELS, KG_DIM, LIPINSKI_DIM,
)
from app.data.preprocessor import load_ddi_corpus, DDISentence
from app.data.entity_linker import build_stage3_resolver, looks_like_drug_class, normalize_text

# Windows PowerShell often defaults to a non-UTF8 codepage. This keeps logs
# stable even when printing unicode symbols in status lines.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Label maps ────────────────────────────────────────────────────────────────

LABEL2IDX = {'false': 0, 'mechanism': 1, 'effect': 2, 'advise': 3, 'int': 4}
IDX2LABEL  = {v: k for k, v in LABEL2IDX.items()}

NER_PAD_LABEL = -100   # ignored by CrossEntropyLoss

# ── Severity silver-label map ─────────────────────────────────────────────────
# This mapping is the ONLY source of severity labels for Stage 3 training.
# It is a heuristic / distant-supervision approach:
#   - DrugBank interaction descriptions contain severity-indicating keywords.
#   - map_severity() in drugbank_processor.py converts descriptions to 0-3.
#   - At training time, the SQLite database provides these pre-computed labels.
# Academic note: this is silver labeling, NOT gold annotation. Results should
# be reported with this caveat in the paper.
SEVERITY_LABELS_MAP = {0: 'safe', 1: 'caution', 2: 'warning', 3: 'danger'}


# ── Severity (Option A): curated risk levels from DDInter ─────────────────────
#
# DDInter provides categorical risk levels (Minor/Moderate/Major/Contraindicated/Unknown).
# We map them into the 4-class MedGuard severity head:
#   safe(0) / caution(1) / warning(2) / danger(3)
#
# Important academic note:
# - This is NOT a heuristic silver label. It is a curated clinical KB label.
# - It is still a "knowledge base label", not directly annotated on DDI Corpus sentences.
#   We therefore evaluate severity on the subset of DDI Corpus pairs that can be
#   linked to DDInter risk levels (coverage reported explicitly).
DDINTER_LEVEL_TO_SEVERITY = {
    "minor": 1,            # caution
    "moderate": 2,         # warning
    "major": 3,            # danger
    "contraindicated": 3,  # danger
}


# ── Name normalization / linking (prevents Stage 3 coverage collapse) ─────────
#
# DDI Corpus entity text and DDInter drug names will not always match exactly.
# Before giving up (and training on almost-zero labeled positives), we apply:
#  1) deterministic normalization (casefold, parentheses removal, punctuation)
#  2) canonicalization through DrugBank (if a normalized name matches DrugBank)
#  3) strict fuzzy fallback only for remaining unmatched names (high cutoff)
#
# Academic note:
# - (1) and (2) are deterministic string linking steps (reportable & reproducible).
# - (3) is optional; if used, report how many matches came from fuzzy linking.
_SALT_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "acetate", "phosphate", "sulfate", "sulphate", "nitrate", "chloride",
    "bromide", "iodide", "tartrate", "citrate", "succinate", "fumarate",
    "gluconate", "mesylate", "besylate", "tosylate", "maleate",
}

_ALIAS_MAP = {
    # Common brand → generic (deterministic, reportable)
    "toradol": "ketorolac",
    "ultram": "tramadol",
    "celebrex": "celecoxib",
}


def normalize_drug_name(name: str) -> str:
    """Conservative, deterministic normalization for linking."""
    if not name:
        return ""
    s = name.strip().lower()
    # drop parenthetical qualifiers (e.g., "Abametapir (topical)")
    if "(" in s and ")" in s:
        import re
        s = re.sub(r"\([^)]*\)", " ", s)
    # normalize punctuation -> spaces
    for ch in [",", ".", ";", ":", "'", "\"", "/", "\\", "+", "-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    # optional salt stripping (helps common variants, but stays conservative)
    parts = [p for p in s.split() if p not in _SALT_WORDS]
    s2 = " ".join(parts).strip()
    out = s2 if s2 else s
    # deterministic aliasing (last step)
    return _ALIAS_MAP.get(out, out)


def load_drugbank_canonical_map(db_path: str) -> Dict[str, str]:
    """
    Build mapping: normalized_name -> canonical normalized name (from DrugBank).
    Used only as a bridging dictionary to improve name matching.
    """
    canon: Dict[str, str] = {}
    if not os.path.exists(db_path):
        return canon
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM drugs")
        for (name,) in c.fetchall():
            n = normalize_drug_name(name or "")
            if not n:
                continue
            canon.setdefault(n, n)
        conn.close()
    except Exception:
        return canon
    return canon


def make_name_resolver(*args, **kwargs):
    raise RuntimeError("Deprecated. Use build_stage3_resolver().")


# ── Resource loaders ──────────────────────────────────────────────────────────

def load_kg_embeddings(kg_path: str) -> Dict[str, np.ndarray]:
    """
    Load node2vec KG embeddings keyed by lower-case drug name.
    Returns empty dict if file missing — training proceeds without KG fusion
    (zero vectors used in model.fuse_drug_features).
    """
    if not os.path.exists(kg_path):
        print(f"  ⚠  KG not found at {kg_path} — training without KG embeddings")
        return {}
    try:
        with open(kg_path, 'rb') as f:
            data = pickle.load(f)
        embeddings   = data.get('embeddings', {})
        name_to_id   = data.get('drug_name_to_id', {})
        name_to_emb  = {
            name: embeddings[did]
            for name, did in name_to_id.items()
            if did in embeddings
        }
        print(f"  ✓  KG loaded — {len(name_to_emb)} drug embeddings")
        return name_to_emb
    except Exception as e:
        print(f"  ⚠  KG load error: {e}")
        return {}


def load_kg_name_maps(kg_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load Knowledge Graph name maps:
      - drug_name_to_id: lower-case name -> DrugBank ID
      - drug_id_to_name: DrugBank ID -> canonical DrugBank name

    These maps allow deterministic canonicalization (brand/variant -> DrugBank name)
    before linking into external KBs like DDInter.
    """
    if not os.path.exists(kg_path):
        return {}, {}
    try:
        with open(kg_path, "rb") as f:
            data = pickle.load(f)
        name_to_id = data.get("drug_name_to_id", {}) or {}
        id_to_name = data.get("drug_id_to_name", {}) or {}
        # Keep both raw lower-case keys and normalized keys so that our
        # normalization pipeline can still match KG entries.
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
        return name_to_id, id_to_name
    except Exception:
        return {}, {}


def load_lipinski_features(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load Lipinski features keyed by DrugBank ID.
    Returns dict[drug_id → np.ndarray shape (5,)].
    Features: molecular_weight, n_hba, n_hbd, logp, ro5_fulfilled.
    Z-score normalized (except ro5_fulfilled which is binary).
    """
    if not os.path.exists(csv_path):
        print(f"  ⚠  Lipinski CSV not found at {csv_path} — training without Lipinski features")
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        feat_cols = ['molecular_weight', 'n_hba', 'n_hbd', 'logp', 'ro5_fulfilled']

        # Z-score normalize continuous features
        for col in ['molecular_weight', 'n_hba', 'n_hbd', 'logp']:
            mean = df[col].mean()
            std  = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std

        # ro5_fulfilled: boolean → float 0/1
        df['ro5_fulfilled'] = df['ro5_fulfilled'].astype(float)

        id_col = 'ID'   # confirmed from CSV inspection
        result = {}
        for _, row in df.iterrows():
            drug_id  = str(row[id_col])
            features = row[feat_cols].values.astype(np.float32)
            result[drug_id] = features

        print(f"  ✓  Lipinski loaded — {len(result)} drug feature vectors")
        return result
    except Exception as e:
        print(f"  ⚠  Lipinski load error: {e}")
        return {}


def load_severity_from_db(db_path: str) -> Dict:
    """
    Load pre-computed DrugBank severity labels.
    These are silver labels from keyword matching — used for Stage 3 only.
    Returns dict[(drug_a_lower, drug_b_lower) → severity_int].
    """
    lookup = {}
    if not os.path.exists(db_path):
        print(f"  ⚠  drugbank.db not found — Stage 3 severity will be zero-only")
        return lookup
    try:
        conn = sqlite3.connect(db_path)
        c    = conn.cursor()
        c.execute('''
            SELECT LOWER(da.name), LOWER(db.name), i.severity
            FROM interactions i
            JOIN drugs da ON i.drug_a_id = da.id
            JOIN drugs db ON i.drug_b_id = db.id
        ''')
        for drug_a, drug_b, severity in c.fetchall():
            lookup[(drug_a, drug_b)] = severity
            lookup[(drug_b, drug_a)] = severity
        conn.close()
        dist = Counter(lookup.values())
        print(f"  ✓  DrugBank severity loaded — {len(lookup)//2} pairs | dist={dict(sorted(dist.items()))}")
    except Exception as e:
        print(f"  ⚠  Severity DB load error: {e}")
    return lookup


def load_severity_from_ddinter(csv_paths: List[str]) -> Dict[tuple, int]:
    """
    Load curated DDInter risk levels from one or more CSV files.

    Expected header (as provided by DDInter downloads):
        DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level

    Returns:
        dict[(drug_a_lower, drug_b_lower) -> severity_int in {1,2,3}]
    Notes:
        - "Unknown" levels are ignored (not used for training).
        - We store both (a,b) and (b,a) for symmetric lookup.
        - 'safe' (0) is NOT assigned here; missing pairs are treated as unlabeled.
    """
    import csv

    lookup: Dict[tuple, int] = {}
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  ⚠  DDInter CSV not found: {path}")
            continue

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # Defensive: tolerate column name variations
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            drug_a_col = cols.get("drug_a") or cols.get("drug a")
            drug_b_col = cols.get("drug_b") or cols.get("drug b")
            level_col  = cols.get("level")  or cols.get("risk level") or cols.get("risk")

            if not (drug_a_col and drug_b_col and level_col):
                print(f"  ⚠  DDInter CSV has unexpected columns: {reader.fieldnames}")
                continue

            for row in reader:
                a = normalize_drug_name((row.get(drug_a_col) or "").strip())
                b = normalize_drug_name((row.get(drug_b_col) or "").strip())
                lvl = (row.get(level_col) or "").strip().lower()
                if not a or not b:
                    continue
                sev = DDINTER_LEVEL_TO_SEVERITY.get(lvl)
                if sev is None:
                    continue
                lookup[(a, b)] = sev
                lookup[(b, a)] = sev

    # lookup contains both (a,b) and (b,a). Report pair-level distribution.
    pair_count = len(lookup) // 2
    dist_pairs = Counter()
    seen = set()
    for (a, b), v in lookup.items():
        if (b, a) in seen:
            continue
        seen.add((a, b))
        dist_pairs[v] += 1
    print(f"  ✓  DDInter severity loaded — {pair_count} pairs | dist={dict(sorted(dist_pairs.items()))}")
    return lookup


def load_name_to_id_map(db_path: str) -> Dict[str, str]:
    """
    Build a lowercase drug-name → DrugBank-ID map from the SQLite database.
    Used to resolve DDI Corpus drug names to IDs for Lipinski lookup at
    training time — matches exactly what routes.py does at inference time.
    Without this, Lipinski features are always zero during training while
    inference uses real values (train/inference mismatch).
    """
    name_to_id: Dict[str, str] = {}
    if not os.path.exists(db_path):
        print(f"  ⚠  drugbank.db not found — Lipinski lookup will be unavailable during training")
        return name_to_id
    try:
        conn = sqlite3.connect(db_path)
        c    = conn.cursor()
        c.execute('SELECT id, LOWER(name) FROM drugs')
        for drug_id, name in c.fetchall():
            name_to_id[name] = drug_id
        conn.close()
        print(f"  ✓  Name→ID map loaded — {len(name_to_id)} drugs")
    except Exception as e:
        print(f"  ⚠  Name→ID map load error: {e}")
    return name_to_id


# ── Tensor helpers ────────────────────────────────────────────────────────────

def drug_name_to_kg_tensor(
    name: str,
    kg_embeddings: Dict[str, np.ndarray],
    device: str,
) -> Optional[torch.Tensor]:
    """Return (1, 128) float tensor or None."""
    emb = kg_embeddings.get(name.lower())
    if emb is None:
        return None
    return torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)


def drug_id_to_lipinski_tensor(
    drug_id: str,
    lipinski: Dict[str, np.ndarray],
    device: str,
) -> Optional[torch.Tensor]:
    """Return (1, 5) float tensor or None."""
    feats = lipinski.get(drug_id)
    if feats is None:
        return None
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)


def kg_or_zero(name: str, kg_embeddings: Dict, device: str) -> torch.Tensor:
    t = drug_name_to_kg_tensor(name, kg_embeddings, device)
    return t if t is not None else torch.zeros(1, KG_DIM, device=device)


def lipinski_or_zero(drug_id: str, lipinski: Dict, device: str) -> torch.Tensor:
    t = drug_id_to_lipinski_tensor(drug_id, lipinski, device)
    return t if t is not None else torch.zeros(1, LIPINSKI_DIM, device=device)


def lip_by_name_or_zero(
    name:       str,
    name_to_id: Dict[str, str],
    lipinski:   Dict[str, np.ndarray],
    device:     str,
) -> torch.Tensor:
    """
    Resolve drug name → DrugBank ID → Lipinski tensor.
    Mirrors routes.py resolve_drug_id() + get_lipinski_tensor() so that
    training and inference see identical Lipinski inputs.
    Returns (1, LIPINSKI_DIM) zero tensor when either lookup fails.
    """
    drug_id = name_to_id.get(name.lower())
    if drug_id is None:
        return torch.zeros(1, LIPINSKI_DIM, device=device)
    return lipinski_or_zero(drug_id, lipinski, device)


# ── Datasets ─────────────────────────────────────────────────────────────────

class NERDataset(Dataset):
    """
    Stage 1 dataset — token-level NER labels.
    One sample per sentence. Labels: -100 (pad/special), 0 (O), 1 (B-DRUG), 2 (I-DRUG).
    """

    def __init__(
        self,
        sentences:  List[DDISentence],
        tokenizer,
        max_length: int = 128,
    ):
        self.samples = []
        self._build(sentences, tokenizer, max_length)

    def _build(self, sentences, tokenizer, max_length):
        for sent in sentences:
            encoding = tokenizer(
                sent.text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt',
            )
            input_ids      = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            offsets        = encoding['offset_mapping'].squeeze(0).tolist()

            # Default: NER_PAD_LABEL for special/padding tokens, 0 (O) for real tokens
            ner_labels = [NER_PAD_LABEL] * max_length
            for idx, (ts, te) in enumerate(offsets):
                if not (ts == 0 and te == 0):
                    ner_labels[idx] = 0   # real token → O

            # Assign B-DRUG / I-DRUG from entity spans
            for entity in sent.entities:
                first = True
                for idx, (ts, te) in enumerate(offsets):
                    if ts == 0 and te == 0:
                        continue
                    if ts >= entity.start and te <= entity.end + 1:
                        ner_labels[idx] = 1 if first else 2
                        first = False

            self.samples.append({
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'ner_labels':     torch.tensor(ner_labels, dtype=torch.long),
                'text':           sent.text,
            })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class DDIDataset(Dataset):
    """
    Stage 2 dataset — sentence-level DDI interaction labels.
    One sample per interaction pair within a sentence.
    Includes drug names (for KG lookup) and DrugBank IDs (for Lipinski lookup).
    """

    def __init__(
        self,
        sentences:  List[DDISentence],
        tokenizer,
        max_length: int = 128,
    ):
        self.samples = []
        self._build(sentences, tokenizer, max_length)

    def _build(self, sentences, tokenizer, max_length):
        for sent in sentences:
            if not sent.interactions:
                continue
            encoding = tokenizer(
                sent.text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids      = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            for interaction in sent.interactions:
                ddi_type  = interaction.get('type', 'false')
                ddi_label = LABEL2IDX.get(ddi_type, 0)

                e1_id  = interaction.get('e1', '')
                e2_id  = interaction.get('e2', '')
                drug_a = next((e.text for e in sent.entities if e.id == e1_id), '')
                drug_b = next((e.text for e in sent.entities if e.id == e2_id), '')

                self.samples.append({
                    'input_ids':      input_ids,
                    'attention_mask': attention_mask,
                    'label':          torch.tensor(ddi_label, dtype=torch.long),
                    'drug_a':         drug_a,
                    'drug_b':         drug_b,
                    'text':           sent.text,
                })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class SeverityDataset(Dataset):
    """
    Stage 3 dataset — sentence-level severity labels.
    Labels are curated DDInter risk levels mapped into {1,2,3}.
    Pairs missing from DDInter are treated as UNLABELED and skipped
    (we do NOT default them to 'safe').
    """

    def __init__(
        self,
        sentences:       List[DDISentence],
        tokenizer,
        severity_lookup: Dict,
        name_resolver:   Optional[Callable[[str], str]] = None,
        max_length:      int = 128,
        neg_to_pos_ratio: float = 1.0,
        seed:            int = 42,
    ):
        self.samples = []
        self._build(sentences, tokenizer, severity_lookup, name_resolver, max_length, neg_to_pos_ratio, seed)

    def _build(self, sentences, tokenizer, severity_lookup, name_resolver, max_length, neg_to_pos_ratio, seed):
        import random
        rng = random.Random(seed)
        pos_samples = []
        neg_samples = []
        for sent in sentences:
            if not sent.interactions:
                continue
            encoding = tokenizer(
                sent.text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids      = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            for interaction in sent.interactions:
                e1_id  = interaction.get('e1', '')
                e2_id  = interaction.get('e2', '')
                drug_a = next((e.text for e in sent.entities if e.id == e1_id), '')
                drug_b = next((e.text for e in sent.entities if e.id == e2_id), '')
                a_key = name_resolver(drug_a) if name_resolver else normalize_drug_name(drug_a)
                b_key = name_resolver(drug_b) if name_resolver else normalize_drug_name(drug_b)

                # Define severity=0 (safe) for explicit "no interaction" cases in DDI Corpus.
                # For interacting pairs, require a curated DDInter risk label (1–3).
                # Academic note: DDInter does not label "safe". To avoid overwhelming
                # the training set with easy negatives, we downsample safe examples.
                ddi_type = (interaction.get('type', 'false') or 'false').lower()
                is_ddi   = bool(interaction.get('ddi', False))
                if (not is_ddi) or ddi_type == "false":
                    sev_label = 0
                else:
                    sev_label = severity_lookup.get((a_key, b_key))
                    if sev_label is None:
                        continue

                sample = {
                    'input_ids':      input_ids,
                    'attention_mask': attention_mask,
                    'severity_label': torch.tensor(sev_label, dtype=torch.long),
                    'drug_a':         drug_a,
                    'drug_b':         drug_b,
                    'text':           sent.text,
                }

                if sev_label == 0:
                    neg_samples.append(sample)
                else:
                    pos_samples.append(sample)

        # Downsample negatives to a controlled ratio
        if neg_to_pos_ratio <= 0:
            chosen_negs = []
        else:
            k = int(round(len(pos_samples) * float(neg_to_pos_ratio)))
            k = min(k, len(neg_samples))
            chosen_negs = rng.sample(neg_samples, k) if k and len(neg_samples) >= k else neg_samples

        self.samples = pos_samples + chosen_negs
        rng.shuffle(self.samples)

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ── Collate functions ─────────────────────────────────────────────────────────

def collate_ner(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'ner_labels':     torch.stack([b['ner_labels']     for b in batch]),
        'text':           [b['text'] for b in batch],
    }


def collate_ddi(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'label':          torch.stack([b['label']          for b in batch]),
        'drug_a':         [b['drug_a'] for b in batch],
        'drug_b':         [b['drug_b'] for b in batch],
        'text':           [b['text']   for b in batch],
    }


def collate_severity(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'severity_label': torch.stack([b['severity_label'] for b in batch]),
        'drug_a':         [b['drug_a'] for b in batch],
        'drug_b':         [b['drug_b'] for b in batch],
        'text':           [b['text']   for b in batch],
    }


# ── Class weight computation ──────────────────────────────────────────────────

def compute_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Balanced class weights using sklearn. Returns float tensor of shape (num_classes,)."""
    arr     = np.array(labels)
    classes = np.unique(arr)
    weights = compute_class_weight('balanced', classes=classes, y=arr)
    t = torch.ones(num_classes, dtype=torch.float32)
    for c, w in zip(classes, weights):
        t[int(c)] = float(w)
    return t


# ── Balanced softmax (logit adjustment) ───────────────────────────────────────
class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax / logit adjustment using training label counts.

    Implementation: CrossEntropyLoss on (logits + log(counts)).
    Reference idea: Menon et al. / "Balanced Softmax" long-tail classification.
    """

    def __init__(self, class_counts: List[int], eps: float = 1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=0.0) + float(eps)
        self.register_buffer("log_prior", torch.log(counts))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj = logits + self.log_prior.to(logits.device)
        return nn.functional.cross_entropy(adj, targets)

# ── Metric reporting ──────────────────────────────────────────────────────────

def report_ner(true, pred):
    names = ['O', 'B-DRUG', 'I-DRUG']
    print(classification_report(true, pred, target_names=names,
                                 labels=[0, 1, 2], zero_division=0))
    return f1_score(true, pred, average='macro', zero_division=0, labels=[1, 2])


def report_ddi(true, pred):
    names = ['false', 'mechanism', 'effect', 'advise', 'int']
    print(classification_report(true, pred, target_names=names,
                                 labels=list(range(5)), zero_division=0))
    return f1_score(true, pred, average='macro', zero_division=0)


def report_severity(true, pred):
    names = ['safe', 'caution', 'warning', 'danger']
    print(classification_report(true, pred, target_names=names,
                                 labels=list(range(4)), zero_division=0))
    return f1_score(true, pred, average='macro', zero_division=0)


# ── Stage 1: NER ─────────────────────────────────────────────────────────────

def train_stage1_ner(
    data_dir:    str,
    output_dir:  str,
    model_name:  str   = "emilyalsentzer/Bio_ClinicalBERT",
    num_epochs:  int   = 5,
    batch_size:  int   = 16,
    lr:          float = 2e-5,
    val_split:   float = 0.15,
):
    """
    Stage 1: Train the NER head.
    All parameters trainable. Encoder learns clinical entity representations.
    """
    print("\n" + "="*60)
    print("STAGE 1 — NER HEAD TRAINING")
    print("="*60)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    corpus_dir = os.path.join(data_dir, 'DDICorpus')
    train_all, test_sents = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(train_all, test_size=val_split, random_state=42)
    print(f"Train: {len(train_sents)} | Val: {len(val_sents)} | Test: {len(test_sents)}")

    train_ds = NERDataset(train_sents, tokenizer)
    val_ds   = NERDataset(val_sents,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_ner, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_ner, num_workers=0)

    # Class weights
    all_labels = [l for s in train_ds.samples for l in s['ner_labels'].tolist()
                  if l != NER_PAD_LABEL]
    ner_w = compute_weights(all_labels, num_classes=3).to(device)
    print(f"NER class weights: {ner_w.tolist()}")

    model = MedGuardModel(model_name=model_name).to(device)
    model.set_stage(STAGE_NER)

    optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * num_epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=ner_w, ignore_index=NER_PAD_LABEL)

    best_f1   = 0.0
    save_path = os.path.join(output_dir, 'stage1_ner_best.pt')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            labs  = batch['ner_labels'].to(device)

            outputs    = model(input_ids=ids, attention_mask=mask)
            B, T, C    = outputs['ner_logits'].shape
            loss       = criterion(outputs['ner_logits'].view(-1, C), labs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(f"  [train] epoch={epoch+1} step={step+1}/{len(train_loader)} loss={loss.item():.4f}")
            # full training run (no diagnostic early stop)

        avg_loss = total_loss / max(1, len(train_loader))

        # ── Evaluate ──
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labs = batch['ner_labels'].to(device)

                logits = model(input_ids=ids, attention_mask=mask)['ner_logits']
                preds  = logits.argmax(-1).view(-1).cpu().tolist()
                true   = labs.view(-1).cpu().tolist()
                for p, t in zip(preds, true):
                    if t != NER_PAD_LABEL:
                        all_pred.append(p)
                        all_true.append(t)

        print(f"\nEpoch {epoch+1}/{num_epochs}  |  loss={avg_loss:.4f}")
        macro_f1 = report_ner(all_true, all_pred)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            print(f"  ✓  Stage 1 best saved (NER macro-F1={best_f1:.4f}) → {save_path}")

    print(f"\nStage 1 complete. Best NER macro-F1 (B-DRUG+I-DRUG): {best_f1:.4f}")
    return save_path


# ── Stage 2: Interaction ──────────────────────────────────────────────────────

def train_stage2_interaction(
    data_dir:           str,
    output_dir:         str,
    stage1_ckpt:        str,
    model_name:         str   = "emilyalsentzer/Bio_ClinicalBERT",
    num_epochs:         int   = 10,
    batch_size:         int   = 4,
    lr:                 float = 1e-4,   # higher LR — encoder frozen
    accumulation_steps: int   = 4,
    val_split:          float = 0.15,
):
    """
    Stage 2: Train the Interaction head.
    Encoder FROZEN (loaded from Stage 1).
    drug_fusion, pair_projection, interaction_head are trainable.
    KG embeddings and Lipinski features are fused here.
    """
    print("\n" + "="*60)
    print("STAGE 2 — INTERACTION HEAD TRAINING")
    print("="*60)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    corpus_dir    = os.path.join(data_dir, 'DDICorpus')
    kg_path       = os.path.join(data_dir, 'knowledge_graph.pkl')
    lipinski_path = os.path.join(data_dir, 'DB_compounds_lipinski.csv')
    db_path       = os.path.join(data_dir, 'drugbank.db')

    train_all, test_sents = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(train_all, test_size=val_split, random_state=42)

    kg_embeddings = load_kg_embeddings(kg_path)
    lipinski      = load_lipinski_features(lipinski_path)
    name_to_id    = load_name_to_id_map(db_path)

    train_ds = DDIDataset(train_sents, tokenizer)
    val_ds   = DDIDataset(val_sents,   tokenizer)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_ddi, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_ddi, num_workers=0)

    ddi_labels_list = [s['label'].item() for s in train_ds.samples]
    ddi_w = compute_weights(ddi_labels_list, num_classes=5).to(device)
    print(f"DDI class weights: {[round(x,3) for x in ddi_w.tolist()]}")
    print(f"DDI class dist: {dict(sorted(Counter(ddi_labels_list).items()))}")

    model = MedGuardModel(model_name=model_name).to(device)
    if os.path.exists(stage1_ckpt):
        model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print(f"  ✓  Loaded Stage 1 weights from {stage1_ckpt}")
    else:
        print(f"  ⚠  Stage 1 checkpoint not found — training from scratch (not recommended)")

    model.set_stage(STAGE_INTERACTION)

    optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * num_epochs // accumulation_steps
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=ddi_w)

    best_f1   = 0.0
    save_path = os.path.join(output_dir, 'stage2_interaction_best.pt')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['label'].to(device)

            kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device)
                               for n in batch['drug_a']], dim=0)
            kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device)
                               for n in batch['drug_b']], dim=0)
            lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                for n in batch['drug_a']], dim=0)
            lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                for n in batch['drug_b']], dim=0)

            outputs  = model(input_ids=ids, attention_mask=mask,
                             kg_embedding_a=kg_a,  kg_embedding_b=kg_b,
                             lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
            loss     = criterion(outputs['interaction_logits'], labs)
            loss     = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 10 == 0:
                print(f"  [train] epoch={epoch+1} step={step+1}/{len(train_loader)} loss={loss.item()*accumulation_steps:.4f}")
            # full training run (no diagnostic early stop)

        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labs = batch['label'].to(device)
                kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device)
                                   for n in batch['drug_a']], dim=0)
                kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device)
                                   for n in batch['drug_b']], dim=0)
                lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                    for n in batch['drug_a']], dim=0)
                lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                    for n in batch['drug_b']], dim=0)
                out  = model(input_ids=ids, attention_mask=mask,
                             kg_embedding_a=kg_a,  kg_embedding_b=kg_b,
                             lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
                all_pred.extend(out['interaction_logits'].argmax(-1).cpu().tolist())
                all_true.extend(labs.cpu().tolist())

        print(f"\nEpoch {epoch+1}/{num_epochs}  |  loss={avg_loss:.4f}")
        macro_f1 = report_ddi(all_true, all_pred)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            print(f"  ✓  Stage 2 best saved (DDI macro-F1={best_f1:.4f}) → {save_path}")

    print(f"\nStage 2 complete. Best DDI macro-F1: {best_f1:.4f}")
    return save_path


# ── Stage 3: Severity ─────────────────────────────────────────────────────────

def train_stage3_severity(
    data_dir:           str,
    output_dir:         str,
    stage2_ckpt:        str,
    model_name:         str   = "emilyalsentzer/Bio_ClinicalBERT",
    num_epochs:         int   = 10,
    batch_size:         int   = 4,
    lr:                 float = 1e-4,
    accumulation_steps: int   = 4,
    val_split:          float = 0.15,
):
    """
    Stage 3: Train the Severity head.
    Encoder FROZEN. Interaction head FROZEN (preserves Stage 2 weights).
    Labels: curated DDInter risk levels (Minor/Moderate/Major/Contraindicated)
    mapped into MedGuard severity classes (caution/warning/danger).

    Academic note:
      - These are curated KB risk levels, not heuristic keyword labels.
      - Severity is evaluated on the subset of DDI Corpus pairs covered by DDInter.
        Report coverage (%) and label distribution explicitly.
    """
    print("\n" + "="*60)
    print("STAGE 3 — SEVERITY HEAD TRAINING")
    print("  (Labels: curated DDInter risk levels)")
    print("="*60)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    corpus_dir    = os.path.join(data_dir, 'DDICorpus')
    kg_path       = os.path.join(data_dir, 'knowledge_graph.pkl')
    ddinter_dir   = data_dir  # expect ddinter_code_*.csv stored alongside other data files
    lipinski_path = os.path.join(data_dir, 'DB_compounds_lipinski.csv')

    train_all, _   = load_ddi_corpus(corpus_dir)
    train_sents, val_sents = train_test_split(train_all, test_size=val_split, random_state=42)

    kg_embeddings    = load_kg_embeddings(kg_path)
    kg_name_to_id, kg_id_to_name = load_kg_name_maps(kg_path)
    lipinski         = load_lipinski_features(lipinski_path)
    # Lipinski name→ID mapping still uses drugbank.db (optional); if missing, Lipinski becomes zeros.
    db_path          = os.path.join(data_dir, 'drugbank.db')
    name_to_id       = load_name_to_id_map(db_path)
    drugbank_canon   = load_drugbank_canonical_map(db_path)

    # DDInter curated severity labels
    ddinter_paths = [
        os.path.join(ddinter_dir, f)
        for f in os.listdir(ddinter_dir)
        if f.lower().startswith("ddinter_code_") and f.lower().endswith(".csv")
    ]
    severity_lookup  = load_severity_from_ddinter(ddinter_paths)

    # Academic long-term resolver:
    # - dictionary-first (offline)
    # - explicit drug-vs-class routing
    # - versioned snapshots to a SQLite DB
    ddinter_vocab = sorted({a for (a, _) in severity_lookup.keys()})
    snapshot_db = os.path.join(data_dir, "linking_snapshots.sqlite")
    resolve_name, resolve_stats = build_stage3_resolver(
        data_dir=data_dir,
        kg_name_to_id=kg_name_to_id,
        kg_id_to_name=kg_id_to_name,
        snapshot_db_path=snapshot_db,
        config_extra={"stage": "3", "resolver_vocab": "ddinter"},
    )
    _flush_linker = resolve_stats.get("_flush")

    # ── Coverage report (prevents wasting training time) ──────────────────────
    pos_total = 0
    pos_matched = 0
    pos_skipped_class = 0
    unmatched_examples = []
    for sent in train_sents:
        for ia in (sent.interactions or []):
            ddi_type = (ia.get("type", "false") or "false").lower()
            is_ddi = bool(ia.get("ddi", False))
            if (not is_ddi) or ddi_type == "false":
                continue
            e1_id = ia.get("e1", "")
            e2_id = ia.get("e2", "")
            drug_a = next((e.text for e in sent.entities if e.id == e1_id), "")
            drug_b = next((e.text for e in sent.entities if e.id == e2_id), "")
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

    cov = (pos_matched / pos_total) if pos_total else 0.0
    print("\n── Severity label coverage (DDI Corpus → DDInter) ──")
    print(f"Positive interaction pairs in corpus : {pos_total:,}")
    print(f"Matched to DDInter risk levels       : {pos_matched:,}")
    print(f"Coverage                             : {cov:.2%}")
    print(f"Skipped (drug-class mentions)        : {pos_skipped_class:,}")
    print(f"Linker snapshot run_id               : {resolve_stats.get('run_id')} (db={snapshot_db})")
    if unmatched_examples:
        print("Unmatched examples (raw -> normalized keys):")
        for ra, rb, ka, kb in unmatched_examples:
            print(f"  - {ra} / {rb}  ->  {ka} / {kb}")

    # Include a controlled number of "safe" (DDI=false) examples so the head
    # can learn to output class 0 without drowning out curated (1–3) labels.
    train_ds = SeverityDataset(
        train_sents, tokenizer, severity_lookup, name_resolver=resolve_name, neg_to_pos_ratio=1.0
    )
    val_ds   = SeverityDataset(
        val_sents,   tokenizer, severity_lookup, name_resolver=resolve_name, neg_to_pos_ratio=1.0
    )
    if callable(_flush_linker):
        _flush_linker()
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Print severity label distribution + coverage (important for academic reporting)
    sev_labels = [s['severity_label'].item() for s in train_ds.samples]
    sev_dist   = dict(sorted(Counter(sev_labels).items()))
    print(f"Severity label dist (DDInter curated): {sev_dist}")
    print("Note: unlabeled pairs (not in DDInter) are excluded from Stage 3 training.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_severity, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_severity, num_workers=0)

    # Academic long-tail training: Balanced Softmax (logit adjustment).
    # Important: do NOT also use WeightedRandomSampler or class weights here.
    counts = Counter(sev_labels)
    class_counts = [int(counts.get(i, 0)) for i in range(4)]
    print(f"Severity class counts (train): {class_counts}")
    criterion = BalancedSoftmaxLoss(class_counts=class_counts).to(device)
    print("Severity loss: BalancedSoftmax (logit adjustment) enabled")

    model = MedGuardModel(model_name=model_name).to(device)
    if os.path.exists(stage2_ckpt):
        model.load_state_dict(torch.load(stage2_ckpt, map_location=device))
        print(f"  ✓  Loaded Stage 2 weights from {stage2_ckpt}")
    else:
        print(f"  ⚠  Stage 2 checkpoint not found — not recommended")

    model.set_stage(STAGE_SEVERITY)

    optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    best_f1   = 0.0
    save_path = os.path.join(output_dir, 'stage3_severity_best.pt')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['severity_label'].to(device)
            kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device)
                               for n in batch['drug_a']], dim=0)
            kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device)
                               for n in batch['drug_b']], dim=0)
            lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                for n in batch['drug_a']], dim=0)
            lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                for n in batch['drug_b']], dim=0)

            outputs = model(input_ids=ids, attention_mask=mask,
                            kg_embedding_a=kg_a,  kg_embedding_b=kg_b,
                            lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
            loss    = criterion(outputs['severity_logits'], labs) / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 10 == 0:
                print(f"  [train] epoch={epoch+1} step={step+1}/{len(train_loader)} loss={loss.item()*accumulation_steps:.4f}")
            # full training run (no diagnostic early stop)

        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labs = batch['severity_label'].to(device)
                kg_a = torch.cat([kg_or_zero(n, kg_embeddings, device)
                                   for n in batch['drug_a']], dim=0)
                kg_b = torch.cat([kg_or_zero(n, kg_embeddings, device)
                                   for n in batch['drug_b']], dim=0)
                lip_a = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                    for n in batch['drug_a']], dim=0)
                lip_b = torch.cat([lip_by_name_or_zero(n, name_to_id, lipinski, device)
                                    for n in batch['drug_b']], dim=0)
                out  = model(input_ids=ids, attention_mask=mask,
                             kg_embedding_a=kg_a,  kg_embedding_b=kg_b,
                             lipinski_feats_a=lip_a, lipinski_feats_b=lip_b)
                all_pred.extend(out['severity_logits'].argmax(-1).cpu().tolist())
                all_true.extend(labs.cpu().tolist())

        print(f"\nEpoch {epoch+1}/{num_epochs}  |  loss={avg_loss:.4f}")
        macro_f1 = report_severity(all_true, all_pred)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            print(f"  ✓  Stage 3 best saved (Severity macro-F1={best_f1:.4f}) → {save_path}")

    print(f"\nStage 3 complete. Best Severity macro-F1: {best_f1:.4f}")
    print("(Severity labels: curated DDInter KB labels; evaluated on linked subset.)")
    return save_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGuard Trainer")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help="Which stage to train: 1 (NER), 2 (Interaction), 3 (Severity), all")
    parser.add_argument('--epochs1', type=int, default=5)
    parser.add_argument('--epochs2', type=int, default=10)
    parser.add_argument('--epochs3', type=int, default=10)
    parser.add_argument('--model',   type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    args = parser.parse_args()

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR   = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

    ckpt1 = os.path.join(OUTPUT_DIR, 'stage1_ner_best.pt')
    ckpt2 = os.path.join(OUTPUT_DIR, 'stage2_interaction_best.pt')

    if args.stage in ('1', 'all'):
        train_stage1_ner(DATA_DIR, OUTPUT_DIR, model_name=args.model, num_epochs=args.epochs1)

    if args.stage in ('2', 'all'):
        if args.stage == '2' and not os.path.exists(ckpt1):
            raise SystemExit(
                f"Stage 2 requires Stage 1 checkpoint at: {ckpt1}\n"
                f"Run: python -m app.models.trainer --stage 1"
            )
        train_stage2_interaction(DATA_DIR, OUTPUT_DIR, ckpt1,
                                 model_name=args.model, num_epochs=args.epochs2)

    if args.stage in ('3', 'all'):
        if args.stage == '3' and not os.path.exists(ckpt2):
            raise SystemExit(
                f"Stage 3 requires Stage 2 checkpoint at: {ckpt2}\n"
                f"Run: python -m app.models.trainer --stage 2 (after stage 1), or --stage all"
            )
        train_stage3_severity(DATA_DIR, OUTPUT_DIR, ckpt2,
                              model_name=args.model, num_epochs=args.epochs3)
