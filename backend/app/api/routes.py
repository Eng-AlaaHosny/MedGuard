# this file defines api endpoints and runs model inference for drug pairs
import os
import pickle
import sqlite3
import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from app.models.medguard_model import MedGuardModel, DDI_LABELS, NER_LABELS, SEVERITY_LABELS, SEVERITY_COLORS, KG_DIM, LIPINSKI_DIM
from app.knowledge_graph.graph_builder import DrugKnowledgeGraph
from app.data.lipinski_processor import LipinskiProcessor
from app.data.kb_normalization import normalize_kb_text
router = APIRouter()
model: Optional[MedGuardModel] = None
interaction_model: Optional[MedGuardModel] = None
tokenizer = None
kg: Optional[DrugKnowledgeGraph] = None
lipinski: Optional[LipinskiProcessor] = None
MAX_LENGTH = 128
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DRUGBANK_DB_PATH = os.path.join(DATA_DIR, 'drugbank.db')

# this class groups logic for DDIRequest
class DDIRequest(BaseModel):
    drug_a: str
    drug_b: str
    text: Optional[str] = ''

# this class groups logic for DetectedEntity
class DetectedEntity(BaseModel):
    text: str
    start: int
    end: int
    label: str

# this class groups logic for DDIResponse
class DDIResponse(BaseModel):
    inference_text: str
    detected_entities: List[DetectedEntity]
    interaction_type: str
    interaction_type_idx: int
    interaction_source: str
    severity_label: str
    severity_level: int
    severity_color: str
    severity_source: str
    interaction_reason: str
    evidence_text: str
    evidence_source: str
    plain_guidance: str
    kg_context: Dict
    lipinski_context: Dict
    confidence: Dict
    modality_coverage: Dict

# this function is used to get resources
def get_resources():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail='Model is still loading — please retry in a moment.')
    return (model, interaction_model, tokenizer, kg, lipinski)

# this function is used to decode ner
def decode_ner(text: str, ner_logits: torch.Tensor, offset_mapping: List[tuple]) -> List[DetectedEntity]:
    predictions = ner_logits.argmax(dim=-1).squeeze(0).tolist()
    entities = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_label: Optional[str] = None

    # this function is used to handle flush
    def flush():
        nonlocal current_start, current_end, current_label
        if current_start is not None:
            entities.append(DetectedEntity(text=text[current_start:current_end], start=current_start, end=current_end, label=current_label))
            current_start = current_end = current_label = None
    for idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == 0 and token_end == 0:
            flush()
            continue
        label_name = NER_LABELS.get(predictions[idx], 'O')
        if label_name == 'B-DRUG':
            flush()
            current_start = token_start
            current_end = token_end
            current_label = 'B-DRUG'
        elif label_name == 'I-DRUG' and current_start is not None:
            current_end = token_end
        else:
            flush()
    flush()
    seen = set()
    unique = []
    for e in entities:
        key = (e.start, e.end)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique

# this function is used to find token span
def find_token_span(drug_name: str, offset_mapping: List[tuple], text: str) -> Optional[List[tuple]]:
    char_start = text.lower().find(drug_name.lower())
    if char_start == -1:
        return None
    char_end = char_start + len(drug_name)
    tok_start = tok_end = None
    for idx, (ts, te) in enumerate(offset_mapping):
        if ts == 0 and te == 0:
            continue
        if ts >= char_start and tok_start is None:
            tok_start = idx
        if te <= char_end:
            tok_end = idx
    if tok_start is not None and tok_end is not None:
        return [(tok_start, tok_end)]
    return None

# this function is used to get kg tensor
def get_kg_tensor(drug_name: str, graph: Optional[DrugKnowledgeGraph], device: str) -> Optional[torch.Tensor]:
    if graph is None:
        return None
    resolved_name = resolve_name_for_kg(drug_name, graph)
    emb = graph.get_drug_embedding(resolved_name)
    if emb is None:
        return None
    return torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

# this function is used to resolve drug id
def resolve_drug_id(drug_name: str, graph: Optional[DrugKnowledgeGraph]) -> Optional[str]:
    if graph is None:
        return None
    resolved_name = resolve_name_for_kg(drug_name, graph)
    return graph.drug_name_to_id.get(resolved_name.lower())

# this function is used to resolve name for kg
def resolve_name_for_kg(drug_name: str, graph: Optional[DrugKnowledgeGraph]) -> str:
    if graph is None:
        return drug_name
    raw = (drug_name or '').strip()
    if not raw:
        return raw
    direct = raw.lower()
    if direct in graph.drug_name_to_id:
        return direct
    norm = normalize_kb_text(raw)
    if norm in graph.drug_name_to_id:
        return norm
    if not os.path.exists(DRUGBANK_DB_PATH):
        return norm
    try:
        with sqlite3.connect(DRUGBANK_DB_PATH) as conn:
            row = conn.execute('\n                SELECT d.name\n                FROM drugbank_synonyms s\n                JOIN drugs d ON d.id = s.drug_id\n                WHERE s.norm_text = ?\n                LIMIT 1\n                ', (norm,)).fetchone()
            if row and row[0]:
                candidate = normalize_kb_text(str(row[0]))
                if candidate in graph.drug_name_to_id:
                    return candidate
            syn_rows = conn.execute('\n                SELECT s2.norm_text\n                FROM drugbank_synonyms s\n                JOIN drugbank_synonyms s2 ON s2.drug_id = s.drug_id\n                WHERE s.norm_text = ?\n                ', (norm,)).fetchall()
            for syn_row in syn_rows:
                if not syn_row or not syn_row[0]:
                    continue
                syn_norm = normalize_kb_text(str(syn_row[0]))
                if syn_norm in graph.drug_name_to_id:
                    return syn_norm
    except Exception:
        pass
    return norm

# this function is used to get lipinski tensor
def get_lipinski_tensor(drug_name: str, lip: Optional[LipinskiProcessor], graph: Optional[DrugKnowledgeGraph], device: str) -> Optional[torch.Tensor]:
    if lip is None:
        return None
    drug_id = resolve_drug_id(drug_name, graph)
    if drug_id is None:
        return None
    feats = lip.get_features(drug_id)
    if feats is None:
        return None
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

# this function is used to build kg context
def build_kg_context(drug_a: str, drug_b: str, graph: Optional[DrugKnowledgeGraph]) -> Dict:
    if graph is None:
        return {'status': 'Knowledge Graph not loaded'}
    resolved_a = resolve_name_for_kg(drug_a, graph)
    resolved_b = resolve_name_for_kg(drug_b, graph)
    a_avail = graph.check_drug_available(resolved_a)
    b_avail = graph.check_drug_available(resolved_b)
    result = {'drug_a': drug_a, 'drug_b': drug_b, 'resolved_drug_a': resolved_a, 'resolved_drug_b': resolved_b, 'drug_a_in_graph': a_avail, 'drug_b_in_graph': b_avail}
    if not a_avail or not b_avail:
        missing = [d for d, av in [(drug_a, a_avail), (drug_b, b_avail)] if not av]
        result['status'] = f"Data Unavailable - {', '.join(missing)} not in KG"
        result['known_interaction'] = None
        return result
    interaction_info = graph.get_interaction_info(resolved_a, resolved_b)
    if interaction_info:
        sev_map = {0: 'safe', 1: 'caution', 2: 'warning', 3: 'danger'}
        result['status'] = 'Known interaction found in DrugBank KG'
        result['known_interaction'] = {'severity': interaction_info.get('severity', 0), 'severity_label': sev_map.get(interaction_info.get('severity', 0), 'unknown'), 'description': interaction_info.get('description', ''), 'type': interaction_info.get('interaction_type', '')}
    else:
        result['status'] = 'Both drugs in KG — no direct interaction edge'
        result['known_interaction'] = None
    return result
DDI_TYPE_DESCRIPTIONS = {'false': 'No known pharmacological interaction between these drugs.', 'mechanism': 'Pharmacokinetic interaction: one drug alters the absorption, distribution, metabolism, or excretion of the other.', 'effect': 'Pharmacodynamic interaction: combined use alters the therapeutic or adverse effects of one or both drugs.', 'advise': 'Clinical guidance exists for this combination; consult prescribing information.', 'int': 'An interaction is reported; mechanism or clinical significance is unspecified.'}
SEVERITY_GUIDANCE = {'safe': "No high-risk signal was detected for this pair. Continue routine monitoring and follow your clinician's advice.", 'caution': 'Use caution with this pair. Discuss with a clinician or pharmacist, especially if symptoms change.', 'warning': 'This pair may carry clinically meaningful risk. Contact your clinician/pharmacist before continuing or adding doses.', 'danger': 'This pair may be high risk. Seek urgent clinician/pharmacist guidance before use, and get immediate help for severe symptoms.'}

# this function is used to build interaction reason
def build_interaction_reason(interaction_type: str, kg_context: Dict, drug_a: str, drug_b: str, severity_label: str) -> str:
    pair = f'{drug_a} and {drug_b}'
    base = DDI_TYPE_DESCRIPTIONS.get(interaction_type, 'Interaction detected.')
    reason = f'{pair}: {base}'
    if interaction_type == 'false' and severity_label in {'caution', 'warning', 'danger'}:
        reason = f'{pair}: The model flagged potential risk (severity: {severity_label}) even though the interaction subtype was not confidently classified.'
    return reason

# this function is used to extract pair evidence
def extract_pair_evidence(kg_context: Dict, severity_label: str) -> tuple[str, str]:
    known = (kg_context or {}).get('known_interaction') if isinstance(kg_context, dict) else None
    if known and known.get('description'):
        sev = known.get('severity')
        if severity_label == 'safe' and isinstance(sev, int) and (sev >= 1):
            return ('', 'none')
        return (str(known.get('description')).strip(), 'kg_context')
    return ('', 'none')

# this function is used to build lipinski context
def build_lipinski_context(drug_a: str, drug_b: str, lip: Optional[LipinskiProcessor], graph: Optional[DrugKnowledgeGraph]) -> Dict:
    if lip is None:
        return {'status': 'Lipinski data not loaded'}
    result = {}
    for drug_name in [drug_a, drug_b]:
        drug_id = resolve_drug_id(drug_name, graph)
        if drug_id and lip.is_drug_available(drug_id):
            feats = lip.get_features(drug_id)
            result[drug_name] = {'drugbank_id': drug_id, 'molecular_weight': float(feats[0]), 'n_hba': float(feats[1]), 'n_hbd': float(feats[2]), 'logp': float(feats[3]), 'ro5_fulfilled': bool(feats[4] > 0.5), 'available': True}
        else:
            result[drug_name] = {'available': False, 'status': 'Data Unavailable — drug not in Lipinski dataset'}
    return result

# this function is used to handle run pair inference
def run_pair_inference(drug_a: str, drug_b: str, inference_text: Optional[str]=None) -> DDIResponse:
    m, im, t, graph, lip = get_resources()
    device = str(next(m.parameters()).device)
    drug_a = drug_a.strip()
    drug_b = drug_b.strip()
    if not drug_a or not drug_b:
        raise HTTPException(status_code=400, detail='Both drug_a and drug_b are required.')
    if inference_text and inference_text.strip():
        infer_txt = inference_text.strip()
    else:
        infer_txt = f'{drug_a} and {drug_b} may interact. Concurrent use of {drug_a} with {drug_b} should be monitored.'
    encoding = t(infer_txt, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LENGTH, return_offsets_mapping=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()
    spans_a = find_token_span(drug_a, offset_mapping, infer_txt)
    spans_b = find_token_span(drug_b, offset_mapping, infer_txt)
    kg_emb_a = get_kg_tensor(drug_a, graph, device)
    kg_emb_b = get_kg_tensor(drug_b, graph, device)
    lip_feats_a = get_lipinski_tensor(drug_a, lip, graph, device)
    lip_feats_b = get_lipinski_tensor(drug_b, lip, graph, device)
    modality_coverage = {drug_a: {'kg_embedding': kg_emb_a is not None, 'lipinski_features': lip_feats_a is not None, 'token_span_found': spans_a is not None}, drug_b: {'kg_embedding': kg_emb_b is not None, 'lipinski_features': lip_feats_b is not None, 'token_span_found': spans_b is not None}}
    with torch.no_grad():
        outputs = m(input_ids=input_ids, attention_mask=attention_mask, drug_a_spans=spans_a, drug_b_spans=spans_b, kg_embedding_a=kg_emb_a, kg_embedding_b=kg_emb_b, lipinski_feats_a=lip_feats_a, lipinski_feats_b=lip_feats_b)
        interaction_outputs = (im or m)(input_ids=input_ids, attention_mask=attention_mask, drug_a_spans=spans_a, drug_b_spans=spans_b, kg_embedding_a=kg_emb_a, kg_embedding_b=kg_emb_b, lipinski_feats_a=lip_feats_a, lipinski_feats_b=lip_feats_b)
    interaction_probs = torch.softmax(interaction_outputs['interaction_logits'], dim=-1).squeeze(0)
    interaction_idx = int(interaction_probs.argmax().item())
    interaction_type = DDI_LABELS.get(interaction_idx, 'false')
    interaction_source = 'model'
    severity_probs = torch.softmax(outputs['severity_logits'], dim=-1).squeeze(0)
    severity_idx = int(severity_probs.argmax().item())
    severity_label = SEVERITY_LABELS.get(severity_idx, 'safe')
    severity_color = SEVERITY_COLORS.get(severity_label, '#28a745')
    severity_source = 'model'
    entities = decode_ner(infer_txt, outputs['ner_logits'], offset_mapping)
    confidence = {'interaction': {DDI_LABELS[i]: round(float(interaction_probs[i].item()), 4) for i in range(len(DDI_LABELS))}, 'severity': {SEVERITY_LABELS[i]: round(float(severity_probs[i].item()), 4) for i in range(len(SEVERITY_LABELS))}}
    kg_context = build_kg_context(drug_a, drug_b, graph)
    lipinski_context = build_lipinski_context(drug_a, drug_b, lip, graph)
    interaction_reason = build_interaction_reason(interaction_type, kg_context, drug_a, drug_b, severity_label)
    evidence_text, evidence_source = extract_pair_evidence(kg_context, severity_label)
    plain_guidance = SEVERITY_GUIDANCE.get(severity_label, 'Discuss this combination with a clinician/pharmacist before use.')
    return DDIResponse(inference_text=infer_txt, detected_entities=entities, interaction_type=interaction_type, interaction_type_idx=interaction_idx, interaction_source=interaction_source, severity_label=severity_label, severity_level=severity_idx, severity_color=severity_color, severity_source=severity_source, interaction_reason=interaction_reason, evidence_text=evidence_text, evidence_source=evidence_source, plain_guidance=plain_guidance, kg_context=kg_context, lipinski_context=lipinski_context, confidence=confidence, modality_coverage=modality_coverage)

@router.post('/analyze', response_model=DDIResponse)
# this async function is used to handle analyze interaction
async def analyze_interaction(request: DDIRequest):
    return run_pair_inference(request.drug_a, request.drug_b, request.text)

@router.get('/health')
# this function is used to handle health check
def health_check():
    return {'status': 'ready' if model is not None else 'loading', 'model_loaded': model is not None, 'interaction_model_loaded': interaction_model is not None, 'tokenizer_loaded': tokenizer is not None, 'kg_loaded': kg is not None, 'lipinski_loaded': lipinski is not None, 'anthropic_assistant_configured': bool(os.environ.get('ANTHROPIC_API_KEY', '').strip())}

@router.get('/drugs')
# this function is used to list kg drugs
def list_kg_drugs(limit: int=50):
    if kg is None:
        raise HTTPException(status_code=503, detail='Knowledge Graph not loaded.')
    drugs = list(kg.drug_name_to_id.keys())[:limit]
    return {'count': len(kg.drug_name_to_id), 'sample': drugs}
