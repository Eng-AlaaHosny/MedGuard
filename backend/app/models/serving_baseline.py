# Serving-path baseline: same 15% split as training, CLS pooling vs API span pooling.
import json
import os
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from app.data.preprocessor import load_ddi_corpus
from app.models.medguard_model import MedGuardModel, DDI_LABELS, SEVERITY_LABELS
from app.models.trainer import (
    DDIDataset,
    NERDataset,
    SeverityDataset,
    build_stage3_resolver,
    kg_or_zero,
    lip_by_name_or_zero,
    load_kg_embeddings,
    load_kg_name_maps,
    load_lipinski_features,
    load_name_to_id_map,
    load_severity_from_ddinter,
    normalize_drug_name,
    report_ner,
)

MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
DDI_NAMES = ['false', 'mechanism', 'effect', 'advise', 'int']
SEV_NAMES = ['safe', 'caution', 'warning', 'danger']


def find_token_span(drug_name: str, offset_mapping, text: str):
    if not drug_name:
        return None
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


def macro_and_report(true, pred, names):
    labels = list(range(len(names)))
    print(classification_report(true, pred, target_names=names, labels=labels, zero_division=0))
    per_class = f1_score(true, pred, average=None, labels=labels, zero_division=0)
    support = Counter(true)
    macro = float(f1_score(true, pred, average='macro', zero_division=0, labels=labels))
    detail = {
        names[i]: {'f1': round(float(per_class[i]), 4), 'support': int(support.get(i, 0))}
        for i in labels
    }
    return macro, detail


def bootstrap_macro(true, pred, n_labels, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    true = np.asarray(true)
    pred = np.asarray(pred)
    n = len(true)
    if n == 0:
        return {'n': 0, 'macro_f1': None, 'ci95_low': None, 'ci95_high': None}
    scores = []
    labels = list(range(n_labels))
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(true[idx], pred[idx], average='macro', zero_division=0, labels=labels))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    point = float(f1_score(true, pred, average='macro', zero_division=0, labels=labels))
    return {
        'n': int(n),
        'macro_f1': round(point, 4),
        'ci95_low': round(float(lo), 4),
        'ci95_high': round(float(hi), 4),
        'note': 'Bootstrap over this split predictions, not retraining seeds.',
    }


def pair_key(a, b):
    na, nb = normalize_drug_name(a), normalize_drug_name(b)
    return tuple(sorted((na, nb)))


def overlap_stats(train_samples, val_samples):
    train_pairs = {pair_key(s['drug_a'], s['drug_b']) for s in train_samples}
    val_pairs = [pair_key(s['drug_a'], s['drug_b']) for s in val_samples]
    hit = sum(1 for p in val_pairs if p in train_pairs)
    return {
        'val_pairs': len(val_pairs),
        'val_pairs_also_in_train': hit,
        'fraction': round(hit / max(1, len(val_pairs)), 4),
        'note': 'Split is by sentence (random_state=42, 15%). This counts normalized drug-pair overlap.',
    }


def load_model(path, device):
    model = MedGuardModel(model_name=MODEL_NAME).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def rss_mb():
    # Working set of this process. argtypes are required; without them the handle is truncated on 64-bit Windows.
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t),
        ]

    fn = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.WinDLL('kernel32').GetCurrentProcess()
    if not fn(handle, ctypes.byref(counters), counters.cb):
        return None
    return {
        'working_set_mb': round(counters.WorkingSetSize / (1024 * 1024), 1),
        'peak_working_set_mb': round(counters.PeakWorkingSetSize / (1024 * 1024), 1),
    }


def dump_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('wrote', path, flush=True)


def predict_pairs(model, samples, tokenizer, kg_embeddings, name_to_id, lipinski, device, pooling, head, batch_size=32, collect_labels=True):
    true, pred = [], []
    span_found = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            encodings = tokenizer(
                [s['text'] for s in batch],
                max_length=128,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt',
            )
            ids = encodings['input_ids'].to(device)
            mask = encodings['attention_mask'].to(device)
            offsets = encodings['offset_mapping'].tolist()
            spans_a = spans_b = None
            if pooling == 'span':
                spans_a, spans_b = [], []
                for sample, off in zip(batch, offsets):
                    sa = find_token_span(sample['drug_a'], off, sample['text'])
                    sb = find_token_span(sample['drug_b'], off, sample['text'])
                    spans_a.append(sa)
                    spans_b.append(sb)
                    if sa and sb:
                        span_found += 1
            kg_a = torch.cat([kg_or_zero(s['drug_a'], kg_embeddings, device) for s in batch], dim=0)
            kg_b = torch.cat([kg_or_zero(s['drug_b'], kg_embeddings, device) for s in batch], dim=0)
            lip_a = torch.cat([lip_by_name_or_zero(s['drug_a'], name_to_id, lipinski, device) for s in batch], dim=0)
            lip_b = torch.cat([lip_by_name_or_zero(s['drug_b'], name_to_id, lipinski, device) for s in batch], dim=0)
            out = model(
                input_ids=ids,
                attention_mask=mask,
                drug_a_spans=spans_a,
                drug_b_spans=spans_b,
                kg_embedding_a=kg_a,
                kg_embedding_b=kg_b,
                lipinski_feats_a=lip_a,
                lipinski_feats_b=lip_b,
            )
            logits = out['interaction_logits'] if head == 'ddi' else out['severity_logits']
            pred.extend(logits.argmax(-1).cpu().tolist())
            if collect_labels:
                label_key = 'label' if head == 'ddi' else 'severity_label'
                true.extend(int(s[label_key].item()) for s in batch)
            done = min(start + batch_size, len(samples))
            if done % 400 == 0 or done == len(samples):
                print(f'    {pooling} {head}: {done}/{len(samples)}', flush=True)
    return true, pred, {'span_both_found': span_found, 'n': len(samples)}


def eval_ner(model, val_sents, tokenizer, device, batch_size=16):
    ds = NERDataset(val_sents, tokenizer)
    true, pred = [], []
    model.eval()
    samples = ds.samples
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            ids = torch.stack([s['input_ids'] for s in batch]).to(device)
            mask = torch.stack([s['attention_mask'] for s in batch]).to(device)
            labs = torch.stack([s['ner_labels'] for s in batch])
            logits = model(input_ids=ids, attention_mask=mask)['ner_logits']
            preds = logits.argmax(-1).cpu()
            for row_pred, row_true in zip(preds, labs):
                for p, t in zip(row_pred.tolist(), row_true.tolist()):
                    if t != -100:
                        pred.append(p)
                        true.append(t)
    print('\nNER (token labels, B+I macro-F1 is the report metric)')
    report_ner(true, pred)
    macro_bi = float(f1_score(true, pred, average='macro', zero_division=0, labels=[1, 2]))
    return {'macro_f1_BI': round(macro_bi, 4), 'tokens': len(true)}


def disk_mb(path):
    return round(os.path.getsize(path) / (1024 * 1024), 1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')
    ckpt_dir = os.path.join(base, 'models', 'checkpoints')
    stage2_path = os.path.join(ckpt_dir, 'stage2_interaction_best.pt')
    stage2_report_path = os.path.join(ckpt_dir, 'stage2_interaction_best_baseline_04190.pt')
    stage3_path = os.path.join(ckpt_dir, 'stage3_severity_best.pt')
    print('device', device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_all, _ = load_ddi_corpus(os.path.join(data_dir, 'DDICorpus'))
    train_sents, val_sents = train_test_split(train_all, test_size=0.15, random_state=42)
    print(f'sentences train={len(train_sents)} val={len(val_sents)}')

    kg_embeddings = load_kg_embeddings(os.path.join(data_dir, 'knowledge_graph.pkl'))
    kg_name_to_id, kg_id_to_name = load_kg_name_maps(os.path.join(data_dir, 'knowledge_graph.pkl'))
    lipinski = load_lipinski_features(os.path.join(data_dir, 'DB_compounds_lipinski.csv'))
    name_to_id = load_name_to_id_map(os.path.join(data_dir, 'drugbank.db'))
    ddinter_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().startswith('ddinter_code_') and f.lower().endswith('.csv')
    ]
    severity_lookup = load_severity_from_ddinter(ddinter_paths)
    resolve_name, _ = build_stage3_resolver(
        data_dir=data_dir,
        kg_name_to_id=kg_name_to_id,
        kg_id_to_name=kg_id_to_name,
        snapshot_db_path=os.path.join(data_dir, 'linking_snapshots.sqlite'),
        config_extra={'stage': 'baseline', 'resolver_vocab': 'ddinter'},
    )

    train_ddi = DDIDataset(train_sents, tokenizer).samples
    val_ddi = DDIDataset(val_sents, tokenizer).samples
    train_sev = SeverityDataset(train_sents, tokenizer, severity_lookup, resolve_name).samples
    val_sev = SeverityDataset(val_sents, tokenizer, severity_lookup, resolve_name).samples
    print(f'DDI pairs train={len(train_ddi)} val={len(val_ddi)}')
    print(f'Severity pairs train={len(train_sev)} val={len(val_sev)}')

    results = {
        'device': device,
        'split': 'sentence-level train_test_split test_size=0.15 random_state=42',
        'pair_overlap_ddi': overlap_stats(train_ddi, val_ddi),
        'pair_overlap_severity': overlap_stats(train_sev, val_sev),
        'disk_mb': {
            'stage2_interaction_best.pt': disk_mb(stage2_path),
            'stage2_interaction_best_baseline_04190.pt': disk_mb(stage2_report_path),
            'stage3_severity_best.pt': disk_mb(stage3_path),
            'api_both': round(disk_mb(stage2_path) + disk_mb(stage3_path), 1),
        },
        'checkpoint_note': (
            'The API loads stage2_interaction_best.pt. That file was overwritten on 28 Apr 2026. '
            'stage2_interaction_best_baseline_04190.pt is the 24 Apr 2026 copy named for the report score 0.4190. '
            'stage3_severity_best.pt is still the 24 Apr 2026 file.'
        ),
    }
    out_dir = os.path.join(base, 'reports')
    out_path = os.path.join(out_dir, 'serving_baseline.json')
    print('DDI pair overlap', results['pair_overlap_ddi'])
    print('Severity pair overlap', results['pair_overlap_severity'])

    print('\nLoading stage2 API file (28 Apr overwrite)')
    stage2 = load_model(stage2_path, device)
    print('\nLoading stage2 report file (baseline_04190)')
    stage2_report = load_model(stage2_report_path, device)
    print('\nLoading stage3 (severity trunk)')
    stage3 = load_model(stage3_path, device)

    print('\nNER from stage3', flush=True)
    results['ner_stage3'] = eval_ner(stage3, val_sents, tokenizer, device)
    results['ram'] = rss_mb()
    dump_results(results, out_path)

    print('\nDDI report checkpoint pooling=cls (trainer protocol check)')
    true_r, pred_r, cov_r = predict_pairs(
        stage2_report, val_ddi, tokenizer, kg_embeddings, name_to_id, lipinski, device, 'cls', 'ddi'
    )
    macro_r, detail_r = macro_and_report(true_r, pred_r, DDI_NAMES)
    results['ddi_report04190_cls'] = {
        'macro_f1': round(macro_r, 4),
        'per_class': detail_r,
        'bootstrap': bootstrap_macro(true_r, pred_r, 5),
        'coverage': cov_r,
        'file': 'stage2_interaction_best_baseline_04190.pt',
    }
    print('bootstrap', results['ddi_report04190_cls']['bootstrap'], flush=True)
    dump_results(results, out_path)

    for pooling in ('cls', 'span'):
        print(f'\nDDI stage2 pooling={pooling}')
        true, pred, cov = predict_pairs(
            stage2, val_ddi, tokenizer, kg_embeddings, name_to_id, lipinski, device, pooling, 'ddi'
        )
        macro, detail = macro_and_report(true, pred, DDI_NAMES)
        results[f'ddi_stage2_{pooling}'] = {
            'macro_f1': round(macro, 4),
            'per_class': detail,
            'bootstrap': bootstrap_macro(true, pred, 5),
            'coverage': cov,
            'file': 'stage2_interaction_best.pt',
        }
        print('bootstrap', results[f'ddi_stage2_{pooling}']['bootstrap'], flush=True)
        if pooling == 'span':
            print('\nDDI report checkpoint pooling=span')
            true_rs, pred_rs, cov_rs = predict_pairs(
                stage2_report, val_ddi, tokenizer, kg_embeddings, name_to_id, lipinski, device, 'span', 'ddi'
            )
            macro_rs, detail_rs = macro_and_report(true_rs, pred_rs, DDI_NAMES)
            results['ddi_report04190_span'] = {
                'macro_f1': round(macro_rs, 4),
                'per_class': detail_rs,
                'bootstrap': bootstrap_macro(true_rs, pred_rs, 5),
                'coverage': cov_rs,
                'file': 'stage2_interaction_best_baseline_04190.pt',
            }
            print('bootstrap', results['ddi_report04190_span']['bootstrap'], flush=True)
        dump_results(results, out_path)
        print(f'DDI stage3 trunk pooling={pooling} (drift check, not the published checkpoint role)')
        true3, pred3, cov3 = predict_pairs(
            stage3, val_ddi, tokenizer, kg_embeddings, name_to_id, lipinski, device, pooling, 'ddi'
        )
        macro3, detail3 = macro_and_report(true3, pred3, DDI_NAMES)
        results[f'ddi_stage3trunk_{pooling}'] = {
            'macro_f1': round(macro3, 4),
            'per_class': detail3,
            'bootstrap': bootstrap_macro(true3, pred3, 5),
            'coverage': cov3,
        }
        print('bootstrap', results[f'ddi_stage3trunk_{pooling}']['bootstrap'], flush=True)
        dump_results(results, out_path)

    for pooling in ('cls', 'span'):
        print(f'\nSeverity stage3 pooling={pooling}')
        true, pred, cov = predict_pairs(
            stage3, val_sev, tokenizer, kg_embeddings, name_to_id, lipinski, device, pooling, 'severity'
        )
        macro, detail = macro_and_report(true, pred, SEV_NAMES)
        results[f'severity_stage3_{pooling}'] = {
            'macro_f1': round(macro, 4),
            'per_class': detail,
            'bootstrap': bootstrap_macro(true, pred, 4),
            'coverage': cov,
            'note': 'Evaluated on the linked severity subset built the same way as training.',
        }
        print('bootstrap', results[f'severity_stage3_{pooling}']['bootstrap'], flush=True)
        dump_results(results, out_path)

    # Drop the report-only copy so the RAM sample matches the two-model API.
    del stage2_report
    import gc
    gc.collect()

    # Latency of the current API pattern: two forwards, span pooling.
    # Labels are not read: DDI samples have no severity_label, and both heads run in one forward.
    n_lat = min(30, len(val_ddi))
    t0 = time.perf_counter()
    predict_pairs(
        stage3, val_ddi[:n_lat], tokenizer, kg_embeddings, name_to_id, lipinski, device,
        'span', 'ddi', batch_size=1, collect_labels=False,
    )
    predict_pairs(
        stage2, val_ddi[:n_lat], tokenizer, kg_embeddings, name_to_id, lipinski, device,
        'span', 'ddi', batch_size=1, collect_labels=False,
    )
    elapsed = time.perf_counter() - t0
    results['latency_dual_forward_span'] = {
        'n': n_lat,
        'seconds_total': round(elapsed, 2),
        'seconds_per_pair': round(elapsed / max(1, n_lat), 3),
        'note': 'Two full forwards (stage3 file + stage2 API file), span pooling, batch 1. CPU/GPU as device field.',
    }
    results['ram'] = rss_mb()
    print('latency', results['latency_dual_forward_span'], flush=True)
    dump_results(results, out_path)


if __name__ == '__main__':
    main()
