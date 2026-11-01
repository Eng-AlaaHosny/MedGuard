# Compare the current demo checkpoints with the span-trained checkpoints
# on validation pairs that do not appear in training. Span pooling, same 15% split.
import json
import os

from sklearn.model_selection import train_test_split

from app.data.preprocessor import load_ddi_corpus
from app.models.serving_baseline import load_model, pair_key, predict_pairs
from app.models.trainer import (
    DDIDataset,
    SeverityDataset,
    build_stage3_resolver,
    load_kg_embeddings,
    load_kg_name_maps,
    load_lipinski_features,
    load_name_to_id_map,
    load_severity_from_ddinter,
    report_ddi,
    report_severity,
)
from transformers import AutoTokenizer

MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'


def disjoint(train_samples, val_samples):
    train_pairs = {pair_key(sample['drug_a'], sample['drug_b']) for sample in train_samples}
    return [sample for sample in val_samples if pair_key(sample['drug_a'], sample['drug_b']) not in train_pairs]


def score(path, samples, tokenizer, kg_embeddings, name_to_id, lipinski, device, head, report_fn):
    if not os.path.exists(path):
        print('missing', path)
        return None
    model = load_model(path, device)
    true, pred, meta = predict_pairs(model, samples, tokenizer, kg_embeddings, name_to_id, lipinski, device, 'span', head, batch_size=16)
    print(path)
    macro = report_fn(true, pred)
    print('span_both_found', meta['span_both_found'], '/', meta['n'])
    del model
    return round(float(macro), 4)


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')
    ckpt_dir = os.path.join(base, 'models', 'checkpoints')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_all, _ = load_ddi_corpus(os.path.join(data_dir, 'DDICorpus'))
    train_sents, val_sents = train_test_split(train_all, test_size=0.15, random_state=42)
    kg_embeddings = load_kg_embeddings(os.path.join(data_dir, 'knowledge_graph.pkl'))
    kg_name_to_id, kg_id_to_name = load_kg_name_maps(os.path.join(data_dir, 'knowledge_graph.pkl'))
    lipinski = load_lipinski_features(os.path.join(data_dir, 'DB_compounds_lipinski.csv'))
    name_to_id = load_name_to_id_map(os.path.join(data_dir, 'drugbank.db'))
    ddinter_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.lower().startswith('ddinter_code_') and name.lower().endswith('.csv')]
    severity_lookup = load_severity_from_ddinter(ddinter_paths)
    resolve_name, _ = build_stage3_resolver(data_dir=data_dir, kg_name_to_id=kg_name_to_id, kg_id_to_name=kg_id_to_name, snapshot_db_path=os.path.join(data_dir, 'linking_snapshots.sqlite'), config_extra={'stage': 'span_eval', 'resolver_vocab': 'ddinter'})
    train_ddi = DDIDataset(train_sents, tokenizer).samples
    val_ddi = disjoint(train_ddi, DDIDataset(val_sents, tokenizer).samples)
    train_sev = SeverityDataset(train_sents, tokenizer, severity_lookup, resolve_name).samples
    val_sev = disjoint(train_sev, SeverityDataset(val_sents, tokenizer, severity_lookup, resolve_name).samples)
    print(f'disjoint DDI {len(val_ddi)} | disjoint severity {len(val_sev)}')
    results = {'device': device, 'ddi_disjoint': len(val_ddi), 'severity_disjoint': len(val_sev)}
    pairs = [
        ('ddi_current', os.path.join(ckpt_dir, 'stage2_interaction_best.pt'), val_ddi, 'ddi', report_ddi),
        ('ddi_span', os.path.join(ckpt_dir, 'stage2_interaction_span.pt'), val_ddi, 'ddi', report_ddi),
        ('severity_current', os.path.join(ckpt_dir, 'stage3_severity_best.pt'), val_sev, 'severity', report_severity),
        ('severity_span', os.path.join(ckpt_dir, 'stage3_severity_span.pt'), val_sev, 'severity', report_severity),
    ]
    for key, path, samples, head, report_fn in pairs:
        results[key] = score(path, samples, tokenizer, kg_embeddings, name_to_id, lipinski, device, head, report_fn)
    results['switch_interaction'] = bool(results.get('ddi_span') is not None and results.get('ddi_current') is not None and results['ddi_span'] > results['ddi_current'])
    results['switch_severity'] = bool(results.get('severity_span') is not None and results.get('severity_current') is not None and results['severity_span'] > results['severity_current'])
    out = os.path.join(base, 'reports', 'span_disjoint_eval.json')
    with open(out, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
