# Dynamic INT8 of the one-trunk runtime. Embeddings, LayerNorm, and attention matmuls stay FP32.
import os
import time

import torch
from sklearn.model_selection import train_test_split
from torch.ao.quantization import quantize_dynamic
from transformers import AutoTokenizer

from app.data.preprocessor import load_ddi_corpus
from app.models.medguard_model import MedGuardModel
from app.models.serving_baseline import (
    DDI_NAMES,
    MODEL_NAME,
    SEV_NAMES,
    bootstrap_macro,
    disk_mb,
    dump_results,
    macro_and_report,
    predict_pairs,
    rss_mb,
)
from app.models.trainer import (
    DDIDataset,
    SeverityDataset,
    build_stage3_resolver,
    load_kg_embeddings,
    load_kg_name_maps,
    load_lipinski_features,
    load_name_to_id_map,
    load_severity_from_ddinter,
)


def load_fp32(path):
    model = MedGuardModel(model_name=MODEL_NAME).cpu()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')
    ckpt_dir = os.path.join(base, 'models', 'checkpoints')
    fp32_path = os.path.join(ckpt_dir, 'stage3_severity_frozen_trunk.pt')
    int8_path = os.path.join(ckpt_dir, 'stage3_severity_frozen_trunk_int8.pt')
    out_path = os.path.join(base, 'reports', 'int8_baseline.json')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_all, _ = load_ddi_corpus(os.path.join(data_dir, 'DDICorpus'))
    _, val_sents = train_test_split(train_all, test_size=0.15, random_state=42)
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
        config_extra={'stage': 'int8', 'resolver_vocab': 'ddinter'},
    )
    val_ddi = DDIDataset(val_sents, tokenizer).samples
    val_sev = SeverityDataset(val_sents, tokenizer, severity_lookup, resolve_name).samples
    print(f'val ddi={len(val_ddi)} sev={len(val_sev)}', flush=True)

    fp32 = load_fp32(fp32_path)
    int8 = quantize_dynamic(load_fp32(fp32_path), {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(int8.state_dict(), int8_path)

    results = {
        'method': 'dynamic INT8 post-training quantization on nn.Linear',
        'left_in_fp32': 'embeddings, LayerNorm, attention matrix multiplies',
        'disk_mb': {'fp32': disk_mb(fp32_path), 'int8': disk_mb(int8_path)},
        'note': 'Scored on the same 15% sentence split as Step 1. Source weights are the frozen-trunk checkpoint.',
    }
    dump_results(results, out_path)

    device = 'cpu'
    for pooling in ('cls', 'span'):
        print(f'\nINT8 DDI pooling={pooling}', flush=True)
        true, pred, cov = predict_pairs(
            int8, val_ddi, tokenizer, kg_embeddings, name_to_id, lipinski, device, pooling, 'ddi'
        )
        macro, detail = macro_and_report(true, pred, DDI_NAMES)
        results[f'ddi_int8_{pooling}'] = {
            'macro_f1': round(macro, 4),
            'per_class': detail,
            'bootstrap': bootstrap_macro(true, pred, 5),
            'coverage': cov,
        }
        print('bootstrap', results[f'ddi_int8_{pooling}']['bootstrap'], flush=True)
        dump_results(results, out_path)

        print(f'\nINT8 severity pooling={pooling}', flush=True)
        true, pred, cov = predict_pairs(
            int8, val_sev, tokenizer, kg_embeddings, name_to_id, lipinski, device, pooling, 'severity'
        )
        macro, detail = macro_and_report(true, pred, SEV_NAMES)
        results[f'severity_int8_{pooling}'] = {
            'macro_f1': round(macro, 4),
            'per_class': detail,
            'bootstrap': bootstrap_macro(true, pred, 4),
            'coverage': cov,
        }
        print('bootstrap', results[f'severity_int8_{pooling}']['bootstrap'], flush=True)
        dump_results(results, out_path)

    sample = val_ddi[:30]
    def time_model(model):
        t0 = time.perf_counter()
        predict_pairs(
            model, sample, tokenizer, kg_embeddings, name_to_id, lipinski, device,
            'span', 'ddi', batch_size=1, collect_labels=False,
        )
        return time.perf_counter() - t0

    fp_s = time_model(fp32)
    q_s = time_model(int8)
    results['latency_span_batch1'] = {
        'n': len(sample),
        'fp32_seconds_per_pair': round(fp_s / len(sample), 3),
        'int8_seconds_per_pair': round(q_s / len(sample), 3),
    }
    results['ram'] = rss_mb()
    print('latency', results['latency_span_batch1'], flush=True)
    dump_results(results, out_path)


if __name__ == '__main__':
    main()
