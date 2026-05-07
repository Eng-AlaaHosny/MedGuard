# this file links drug names, routes class mentions, and stores link snapshots
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from app.data.kb_normalization import normalize_kb_text
from app.data.ddinter_vocabulary import rebuild_ddinter_vocabulary
CLASS_MARKERS = ('inhibitor', 'inhibitors', 'blocker', 'blockers', 'agonist', 'agonists', 'antagonist', 'antagonists', 'antibiotic', 'antibiotics', 'antiviral', 'antivirals', 'antifungal', 'antifungals', 'antidepressant', 'antidepressants', 'antipsychotic', 'antipsychotics', 'analgesic', 'analgesics', 'diuretic', 'diuretics', 'corticosteroid', 'corticosteroids', 'statin', 'statins', 'beta blocker', 'beta-blocker', 'ace inhibitor', 'ssri', 'mao inhibitor', 'protease inhibitor')

# this function is used to handle looks like drug class
def looks_like_drug_class(text_norm: str) -> bool:
    s = (text_norm or '').strip().lower()
    if not s:
        return False
    for m in CLASS_MARKERS:
        if m in s:
            return True
    return False

# this function is used to normalize text
def normalize_text(name: str) -> str:
    return normalize_kb_text(name)

# this function is used to handle ensure ddinter vocab table
def ensure_ddinter_vocab_table(drugbank_db_path: str, data_dir: str) -> int:
    if not os.path.exists(drugbank_db_path):
        return 0
    conn = sqlite3.connect(drugbank_db_path)
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ddinter_drug_names'")
        if c.fetchone() is None:
            conn.close()
            return rebuild_ddinter_vocabulary(drugbank_db_path, data_dir)
        c.execute('SELECT COUNT(*) FROM ddinter_drug_names')
        n = int(c.fetchone()[0] or 0)
    finally:
        conn.close()
    if n == 0:
        return rebuild_ddinter_vocabulary(drugbank_db_path, data_dir)
    return n

# this class groups logic for LinkingSnapshotDB
class LinkingSnapshotDB:

    # this function is used to set up initial values for this object
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init()

    # this function is used to handle init
    def _init(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            c = conn.cursor()
            c.execute('\n                CREATE TABLE IF NOT EXISTS linker_runs (\n                  run_id TEXT PRIMARY KEY,\n                  created_at INTEGER,\n                  config_json TEXT\n                )\n                ')
            c.execute('\n                CREATE TABLE IF NOT EXISTS linker_mappings (\n                  run_id TEXT,\n                  raw TEXT,\n                  normalized TEXT,\n                  entity_type TEXT,\n                  resolved TEXT,\n                  method TEXT,\n                  created_at INTEGER\n                )\n                ')
            c.execute('CREATE INDEX IF NOT EXISTS idx_linker_mappings_run ON linker_mappings(run_id)')
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    # this function is used to handle make run id
    def make_run_id(config: dict) -> str:
        payload = json.dumps(config, sort_keys=True, ensure_ascii=True).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()[:16]

    # this function is used to handle ensure run
    def ensure_run(self, run_id: str, config: dict) -> None:
        conn = sqlite3.connect(self.path)
        try:
            c = conn.cursor()
            c.execute('SELECT run_id FROM linker_runs WHERE run_id = ?', (run_id,))
            if c.fetchone() is None:
                c.execute('INSERT INTO linker_runs(run_id, created_at, config_json) VALUES (?, ?, ?)', (run_id, int(time.time()), json.dumps(config, sort_keys=True)))
                conn.commit()
        finally:
            conn.close()

    # this function is used to handle record
    def record(self, run_id: str, raw: str, normalized: str, entity_type: str, resolved: str, method: str) -> None:
        conn = sqlite3.connect(self.path)
        try:
            c = conn.cursor()
            c.execute('\n                INSERT INTO linker_mappings(run_id, raw, normalized, entity_type, resolved, method, created_at)\n                VALUES (?, ?, ?, ?, ?, ?, ?)\n                ', (run_id, raw, normalized, entity_type, resolved, method, int(time.time())))
            conn.commit()
        finally:
            conn.close()

# this class groups logic for BufferedSnapshotWriter
class BufferedSnapshotWriter:

    # this function is used to set up initial values for this object
    def __init__(self, snapshot_db: LinkingSnapshotDB, run_id: str, flush_every: int=500):
        self.snapshot_db = snapshot_db
        self.run_id = run_id
        self.flush_every = int(flush_every)
        self._buf = []

    # this function is used to handle record
    def record(self, raw: str, normalized: str, entity_type: str, resolved: str, method: str) -> None:
        self._buf.append((self.run_id, raw, normalized, entity_type, resolved, method, int(time.time())))
        if len(self._buf) >= self.flush_every:
            self.flush()

    # this function is used to handle flush
    def flush(self) -> None:
        if not self._buf:
            return
        conn = sqlite3.connect(self.snapshot_db.path)
        try:
            c = conn.cursor()
            c.executemany('\n                INSERT INTO linker_mappings(run_id, raw, normalized, entity_type, resolved, method, created_at)\n                VALUES (?, ?, ?, ?, ?, ?, ?)\n                ', self._buf)
            conn.commit()
        finally:
            conn.close()
        self._buf.clear()

@dataclass(frozen=True)
# this class groups logic for LinkResult
class LinkResult:
    entity_type: str
    resolved: str
    method: str

# this class groups logic for DictionaryFirstLinker
class DictionaryFirstLinker:

    # this function is used to set up initial values for this object
    def __init__(self, drugbank_db_path: Optional[str], kg_name_to_id: Optional[Dict[str, str]]=None, kg_id_to_name: Optional[Dict[str, str]]=None):
        self.drugbank_db_path = drugbank_db_path
        self.kg_name_to_id = kg_name_to_id or {}
        self.kg_id_to_name = kg_id_to_name or {}
        self.ddinter: Set[str] = set()
        self.synonym_norm_to_drug: Dict[str, List[str]] = {}
        self.drug_id_to_canon: Dict[str, str] = {}
        self.drug_id_to_syns: Dict[str, Set[str]] = {}
        self.drugbank_names: Set[str] = set()
        self._load_drugbank_linking_tables() if drugbank_db_path else None

    # this function is used to handle load drugbank linking tables
    def _load_drugbank_linking_tables(self) -> None:
        if not self.drugbank_db_path or not os.path.exists(self.drugbank_db_path):
            return
        try:
            conn = sqlite3.connect(self.drugbank_db_path)
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ddinter_drug_names'")
            if c.fetchone() is not None:
                c.execute('SELECT norm_name FROM ddinter_drug_names')
                self.ddinter = {r[0] for r in c.fetchall() if r[0]}
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='drugbank_synonyms'")
            if c.fetchone() is not None:
                c.execute('SELECT drug_id, norm_text FROM drugbank_synonyms')
                for did, nt in c.fetchall():
                    if not did or not nt:
                        continue
                    self.synonym_norm_to_drug.setdefault(nt, []).append(did)
                for k in self.synonym_norm_to_drug:
                    self.synonym_norm_to_drug[k] = sorted(set(self.synonym_norm_to_drug[k]))
                c.execute('SELECT drug_id, norm_text FROM drugbank_synonyms')
                rows = c.fetchall()
                tmp: Dict[str, Set[str]] = {}
                for did, nt in rows:
                    tmp.setdefault(did, set()).add(nt)
                self.drug_id_to_syns = tmp
            c.execute('SELECT id, name FROM drugs')
            for did, name in c.fetchall():
                cn = normalize_kb_text(name or '')
                if did and cn:
                    self.drug_id_to_canon[str(did)] = cn
                    self.drugbank_names.add(cn)
            conn.close()
        except Exception:
            return

    # this function is used to handle pick drug id
    def _pick_drug_id(self, norm: str) -> Optional[str]:
        cands = self.synonym_norm_to_drug.get(norm) or []
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        return sorted(cands)[0]

    # this function is used to handle best ddinter name for drug
    def _best_ddinter_name_for_drug(self, drug_id: str) -> Optional[str]:
        canon = self.drug_id_to_canon.get(drug_id, '')
        if canon and canon in self.ddinter:
            return canon
        syns = sorted(self.drug_id_to_syns.get(drug_id, set()))
        for s in syns:
            if s in self.ddinter:
                return s
        return None

    # this function is used to handle link
    def link(self, raw: str) -> LinkResult:
        norm = normalize_kb_text(raw)
        if looks_like_drug_class(norm):
            return LinkResult(entity_type='class', resolved=norm, method='class_routing')
        if norm in self.ddinter:
            return LinkResult(entity_type='drug', resolved=norm, method='ddinter_surface')
        did = self._pick_drug_id(norm)
        if did:
            anchored = self._best_ddinter_name_for_drug(did)
            if anchored is not None:
                method = 'ddinter_from_drugbank_primary' if anchored == self.drug_id_to_canon.get(did) else 'ddinter_from_drugbank_synonym'
                return LinkResult(entity_type='drug', resolved=anchored, method=method)
        kdid = self.kg_name_to_id.get(norm)
        if kdid:
            canon = self.kg_id_to_name.get(kdid)
            if canon:
                canon_norm = normalize_kb_text(canon)
                if canon_norm and canon_norm in self.ddinter:
                    return LinkResult(entity_type='drug', resolved=canon_norm, method='ddinter_via_kg')
                if canon_norm:
                    did2 = self._pick_drug_id(canon_norm)
                    if did2:
                        a2 = self._best_ddinter_name_for_drug(did2)
                        if a2 is not None:
                            return LinkResult(entity_type='drug', resolved=a2, method='ddinter_via_kg_drugbank')
        if norm in self.drugbank_names:
            return LinkResult(entity_type='drug', resolved=norm, method='drugbank_only')
        return LinkResult(entity_type='drug', resolved=norm, method='passthrough')

# this function is used to build stage3 resolver
def build_stage3_resolver(data_dir: str, kg_name_to_id: Dict[str, str], kg_id_to_name: Dict[str, str], snapshot_db_path: str, config_extra: Optional[dict]=None) -> Tuple[callable, dict]:
    drugbank_db = os.path.join(data_dir, 'drugbank.db')
    try:
        n_dd = ensure_ddinter_vocab_table(drugbank_db, data_dir)
    except Exception:
        n_dd = 0
    linker = DictionaryFirstLinker(drugbank_db_path=drugbank_db, kg_name_to_id=kg_name_to_id, kg_id_to_name=kg_id_to_name)
    snap = LinkingSnapshotDB(snapshot_db_path)
    config = {'type': 'drugbank_synonyms+ddinter_vocab', 'drugbank_db': bool(os.path.exists(drugbank_db)), 'ddinter_vocab_n': n_dd, 'class_markers': sorted(list(CLASS_MARKERS))}
    if config_extra:
        config.update(config_extra)
    run_id = snap.make_run_id(config)
    snap.ensure_run(run_id, config)
    writer = BufferedSnapshotWriter(snap, run_id=run_id, flush_every=int(os.environ.get('MEDGUARD_LINKER_FLUSH', '500')))
    stats = {'run_id': run_id, 'class_routed': 0, 'ddinter_surface': 0, 'ddinter_from_drugbank_primary': 0, 'ddinter_from_drugbank_synonym': 0, 'ddinter_via_kg': 0, 'ddinter_via_kg_drugbank': 0, 'drugbank_only': 0, 'passthrough': 0}

    # this function is used to handle resolve
    def resolve(raw: str) -> str:
        r = linker.link(raw or '')
        if r.entity_type == 'class':
            stats['class_routed'] += 1
        elif r.method in stats:
            stats[r.method] += 1
        else:
            stats['passthrough'] += 1
        writer.record(raw=raw or '', normalized=normalize_text(raw or ''), entity_type=r.entity_type, resolved=r.resolved, method=r.method)
        return r.resolved
    stats['_flush'] = writer.flush
    return (resolve, stats)
