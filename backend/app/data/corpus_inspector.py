"""
DDI Corpus Inspector (Analysis Tool — NOT used at inference)
=============================================================
This module is a RESEARCH UTILITY only.

Purpose:
  - Explore the DDI Corpus training data.
  - Find examples of how specific drug pairs appear in the corpus.
  - Support error analysis and qualitative evaluation.

It is NOT used in the inference pipeline (routes.py).
Substituting training sentences at inference time would mean the model
never actually processes the user's query — this was removed in the
academic cleanup (see routes.py). Domain mismatch between user queries
and training sentences is acknowledged as a limitation in the paper.

Usage (analysis only):
    inspector = DDICorpusInspector()
    inspector.build(corpus_dir)
    examples = inspector.find_examples("warfarin", "aspirin", n=5)
    stats    = inspector.stats()
"""

import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class DDICorpusInspector:
    """
    Indexes DDI Corpus sentences for exploratory analysis.
    Useful for:
      - Counting how often a drug pair appears in training data
      - Retrieving example sentences for qualitative evaluation
      - Computing corpus coverage statistics
    """

    def __init__(self):
        self.pair_index: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        self.drug_index: Dict[str, List[Dict]]             = defaultdict(list)
        self.built      = False
        self._sentence_count = 0

    def build(self, corpus_dir: str):
        """Build index from DDI Corpus XML files."""
        import xml.etree.ElementTree as ET

        train_dir = os.path.join(corpus_dir, 'Train')
        if not os.path.exists(train_dir):
            print(f"  DDI Corpus not found at {train_dir}")
            return

        count = 0
        for folder in os.listdir(train_dir):
            folder_path = os.path.join(train_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if not fname.endswith('.xml'):
                    continue
                try:
                    tree = ET.parse(os.path.join(folder_path, fname))
                    for sentence in tree.getroot().iter('sentence'):
                        self._index_sentence(sentence)
                        count += 1
                except Exception:
                    continue

        self._sentence_count = count
        self.built = True
        print(f"DDI Corpus Inspector: {count} sentences, "
              f"{len(self.pair_index)} drug pairs, "
              f"{len(self.drug_index)} unique drugs")

    def _index_sentence(self, sentence_elem):
        text = sentence_elem.get('text', '').strip()
        if not text:
            return

        entities = {}
        for entity in sentence_elem.findall('entity'):
            eid   = entity.get('id', '')
            ename = entity.get('text', '').strip().lower()
            if eid and ename:
                entities[eid] = ename

        entity_names = list(entities.values())

        for name in entity_names:
            self.drug_index[name].append({'text': text, 'entities': entity_names})

        for pair in sentence_elem.findall('pair'):
            e1_id  = pair.get('e1', '')
            e2_id  = pair.get('e2', '')
            itype  = pair.get('type', 'false')
            is_ddi = pair.get('ddi', 'false') == 'true'
            drug_a = entities.get(e1_id, '')
            drug_b = entities.get(e2_id, '')

            if not drug_a or not drug_b:
                continue

            entry = {
                'text':   text,
                'type':   itype,
                'is_ddi': is_ddi,
                'drug_a': drug_a,
                'drug_b': drug_b,
            }
            self.pair_index[(drug_a, drug_b)].append(entry)
            self.pair_index[(drug_b, drug_a)].append(entry)

    def find_examples(
        self,
        drug_a:    str,
        drug_b:    str,
        n:         int = 5,
        ddi_only:  bool = False,
    ) -> List[Dict]:
        """
        Find example sentences from the corpus for a drug pair.
        For analysis only — not used during inference.
        """
        a = drug_a.lower()
        b = drug_b.lower()
        candidates = self.pair_index.get((a, b), [])
        if ddi_only:
            candidates = [c for c in candidates if c['is_ddi']]
        return sorted(candidates, key=lambda x: len(x['text']), reverse=True)[:n]

    def drug_in_corpus(self, drug_name: str) -> bool:
        return drug_name.lower() in self.drug_index

    def pair_in_corpus(self, drug_a: str, drug_b: str) -> bool:
        return (drug_a.lower(), drug_b.lower()) in self.pair_index

    def stats(self) -> Dict:
        return {
            'total_sentences': self._sentence_count,
            'total_pairs':     len(self.pair_index) // 2,
            'total_drugs':     len(self.drug_index),
            'built':           self.built,
        }
