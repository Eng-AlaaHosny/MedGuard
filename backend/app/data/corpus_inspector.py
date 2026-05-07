# this file checks corpus stats for quick data inspection
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# this class groups logic for DDICorpusInspector
class DDICorpusInspector:

    # this function is used to set up initial values for this object
    def __init__(self):
        self.pair_index: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        self.drug_index: Dict[str, List[Dict]] = defaultdict(list)
        self.built = False
        self._sentence_count = 0

    # this function is used to handle build
    def build(self, corpus_dir: str):
        import xml.etree.ElementTree as ET
        train_dir = os.path.join(corpus_dir, 'Train')
        if not os.path.exists(train_dir):
            print(f'  DDI Corpus not found at {train_dir}')
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
        print(f'DDI Corpus Inspector: {count} sentences, {len(self.pair_index)} drug pairs, {len(self.drug_index)} unique drugs')

    # this function is used to handle index sentence
    def _index_sentence(self, sentence_elem):
        text = sentence_elem.get('text', '').strip()
        if not text:
            return
        entities = {}
        for entity in sentence_elem.findall('entity'):
            eid = entity.get('id', '')
            ename = entity.get('text', '').strip().lower()
            if eid and ename:
                entities[eid] = ename
        entity_names = list(entities.values())
        for name in entity_names:
            self.drug_index[name].append({'text': text, 'entities': entity_names})
        for pair in sentence_elem.findall('pair'):
            e1_id = pair.get('e1', '')
            e2_id = pair.get('e2', '')
            itype = pair.get('type', 'false')
            is_ddi = pair.get('ddi', 'false') == 'true'
            drug_a = entities.get(e1_id, '')
            drug_b = entities.get(e2_id, '')
            if not drug_a or not drug_b:
                continue
            entry = {'text': text, 'type': itype, 'is_ddi': is_ddi, 'drug_a': drug_a, 'drug_b': drug_b}
            self.pair_index[drug_a, drug_b].append(entry)
            self.pair_index[drug_b, drug_a].append(entry)

    # this function is used to find examples
    def find_examples(self, drug_a: str, drug_b: str, n: int=5, ddi_only: bool=False) -> List[Dict]:
        a = drug_a.lower()
        b = drug_b.lower()
        candidates = self.pair_index.get((a, b), [])
        if ddi_only:
            candidates = [c for c in candidates if c['is_ddi']]
        return sorted(candidates, key=lambda x: len(x['text']), reverse=True)[:n]

    # this function is used to handle drug in corpus
    def drug_in_corpus(self, drug_name: str) -> bool:
        return drug_name.lower() in self.drug_index

    # this function is used to handle pair in corpus
    def pair_in_corpus(self, drug_a: str, drug_b: str) -> bool:
        return (drug_a.lower(), drug_b.lower()) in self.pair_index

    # this function is used to handle stats
    def stats(self) -> Dict:
        return {'total_sentences': self._sentence_count, 'total_pairs': len(self.pair_index) // 2, 'total_drugs': len(self.drug_index), 'built': self.built}
