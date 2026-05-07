# this file reads ddi corpus files and builds clean training samples
import os
import torch
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
# this class groups logic for DrugEntity
class DrugEntity:
    id: str
    text: str
    start: int
    end: int

@dataclass
# this class groups logic for DDISentence
class DDISentence:
    id: str
    text: str
    entities: List[DrugEntity]
    interactions: List[Dict]

# this function is used to parse char offset
def parse_char_offset(offset_str: str):
    first_span = offset_str.split(';')[0]
    parts = first_span.split('-')
    start = int(parts[0])
    end = int(parts[1])
    return (start, end)

# this function is used to parse ddi xml
def parse_ddi_xml(file_path: str) -> List[DDISentence]:
    sentences = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for sentence in root.iter('sentence'):
        sent_id = sentence.get('id', '')
        sent_text = sentence.get('text', '')
        entities = []
        for entity in sentence.findall('entity'):
            offset = entity.get('charOffset', '0-0')
            start, end = parse_char_offset(offset)
            e = DrugEntity(id=entity.get('id', ''), text=entity.get('text', ''), start=start, end=end)
            entities.append(e)
        interactions = []
        for pair in sentence.findall('pair'):
            interaction = {'id': pair.get('id', ''), 'e1': pair.get('e1', ''), 'e2': pair.get('e2', ''), 'ddi': pair.get('ddi', 'false') == 'true', 'type': pair.get('type', 'false')}
            interactions.append(interaction)
        sentences.append(DDISentence(id=sent_id, text=sent_text, entities=entities, interactions=interactions))
    return sentences

# this function is used to get ner labels
def get_ner_labels(sentence: DDISentence, tokenizer, max_length: int=128):
    encoding = tokenizer(sentence.text, max_length=max_length, truncation=True, padding='max_length', return_offsets_mapping=True, return_tensors='pt')
    offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()
    labels = [0] * max_length
    for entity in sentence.entities:
        entity_start = entity.start
        entity_end = entity.end
        first_token = True
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == 0 and token_end == 0:
                continue
            if token_start >= entity_start and token_end <= entity_end + 1:
                if first_token:
                    labels[idx] = 1
                    first_token = False
                else:
                    labels[idx] = 2
    return (encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), torch.tensor(labels, dtype=torch.long))

# this function is used to load ddi corpus
def load_ddi_corpus(data_dir: str) -> Tuple[List[DDISentence], List[DDISentence]]:
    train_sentences = []
    test_sentences = []
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test', 'Test for DDI Extraction task')
    if os.path.exists(train_dir):
        for folder in os.listdir(train_dir):
            folder_path = os.path.join(train_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.xml'):
                        filepath = os.path.join(folder_path, file)
                        train_sentences.extend(parse_ddi_xml(filepath))
    if os.path.exists(test_dir):
        for folder in os.listdir(test_dir):
            folder_path = os.path.join(test_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.xml'):
                        filepath = os.path.join(folder_path, file)
                        test_sentences.extend(parse_ddi_xml(filepath))
    print(f'Loaded {len(train_sentences)} train sentences')
    print(f'Loaded {len(test_sentences)} test sentences')
    return (train_sentences, test_sentences)
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(base_dir, 'DDICorpus')
    train, test = load_ddi_corpus(corpus_dir)
    print(f'Sample sentence: {(train[0].text if train else 'No data found')}')
    print(f'Sample entities: {(train[0].entities if train else 'No data found')}')
