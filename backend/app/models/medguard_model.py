# this file defines the model architecture and stage freezing behavior
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional
DDI_LABELS = {0: 'false', 1: 'mechanism', 2: 'effect', 3: 'advise', 4: 'int'}
NER_LABELS = {0: 'O', 1: 'B-DRUG', 2: 'I-DRUG'}
SEVERITY_LABELS = {0: 'safe', 1: 'caution', 2: 'warning', 3: 'danger'}
SEVERITY_COLORS = {'safe': '#28a745', 'caution': '#ffc107', 'warning': '#fd7e14', 'danger': '#dc3545'}
STAGE_NER = 'ner'
STAGE_INTERACTION = 'interaction'
STAGE_SEVERITY = 'severity'
KG_DIM = 128
LIPINSKI_DIM = 5

# this class groups logic for MedGuardModel
class MedGuardModel(nn.Module):

    # this function is used to set up initial values for this object
    def __init__(self, model_name: str='emilyalsentzer/Bio_ClinicalBERT', num_ner_labels: int=3, num_ddi_labels: int=5, num_severity_labels: int=4, kg_dim: int=KG_DIM, lipinski_dim: int=LIPINSKI_DIM, dropout: float=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        fusion_input_dim = self.hidden_size + kg_dim + lipinski_dim
        self.ner_head = nn.Linear(self.hidden_size, num_ner_labels)
        self.drug_fusion = nn.Sequential(nn.Linear(fusion_input_dim, self.hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.pair_projection = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.interaction_head = nn.Linear(self.hidden_size, num_ddi_labels)
        self.severity_head = nn.Linear(self.hidden_size, num_severity_labels)

    # this function is used to handle encode
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_repr = outputs.last_hidden_state
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return (token_repr, cls_repr)

    # this function is used to get entity representation
    def get_entity_representation(self, token_repr: torch.Tensor, entity_spans: Optional[List]=None) -> torch.Tensor:
        if entity_spans is None or len(entity_spans) == 0:
            return token_repr[:, 0, :]
        start, end = entity_spans[0]
        span_repr = token_repr[:, start:end + 1, :]
        return span_repr.mean(dim=1)

    # this function is used to handle fuse drug features
    def fuse_drug_features(self, bert_repr: torch.Tensor, kg_embedding: Optional[torch.Tensor]=None, lipinski_feats: Optional[torch.Tensor]=None) -> torch.Tensor:
        B = bert_repr.size(0)
        device = bert_repr.device
        if kg_embedding is None:
            kg_embedding = torch.zeros(B, KG_DIM, device=device)
        if lipinski_feats is None:
            lipinski_feats = torch.zeros(B, LIPINSKI_DIM, device=device)
        combined = torch.cat([bert_repr, kg_embedding, lipinski_feats], dim=-1)
        return self.drug_fusion(combined)

    # this function is used to handle forward
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, drug_a_spans: Optional[List]=None, drug_b_spans: Optional[List]=None, kg_embedding_a: Optional[torch.Tensor]=None, kg_embedding_b: Optional[torch.Tensor]=None, lipinski_feats_a: Optional[torch.Tensor]=None, lipinski_feats_b: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        token_repr, cls_repr = self.encode(input_ids, attention_mask)
        token_repr = self.dropout(token_repr)
        cls_repr = self.dropout(cls_repr)
        ner_logits = self.ner_head(token_repr)
        h_a_bert = self.get_entity_representation(token_repr, drug_a_spans)
        h_b_bert = self.get_entity_representation(token_repr, drug_b_spans)
        h_a = self.fuse_drug_features(h_a_bert, kg_embedding_a, lipinski_feats_a)
        h_b = self.fuse_drug_features(h_b_bert, kg_embedding_b, lipinski_feats_b)
        pair_repr = torch.cat([h_a, h_b, cls_repr], dim=-1)
        pair_repr = self.pair_projection(pair_repr)
        interaction_logits = self.interaction_head(pair_repr)
        severity_logits = self.severity_head(pair_repr)
        return {'ner_logits': ner_logits, 'interaction_logits': interaction_logits, 'severity_logits': severity_logits}

    # this function is used to handle set stage
    def set_stage(self, stage: str):
        if stage == STAGE_NER:
            for p in self.parameters():
                p.requires_grad = True
        elif stage == STAGE_INTERACTION:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.ner_head.parameters():
                p.requires_grad = False
            for p in self.drug_fusion.parameters():
                p.requires_grad = True
            for p in self.pair_projection.parameters():
                p.requires_grad = True
            for p in self.interaction_head.parameters():
                p.requires_grad = True
            for p in self.severity_head.parameters():
                p.requires_grad = True
        elif stage == STAGE_SEVERITY:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.ner_head.parameters():
                p.requires_grad = False
            for p in self.interaction_head.parameters():
                p.requires_grad = False
            for p in self.drug_fusion.parameters():
                p.requires_grad = True
            for p in self.pair_projection.parameters():
                p.requires_grad = True
            for p in self.severity_head.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown stage: {stage}. Use '{STAGE_NER}', '{STAGE_INTERACTION}', or '{STAGE_SEVERITY}'.")
        trainable = sum((p.numel() for p in self.parameters() if p.requires_grad))
        total = sum((p.numel() for p in self.parameters()))
        print(f"  Stage '{stage}': {trainable:,} / {total:,} parameters trainable")

# this function is used to load tokenizer
def load_tokenizer(model_name: str='emilyalsentzer/Bio_ClinicalBERT'):
    return AutoTokenizer.from_pretrained(model_name)

# this function is used to load model
def load_model(model_name: str='emilyalsentzer/Bio_ClinicalBERT') -> MedGuardModel:
    return MedGuardModel(model_name=model_name)
if __name__ == '__main__':
    print('Loading Bio_ClinicalBERT tokenizer and model...')
    tokenizer = load_tokenizer()
    model = load_model()
    model.eval()
    test_sentence = 'Warfarin and aspirin interaction may increase bleeding risk.'
    inputs = tokenizer(test_sentence, return_tensors='pt', max_length=128, truncation=True, padding=True)
    lipinski_a = torch.tensor([[400.0, 4, 2, 2.5, 1.0]])
    lipinski_b = torch.tensor([[180.0, 3, 1, 1.2, 1.0]])
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], lipinski_feats_a=lipinski_a, lipinski_feats_b=lipinski_b)
    print('\n✅ Model loaded successfully!')
    print(f'  NER logits shape:         {outputs['ner_logits'].shape}')
    print(f'  Interaction logits shape: {outputs['interaction_logits'].shape}')
    print(f'  Severity logits shape:    {outputs['severity_logits'].shape}')
    print(f'\n  NER classes:      {list(NER_LABELS.values())}')
    print(f'  DDI classes:      {list(DDI_LABELS.values())}')
    print(f'  Severity classes: {list(SEVERITY_LABELS.values())}')
    print('\n── Stage freeze test ──')
    model.set_stage(STAGE_NER)
    model.set_stage(STAGE_INTERACTION)
    model.set_stage(STAGE_SEVERITY)
    print('\nAll stages OK — model architecture correct!')
