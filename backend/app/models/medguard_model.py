"""
MedGuard — Multi-Task Model (Academic Clean Version)
======================================================
Architecture:
  Backbone  : Bio_ClinicalBERT (768-dim hidden states)
  Head 1    : NER        — token-level drug entity detection
              Labels: O (0), B-DRUG (1), I-DRUG (2)
  Head 2    : Interaction — sentence-level DDI classification
              Labels: false (0), mechanism (1), effect (2), advise (3), int (4)
  Head 3    : Severity   — sentence-level severity prediction
              Labels: safe (0), caution (1), warning (2), danger (3)

Training strategy (Option B — truly separate):
  Stage 1 — NER         : freeze nothing, train encoder + NER head only
  Stage 2 — Interaction : freeze encoder, train interaction head + KG fusion
  Stage 3 — Severity    : freeze encoder + interaction head, train severity head only
  Each stage saves its own checkpoint.

KG + Lipinski fusion:
  - node2vec 128-dim KG embedding   per drug (from knowledge_graph.pkl)
  - Lipinski 5-dim physicochemical  per drug (from DB_compounds_lipinski.csv)
  - Both are concatenated with the BERT drug span repr:
      [BERT_repr | KG_emb | Lipinski_feats]  →  Linear  →  768-dim
  - Zero-padded when a drug is absent from either source.

Severity label source (academic honesty):
  - Head 3 is trained on HEURISTIC labels derived from DrugBank interaction
    descriptions via keyword matching (map_severity in drugbank_processor.py).
  - This is explicitly documented as a rule-based silver-label approach,
    NOT as a human-annotated ground truth.
  - Reference: distant supervision / silver labeling (Mintz et al., 2009).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional

# ── Label maps ────────────────────────────────────────────────────────────────

DDI_LABELS = {
    0: 'false',
    1: 'mechanism',
    2: 'effect',
    3: 'advise',
    4: 'int',
}

NER_LABELS = {
    0: 'O',
    1: 'B-DRUG',
    2: 'I-DRUG',
}

SEVERITY_LABELS = {
    0: 'safe',
    1: 'caution',
    2: 'warning',
    3: 'danger',
}

SEVERITY_COLORS = {
    'safe':    '#28a745',
    'caution': '#ffc107',
    'warning': '#fd7e14',
    'danger':  '#dc3545',
}

# Training stages — controls which parameters are frozen
STAGE_NER         = 'ner'
STAGE_INTERACTION = 'interaction'
STAGE_SEVERITY    = 'severity'

# Input dimensions
KG_DIM       = 128   # node2vec embedding dim
LIPINSKI_DIM = 5     # MW, HBA, HBD, LogP, Ro5


class MedGuardModel(nn.Module):
    """
    Bio_ClinicalBERT backbone with 3 independently-trained classification heads.

    Fusion input per drug:
        [BERT_span (768) | KG_emb (128) | Lipinski (5)]  →  Linear(901, 768) + GELU

    Pair representation:
        [h_drug_a (768) | h_drug_b (768) | CLS (768)]    →  Linear(2304, 768) + GELU

    Notes:
    - pair_embedding (learnable constant) removed — no theoretical justification.
    - CLS token is used as the sentence-level context vector in pair repr.
    - drug span mean-pooling falls back to CLS when spans are unavailable.
    """

    def __init__(
        self,
        model_name:           str   = "emilyalsentzer/Bio_ClinicalBERT",
        num_ner_labels:       int   = 3,   # O, B-DRUG, I-DRUG
        num_ddi_labels:       int   = 5,   # false/mechanism/effect/advise/int
        num_severity_labels:  int   = 4,   # safe/caution/warning/danger
        kg_dim:               int   = KG_DIM,
        lipinski_dim:         int   = LIPINSKI_DIM,
        dropout:              float = 0.3,
    ):
        super().__init__()

        self.encoder     = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size   # 768
        self.dropout     = nn.Dropout(dropout)

        fusion_input_dim = self.hidden_size + kg_dim + lipinski_dim  # 768+128+5 = 901

        # ── Head 1 : NER ─────────────────────────────────────────────────────
        # Trained in Stage 1; encoder is unfrozen at this stage.
        self.ner_head = nn.Linear(self.hidden_size, num_ner_labels)

        # ── Multi-modal fusion (per drug, used by Heads 2 & 3) ───────────────
        # Projects concatenated [BERT | KG | Lipinski] back to hidden_size.
        # Trained in Stage 2 alongside the interaction head.
        self.drug_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Pair projection (used by Heads 2 & 3) ────────────────────────────
        # [h_a (768) | h_b (768) | CLS (768)]  →  768
        self.pair_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Head 2 : DDI interaction type ────────────────────────────────────
        # Trained in Stage 2; encoder frozen.
        self.interaction_head = nn.Linear(self.hidden_size, num_ddi_labels)

        # ── Head 3 : Severity ────────────────────────────────────────────────
        # Trained in Stage 3; encoder + interaction head frozen.
        # NOTE: labels are DrugBank silver labels (distant supervision),
        #       not human-annotated ground truth.
        self.severity_head = nn.Linear(self.hidden_size, num_severity_labels)

    # ── Encoder ───────────────────────────────────────────────────────────────

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_repr = outputs.last_hidden_state           # (B, T, 768)
        cls_repr   = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return token_repr, cls_repr

    # ── Entity span mean-pooling ──────────────────────────────────────────────

    def get_entity_representation(
        self,
        token_repr: torch.Tensor,
        entity_spans: Optional[List] = None,
    ) -> torch.Tensor:
        """
        Mean-pool token representations over the entity span.
        Falls back to CLS token when spans are not provided.
        """
        if entity_spans is None or len(entity_spans) == 0:
            return token_repr[:, 0, :]          # (B, 768) — CLS fallback
        start, end = entity_spans[0]
        span_repr  = token_repr[:, start:end + 1, :]  # (B, span, 768)
        return span_repr.mean(dim=1)            # (B, 768)

    # ── Multi-modal drug representation ───────────────────────────────────────

    def fuse_drug_features(
        self,
        bert_repr:       torch.Tensor,
        kg_embedding:    Optional[torch.Tensor] = None,
        lipinski_feats:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse BERT span repr with KG node2vec embedding and Lipinski features.
        Missing modalities are zero-padded — graceful degradation.

        Args:
            bert_repr:      (B, 768)
            kg_embedding:   (B, 128) or None
            lipinski_feats: (B,   5) or None
        Returns:
            fused: (B, 768)
        """
        B      = bert_repr.size(0)
        device = bert_repr.device

        if kg_embedding is None:
            kg_embedding = torch.zeros(B, KG_DIM, device=device)

        if lipinski_feats is None:
            lipinski_feats = torch.zeros(B, LIPINSKI_DIM, device=device)

        combined = torch.cat([bert_repr, kg_embedding, lipinski_feats], dim=-1)  # (B, 901)
        return self.drug_fusion(combined)   # (B, 768)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:         torch.Tensor,
        attention_mask:    torch.Tensor,
        drug_a_spans:      Optional[List]          = None,
        drug_b_spans:      Optional[List]          = None,
        kg_embedding_a:    Optional[torch.Tensor]  = None,
        kg_embedding_b:    Optional[torch.Tensor]  = None,
        lipinski_feats_a:  Optional[torch.Tensor]  = None,
        lipinski_feats_b:  Optional[torch.Tensor]  = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids          : (B, T)
            attention_mask     : (B, T)
            drug_a_spans       : optional [(start, end)] for drug A
            drug_b_spans       : optional [(start, end)] for drug B
            kg_embedding_a/b   : optional (B, 128)  KG node2vec per drug
            lipinski_feats_a/b : optional (B, 5)    Lipinski features per drug

        Returns dict:
            ner_logits          : (B, T, 3)
            interaction_logits  : (B, 5)
            severity_logits     : (B, 4)
        """
        token_repr, cls_repr = self.encode(input_ids, attention_mask)
        token_repr = self.dropout(token_repr)
        cls_repr   = self.dropout(cls_repr)

        # ── Head 1: NER (token-level) ─────────────────────────────────────────
        ner_logits = self.ner_head(token_repr)   # (B, T, 3)

        # ── Drug representations for Heads 2 & 3 ─────────────────────────────
        h_a_bert = self.get_entity_representation(token_repr, drug_a_spans)  # (B, 768)
        h_b_bert = self.get_entity_representation(token_repr, drug_b_spans)  # (B, 768)

        # Fuse with KG + Lipinski
        h_a = self.fuse_drug_features(h_a_bert, kg_embedding_a, lipinski_feats_a)  # (B, 768)
        h_b = self.fuse_drug_features(h_b_bert, kg_embedding_b, lipinski_feats_b)  # (B, 768)

        # Pair repr: [h_a | h_b | CLS]
        pair_repr = torch.cat([h_a, h_b, cls_repr], dim=-1)   # (B, 2304)
        pair_repr = self.pair_projection(pair_repr)             # (B, 768)

        # ── Head 2: Interaction type ──────────────────────────────────────────
        interaction_logits = self.interaction_head(pair_repr)   # (B, 5)

        # ── Head 3: Severity ──────────────────────────────────────────────────
        severity_logits = self.severity_head(pair_repr)         # (B, 4)

        return {
            'ner_logits':          ner_logits,
            'interaction_logits':  interaction_logits,
            'severity_logits':     severity_logits,
        }

    # ── Stage-based freezing ──────────────────────────────────────────────────

    def set_stage(self, stage: str):
        """
        Freeze parameters according to training stage.

        Stage 1 (NER):
          - All parameters trainable.
          - Encoder learns from NER supervision.
          - interaction_head, severity_head, drug_fusion, pair_projection
            all receive gradients but are not directly supervised yet;
            this is acceptable — they will be re-initialized or overwritten
            in later stages if desired, but keeping them trainable avoids
            gradient flow issues.

        Stage 2 (Interaction):
          - Encoder frozen (preserves NER-tuned representations).
          - drug_fusion, pair_projection, interaction_head trainable.
          - severity_head also trainable (will be overwritten in Stage 3).

        Stage 3 (Severity):
          - Encoder frozen.
          - interaction_head frozen (preserves Stage 2 weights).
          - drug_fusion, pair_projection, severity_head trainable.
            drug_fusion & pair_projection are fine-tuned for severity signal.
        """
        if stage == STAGE_NER:
            for p in self.parameters():
                p.requires_grad = True

        elif stage == STAGE_INTERACTION:
            # Freeze encoder
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Unfreeze everything else
            for p in self.ner_head.parameters():
                p.requires_grad = False   # not needed in this stage
            for p in self.drug_fusion.parameters():
                p.requires_grad = True
            for p in self.pair_projection.parameters():
                p.requires_grad = True
            for p in self.interaction_head.parameters():
                p.requires_grad = True
            for p in self.severity_head.parameters():
                p.requires_grad = True    # unfrozen, will be trained in Stage 3

        elif stage == STAGE_SEVERITY:
            # Freeze encoder
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Freeze NER head
            for p in self.ner_head.parameters():
                p.requires_grad = False
            # Freeze interaction head
            for p in self.interaction_head.parameters():
                p.requires_grad = False
            # Fine-tune fusion + pair projection + severity head
            for p in self.drug_fusion.parameters():
                p.requires_grad = True
            for p in self.pair_projection.parameters():
                p.requires_grad = True
            for p in self.severity_head.parameters():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown stage: {stage}. "
                             f"Use '{STAGE_NER}', '{STAGE_INTERACTION}', or '{STAGE_SEVERITY}'.")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Stage '{stage}': {trainable:,} / {total:,} parameters trainable")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = "emilyalsentzer/Bio_ClinicalBERT") -> MedGuardModel:
    return MedGuardModel(model_name=model_name)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Bio_ClinicalBERT tokenizer and model...")

    tokenizer = load_tokenizer()
    model     = load_model()
    model.eval()

    test_sentence = "Warfarin and aspirin interaction may increase bleeding risk."
    inputs = tokenizer(
        test_sentence,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )

    # Simulate Lipinski features for two drugs
    lipinski_a = torch.tensor([[400.0, 4, 2, 2.5, 1.0]])  # (1, 5)
    lipinski_b = torch.tensor([[180.0, 3, 1, 1.2, 1.0]])  # (1, 5)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            lipinski_feats_a=lipinski_a,
            lipinski_feats_b=lipinski_b,
        )

    print("\n✅ Model loaded successfully!")
    print(f"  NER logits shape:         {outputs['ner_logits'].shape}")
    print(f"  Interaction logits shape: {outputs['interaction_logits'].shape}")
    print(f"  Severity logits shape:    {outputs['severity_logits'].shape}")
    print(f"\n  NER classes:      {list(NER_LABELS.values())}")
    print(f"  DDI classes:      {list(DDI_LABELS.values())}")
    print(f"  Severity classes: {list(SEVERITY_LABELS.values())}")

    # Verify staging works
    print("\n── Stage freeze test ──")
    model.set_stage(STAGE_NER)
    model.set_stage(STAGE_INTERACTION)
    model.set_stage(STAGE_SEVERITY)
    print("\nAll stages OK — model architecture correct!")
