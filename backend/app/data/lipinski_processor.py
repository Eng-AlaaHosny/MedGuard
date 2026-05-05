"""
Lipinski Physicochemical Feature Processor
===========================================
Loads DB_compounds_lipinski.csv and provides normalized feature vectors
for use in the MedGuard model's drug_fusion layer.

Features (5-dimensional):
  molecular_weight  — continuous, z-score normalized
  n_hba             — continuous, z-score normalized (H-bond acceptors)
  n_hbd             — continuous, z-score normalized (H-bond donors)
  logp              — continuous, z-score normalized (lipophilicity)
  ro5_fulfilled     — binary float (0.0 / 1.0), not normalized

Normalization:
  Applied at load time (not lazily) so all tensors are inference-ready.
  ro5_fulfilled is excluded from z-score normalization (binary feature).

Lookup key: DrugBank ID (e.g. "DB00006").
  - resolve_drug_name_to_id() in routes.py maps drug names → IDs via KG.
  - Returns None when drug not in dataset → zero vector in model.

Coverage: 2,647 compounds. 2,145 overlap with the KG (4,499 nodes).
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

LIPINSKI_FEATURES  = ['molecular_weight', 'n_hba', 'n_hbd', 'logp', 'ro5_fulfilled']
CONTINUOUS_FEATURES = ['molecular_weight', 'n_hba', 'n_hbd', 'logp']
LIPINSKI_DIM        = len(LIPINSKI_FEATURES)  # 5


class LipinskiProcessor:
    """
    Processes DrugBank Lipinski CSV.
    Key design decisions:
      - Normalization happens at load() time — always normalized before use.
      - ro5_fulfilled treated as binary (0.0 / 1.0), not normalized.
      - Lookup by DrugBank ID only (not by name — avoids ambiguity).
      - Returns None for missing drugs; callers zero-pad the model input.
    """

    def __init__(self):
        self.drug_id_to_features: Dict[str, np.ndarray] = {}
        self._loaded = False

    def load(self, csv_path: str):
        """Load CSV, normalize continuous features, index by DrugBank ID."""
        print(f"Loading Lipinski data from {csv_path}...")

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} compounds | columns: {list(df.columns)}")

        id_col = self._find_id_column(df)
        print(f"  ID column: {id_col}")

        # Z-score normalize continuous features
        means = {}
        stds  = {}
        for col in CONTINUOUS_FEATURES:
            if col in df.columns:
                means[col] = df[col].mean()
                stds[col]  = df[col].std()
                if stds[col] > 0:
                    df[col] = (df[col] - means[col]) / stds[col]
            else:
                means[col] = 0.0
                stds[col]  = 1.0

        # ro5_fulfilled → binary float
        if 'ro5_fulfilled' in df.columns:
            df['ro5_fulfilled'] = df['ro5_fulfilled'].astype(float)

        # Index by DrugBank ID
        for _, row in df.iterrows():
            drug_id = str(row[id_col])
            features = np.array(
                [float(row[f]) if f in df.columns else 0.0 for f in LIPINSKI_FEATURES],
                dtype=np.float32,
            )
            self.drug_id_to_features[drug_id] = features

        self._loaded = True
        print(f"  Indexed {len(self.drug_id_to_features)} drugs (normalized)")
        print(f"  Continuous feature means: { {k: round(v,3) for k,v in means.items()} }")

    def _find_id_column(self, df: pd.DataFrame) -> str:
        """Find DrugBank ID column. CSV confirmed to use 'ID'."""
        candidates = ['ID', 'drugbank_id', 'DrugBank ID', 'drugbank-id', 'drug_id', 'DrugBankID']
        for name in candidates:
            if name in df.columns:
                return name
        return df.columns[0]

    def get_features(self, drug_id: str) -> Optional[np.ndarray]:
        """
        Get normalized Lipinski features for a drug by DrugBank ID.
        Returns None if drug not in dataset — caller should zero-pad.
        """
        return self.drug_id_to_features.get(drug_id)

    def is_drug_available(self, drug_id: str) -> bool:
        return drug_id in self.drug_id_to_features

    def get_feature_dim(self) -> int:
        return LIPINSKI_DIM

    def coverage(self) -> int:
        return len(self.drug_id_to_features)

    def __len__(self):
        return len(self.drug_id_to_features)


if __name__ == "__main__":
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    csv_path  = os.path.join(base_dir, 'DB_compounds_lipinski.csv')

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
    else:
        processor = LipinskiProcessor()
        processor.load(csv_path)

        print(f"\nFeature dimension : {processor.get_feature_dim()}")
        print(f"Total drugs       : {processor.coverage()}")

        first_id = list(processor.drug_id_to_features.keys())[0]
        print(f"Sample ID         : {first_id}")
        print(f"Sample features   : {processor.get_features(first_id)}")
        print(f"\nLipinski processor OK!")
