# this file loads lipinski features and serves vectors for drugs
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
LIPINSKI_FEATURES = ['molecular_weight', 'n_hba', 'n_hbd', 'logp', 'ro5_fulfilled']
CONTINUOUS_FEATURES = ['molecular_weight', 'n_hba', 'n_hbd', 'logp']
LIPINSKI_DIM = len(LIPINSKI_FEATURES)

# this class groups logic for LipinskiProcessor
class LipinskiProcessor:

    # this function is used to set up initial values for this object
    def __init__(self):
        self.drug_id_to_features: Dict[str, np.ndarray] = {}
        self._loaded = False

    # this function is used to handle load
    def load(self, csv_path: str):
        print(f'Loading Lipinski data from {csv_path}...')
        df = pd.read_csv(csv_path)
        print(f'  Loaded {len(df)} compounds | columns: {list(df.columns)}')
        id_col = self._find_id_column(df)
        print(f'  ID column: {id_col}')
        means = {}
        stds = {}
        for col in CONTINUOUS_FEATURES:
            if col in df.columns:
                means[col] = df[col].mean()
                stds[col] = df[col].std()
                if stds[col] > 0:
                    df[col] = (df[col] - means[col]) / stds[col]
            else:
                means[col] = 0.0
                stds[col] = 1.0
        if 'ro5_fulfilled' in df.columns:
            df['ro5_fulfilled'] = df['ro5_fulfilled'].astype(float)
        for _, row in df.iterrows():
            drug_id = str(row[id_col])
            features = np.array([float(row[f]) if f in df.columns else 0.0 for f in LIPINSKI_FEATURES], dtype=np.float32)
            self.drug_id_to_features[drug_id] = features
        self._loaded = True
        print(f'  Indexed {len(self.drug_id_to_features)} drugs (normalized)')
        print(f'  Continuous feature means: { {k: round(v, 3) for k, v in means.items()}}')

    # this function is used to handle find id column
    def _find_id_column(self, df: pd.DataFrame) -> str:
        candidates = ['ID', 'drugbank_id', 'DrugBank ID', 'drugbank-id', 'drug_id', 'DrugBankID']
        for name in candidates:
            if name in df.columns:
                return name
        return df.columns[0]

    # this function is used to get features
    def get_features(self, drug_id: str) -> Optional[np.ndarray]:
        return self.drug_id_to_features.get(drug_id)

    # this function is used to check whether drug available
    def is_drug_available(self, drug_id: str) -> bool:
        return drug_id in self.drug_id_to_features

    # this function is used to get feature dim
    def get_feature_dim(self) -> int:
        return LIPINSKI_DIM

    # this function is used to handle coverage
    def coverage(self) -> int:
        return len(self.drug_id_to_features)

    # this function is used to handle len
    def __len__(self):
        return len(self.drug_id_to_features)
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'DB_compounds_lipinski.csv')
    if not os.path.exists(csv_path):
        print(f'File not found: {csv_path}')
    else:
        processor = LipinskiProcessor()
        processor.load(csv_path)
        print(f'\nFeature dimension : {processor.get_feature_dim()}')
        print(f'Total drugs       : {processor.coverage()}')
        first_id = list(processor.drug_id_to_features.keys())[0]
        print(f'Sample ID         : {first_id}')
        print(f'Sample features   : {processor.get_features(first_id)}')
        print(f'\nLipinski processor OK!')
