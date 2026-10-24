# Standard library imports
import io
import re
import logging
from typing import Any, Dict, Tuple

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DataPreparation:
    """
    Class encapsulating data ingestion, cleaning, feature engineering, and 
    building the sklearn ColumnTransformer preprocessor.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = None

    def load_raw_data(self) -> pd.DataFrame:
        """Loads dataset with fallback handling for binary offset headers."""
        file_path = self.config['file_path']
        logging.info(f"Ingesting raw dataset from: {file_path}")
        with open(file_path, 'rb') as f:
            content = f.read()

        header_start = content.find(b'ID,URL')
        data_bytes = content[header_start:]
        text = data_bytes.decode('latin1')

        lines = text.splitlines()
        csv_lines = [
            line for line in lines 
            if line.startswith(('ID,', '0,', '1,', '2,', '3,', '4,', '5,', '6,', '7,', '8,', '9,')) 
            or (len(line) > 0 and line[0].isdigit())
        ]

        csv_text = '\n'.join(csv_lines)
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = [c.strip().replace('\\', '') for c in df.columns]
        return df

    def clean_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes deduplication, missing value imputation, and feature creation."""
        logging.info("Executing data cleaning and feature engineering pipeline...")
        cleaned_df = df.copy()

        # 1. Deduplication
        init_len = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        logging.info(f"Removed {init_len - len(cleaned_df)} duplicate rows.")

        # 2. Missing Value Imputation (Copy-on-write safe reassignments)
        cleaned_df['num_videos'] = cleaned_df['num_videos'].fillna(0)
        cleaned_df['data_channel'] = cleaned_df['data_channel'].fillna('Unknown')
        cleaned_df['num_imgs'] = cleaned_df['num_imgs'].fillna(cleaned_df['num_imgs'].median())

        median_cols = [
            'num_hrefs', 'num_self_hrefs', 
            'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares'
        ]
        for col in median_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

        # 3. Drop Identifiers
        cleaned_df.drop(columns=self.config.get('drop_columns', ['ID', 'URL']), inplace=True, errors='ignore')

        # 4. Feature Engineering
        cleaned_df['self_link_ratio'] = cleaned_df['num_self_hrefs'] / (cleaned_df['num_hrefs'] + 1e-5)
        cleaned_df['total_media'] = cleaned_df['num_imgs'] + cleaned_df['num_videos']
        cleaned_df['log_shares'] = np.log1p(cleaned_df['shares'])

        logging.info(f"Data cleaning and feature engineering complete. Shape: {cleaned_df.shape}")
        return cleaned_df

    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Constructs the ColumnTransformer for numeric scaling and nominal one-hot encoding."""
        nominal_features = self.config['nominal_features']
        numerical_features = [col for col in X.columns if col not in nominal_features]

        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('nom', nominal_transformer, nominal_features)
            ],
            remainder='passthrough',
            n_jobs=-1
        )
        return self.preprocessor
