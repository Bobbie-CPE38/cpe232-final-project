import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


TIER_MAP = {
    'Shared': 'Economy', 'UberPool': 'Economy',
    'Lyft': 'Mid', 'UberX': 'Mid', 'WAV': 'Mid',
    'Lyft XL': 'Mid', 'UberXL': 'Mid',
    'Lux': 'Premium', 'Black': 'Premium',
    'Lux Black': 'Premium', 'Black SUV': 'Premium', 'Lux Black XL': 'Premium'
}

TIER_ENC_MAP = {
    'Economy': 0,
    'Mid': 1,
    'Premium': 2,
    'Unknown': -1
}


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df['tier'] = df['name'].map(TIER_MAP).fillna('Unknown')
    df['route'] = df['source'] + ' → ' + df['destination']
    return df


def add_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    df['log_distance'] = np.log1p(df['distance'])
    df['surge_x_distance'] = df['surge_multiplier'] * df['distance']
    df['surge_intensity'] = (df['surge_multiplier'] - 1) * df['distance']
    df['is_surge'] = (df['surge_multiplier'] > 1).astype(int)
    return df


def add_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    df['tier_enc'] = df['tier'].map(TIER_ENC_MAP).fillna(-1)
    df['cab_type_enc'] = (df['cab_type'] == 'Lyft').astype(int)
    return df


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    df['route_frequency'] = df['route'].map(df['route'].value_counts())
    return df


# TEMP ยังไม่เหมาะกับ pipeline
def add_label_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    le_name = LabelEncoder()
    le_dest = LabelEncoder()
    le_source = LabelEncoder()
    df['name_enc'] = le_name.fit_transform(df['name'].astype(str))
    df['destination_enc'] = le_dest.fit_transform(df['destination'].astype(str))
    df['source_enc'] = le_source.fit_transform(df['source'].astype(str))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_basic_features(df)
    df = add_numerical_features(df)
    df = add_encoded_features(df)
    df = add_frequency_features(df)

    # TEMP
    df = add_label_encoded_features(df)

    return df