import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # one-hot
    df = pd.get_dummies(
        df,
        columns=['cab_type', 'short_summary'],
        drop_first=True
    )

    # label encoding
    le = LabelEncoder()
    target_cols = ['name', 'source', 'destination']
    for col in target_cols:
        df[col] = le.fit_transform(df[col])

    return df