import pandas as pd

def load_rideshare_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)