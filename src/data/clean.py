import pandas as pd

COLS_TO_DROP = [
    'id', 'timestamp', 'datetime', 'timezone', 'long_summary', 
    'icon', 'visibility.1', 'windGustTime', 'sunriseTime', 'sunsetTime',
    'uvIndexTime', 'temperatureHighTime', 'temperatureLowTime', 
    'apparentTemperatureHighTime', 'apparentTemperatureLowTime',
    'temperatureMinTime', 'temperatureMaxTime', 'apparentTemperatureMinTime', 
    'apparentTemperatureMaxTime', 'product_id'
]

def drop_missing_price(df) -> pd.DataFrame:
    return df.dropna(subset=['price'])

def drop_unused_columns(df) -> pd.DataFrame:
    return df.drop(columns=COLS_TO_DROP)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = drop_missing_price(df)
    df = drop_unused_columns(df)
    return df