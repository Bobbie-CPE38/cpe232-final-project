import pandas as pd

COLS_TO_DROP = [
    'id', 'timestamp', 'datetime', 'timezone', 'long_summary', 
    'icon', 'visibility.1', 'windGustTime', 'sunriseTime', 'sunsetTime',
    'uvIndexTime', 'temperatureHighTime', 'temperatureLowTime', 
    'apparentTemperatureHighTime', 'apparentTemperatureLowTime',
    'temperatureMinTime', 'temperatureMaxTime', 'apparentTemperatureMinTime', 
    'apparentTemperatureMaxTime', 'product_id'
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop missing target
    df = df.dropna(subset=['price'])

    # drop unused columns
    df = df.drop(columns=COLS_TO_DROP)

    return df