from src.data.load_data import load_rideshare_data
from src.data.save_data import save_csv
from src.data.clean_data import clean_data

from src.features.encode import encode_features
from src.features.split_scale import split_and_scale


def run_pipeline():
    input_path = "data/raw/rideshare_kaggle.csv"

    clean_output = "data/processed/rideshare_cleaned.csv"
    encoded_output = "data/processed/rideshare_encoded.csv"

    # load
    df = load_rideshare_data(input_path)

    # clean
    df_cleaned = clean_data(df)

    # encode
    df_encoded = encode_features(df_cleaned)

    # save
    save_csv(df_cleaned, clean_output)
    save_csv(df_encoded, encoded_output)

    # split + scale (for modeling stage)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_encoded)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    run_pipeline()