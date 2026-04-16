import time

from src.data.load import load_rideshare_data
from src.data.save import save_csv
from src.data.clean import clean_data

from src.features.encode import encode_features
from src.features.split import split_data
from src.features.scale import scale_data


def run_pipeline(sample_size=None, do_scale=True):
    input_path = "data/raw/rideshare_kaggle.csv"

    clean_output = "data/processed/rideshare_cleaned.csv"
    encoded_output = "data/processed/rideshare_encoded.csv"

    # Helper func
    def log(step):
        print(f"[{step}] done in {time.time() - start:.2f}s")

    # load
    start = time.time()
    print("Loading data...")
    df = load_rideshare_data(input_path)
    log("load")

    # optional sampling (for faster dev)
    if sample_size:
        print(f"Sampling {sample_size} rows...")
        df = df.sample(sample_size, random_state=42)

    # clean
    start = time.time()
    print("Cleaning data...")
    df_cleaned = clean_data(df)
    log("clean")

    # encode
    start = time.time()
    print("Encoding features...")
    df_encoded = encode_features(df_cleaned)
    log("encode")

    # save
    start = time.time()
    print("Saving files...")
    save_csv(df_cleaned, clean_output)
    save_csv(df_encoded, encoded_output)
    log("save")

    # split + optional scale
    start = time.time()
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    if do_scale:
        print("Scaling data...")
        X_train, X_test, scaler = scale_data(X_train, X_test)
    else:
        scaler = None

    log("split/scale")

    print("Pipeline finished.")
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Dev
    run_pipeline(sample_size=50000, do_scale=False)

    # Final
    # run_pipeline(sample_size=None, do_scale=True)