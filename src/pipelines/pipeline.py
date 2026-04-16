import time

from src.data.load import load_rideshare_data
from src.data.save import save_csv
from src.data.clean import clean_data

from src.features.build import build_features


def data_prep(sample_size=None):
    input_path = "data/raw/rideshare_kaggle.csv"

    output_cleaned = "data/processed/rideshare_cleaned.csv"
    output_features = "data/processed/rideshare_feature_engineering.csv"

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

    # feature en
    start = time.time()
    print("Building features...")
    df_features = build_features(df_cleaned)
    log("feature_engineering")

    # save
    start = time.time()
    print("Saving files...")
    save_csv(df_cleaned, output_cleaned)
    save_csv(df_features, output_features)
    log("save")

    print("Data Preparation Pipeline finished.")


if __name__ == "__main__":
    # Dev
    data_prep(sample_size=50000)
    # train()

    # Final
    # data_prep()
    # train()






# # split + optional scale
# start = time.time()
# print("Splitting data...")
# X_train, X_test, y_train, y_test = split_data(df_features)
# if do_scale:
#     print("Scaling data...")
#     X_train, X_test, scaler = scale_data(X_train, X_test)
# else:
#     scaler = None

# log("split/scale")