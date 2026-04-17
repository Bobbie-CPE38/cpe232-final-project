import time
import joblib
import json

from src.data.load import load_rideshare_data
from src.data.save import save_csv
from src.data.clean import clean_data

from src.features.build import build_features
from src.features.split import split_data

from src.models.train import train_model
from src.models.evaluate import evaluate_model

from src.utils import get_base_dir, log

    
BASE_DIR = get_base_dir()
RAW_DATA_PATH = BASE_DIR / "data/raw/rideshare_kaggle.csv"
CLEANED_DATA_PATH = BASE_DIR / "data/processed/rideshare_cleaned.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data/processed/rideshare_feature_engineering.csv"
MODEL_PATH = BASE_DIR / "models/model.pkl"
METRICS_PATH = BASE_DIR / "models/metrics.json"

def data_prep(sample_size=None):
    # load
    start = time.time()
    print("Loading data...")
    df = load_rideshare_data(RAW_DATA_PATH)
    log("load", start)

    # optional sampling (for faster dev)
    if sample_size:
        print(f"Sampling {sample_size} rows...")
        df = df.sample(sample_size, random_state=42)

    # clean
    start = time.time()
    print("Cleaning data...")
    df_cleaned = clean_data(df)
    log("clean", start)

    # feature en
    start = time.time()
    print("Building features...")
    df_features = build_features(df_cleaned)
    log("feature_engineering", start)

    # save
    start = time.time()
    print("Saving files...")
    save_csv(df_cleaned, CLEANED_DATA_PATH)
    save_csv(df_features, PROCESSED_DATA_PATH)
    log("save", start)

    print("Data Preparation Pipeline finished.")


# def train():
#     # load
#     start = time.time()
#     print("Loading features...")
#     df = load_rideshare_data(PROCESSED_DATA_PATH)
#     log("load", start)

#     # split
#     start = time.time()
#     print("Splitting data...")
#     X_train, X_test, y_train, y_test = split_data(df)
#     log("split", start)

#     # train (with CV + tuning)
#     start = time.time()
#     print("Training...")
#     model, best_params, cv_mae = train_model(X_train, y_train)
#     log("train", start)

#     # evaluate on test set
#     start = time.time()
#     print("Evaluating on test set...")
#     metrics_eval = evaluate_model(model, X_test, y_test)
#     test_mae = metrics_eval["mae"]
#     print(f"Test MAE: {test_mae:.4f}")
#     log("evaluate", start)

#     # save model
#     start = time.time()
#     print("Saving model...")
#     joblib.dump(model, MODEL_PATH)
#     log("save_model", start)

#     # save metrics
#     start = time.time()
#     print("Saving metrics...")
#     metrics = {
#         "cv_mae": cv_mae,
#         "test_mae": test_mae,
#         "best_params": best_params
#     }

#     with open(METRICS_PATH, "w") as f:
#         json.dump(metrics, f, indent=4)

#     log("save_metrics", start)

#     print("Training pipeline finished.")

# def run_all(sample_size=None):
#     data_prep(sample_size=sample_size)
#     train()

if __name__ == "__main__":
    pass
    # Dev
    # run_all(sample_size=50000)

    # Final
    # run_all()