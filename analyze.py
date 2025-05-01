import os
import glob
import json
import pandas as pd
import argparse
from collections import defaultdict

def parse_args():
    """
    Define and parse script arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--python_path", 
        type=str, 
        default="./", 
        help="What the default Python path should be set to"
    )
    
    args = parser.parse_args()
    return args


def process_results():

    data_settings = [
        "mfcc_intensity",
        "mfcc_emotion",
        "gfcc_intensity",
        "gfcc_emotion",
    ]
    for data_setting in data_settings:
        base_dir = f"./results/{data_setting}"
        summary_dir = f"./results/_summary"
        os.makedirs(summary_dir, exist_ok=True)

        records = []

        for file_path in glob.glob(os.path.join(base_dir, "*_test.json")):
            filename = os.path.basename(file_path)
            if "__test.json" not in filename or "cr__" not in filename:
                continue

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                accuracy = data.get("accuracy")
                precision = data.get("macro avg", {}).get("precision")
                recall = data.get("macro avg", {}).get("recall")
                f1 = data.get("macro avg", {}).get("f1-score")
            except Exception:
                continue

            model_setting = filename.split("cr__")[-1].replace("__test.json", "")
            records.append((model_setting, accuracy, precision, recall, f1))

        df = pd.DataFrame(records, columns=["model_setting", "accuracy", "precision", "recall", "f1"])

        # Extract model prefix (everything before __rs_{n})
        df["model_prefix"] = df["model_setting"].str.replace(r"__rs_[1-5]", "", regex=True)

        # Average accuracy by model prefix
        df_avg = df.groupby("model_prefix", as_index=False).agg({
            "accuracy": ["mean", "std"],
            "precision": "mean",
            "recall": "mean",
            "f1": "mean"
        })
        df_avg.columns = ["model_prefix", "accuracy_mean", "accuracy_std", "precision_mean", "recall_mean", "f1_mean"]

        # First sort: descending by accuracy
        df_sorted = df_avg.sort_values("accuracy_mean", ascending=False)

        # Second sort: by first letter of model_prefix, then descending accuracy
        df_avg["first_letter"] = df_avg["model_prefix"].str[0]
        df_sorted_per_model = df_avg.sort_values(["first_letter", "accuracy_mean"], ascending=[True, False]).drop(columns=["first_letter"])

        # Save to CSV
        df_sorted.to_csv(os.path.join(summary_dir, f"{data_setting}_sorted.csv"), index=False)
        df_sorted_per_model.to_csv(os.path.join(summary_dir, f"{data_setting}_sorted_per_model.csv"), index=False)


if __name__ == "__main__":
    
    args = parse_args()
    os.environ['PYTHONPATH'] = args.python_path
    process_results()

"""

### Run Command

python analyze.py

"""