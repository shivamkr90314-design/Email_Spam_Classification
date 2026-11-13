
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "database" / "SMSSpamCollection"

def check_dataset():
    print("Checking dataset...")

    if not DATASET_PATH.exists():
        print("ERROR: Dataset not found!")
        print("Expected location:", DATASET_PATH)
        return

    try:
        df = pd.read_csv(
            DATASET_PATH,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1"
        )

        print("\nDataset loaded successfully!")
        print("Total messages:", len(df))

        print("\nFirst 5 messages:")
        print(df.head())

        print("\nSpam and Ham count:")
        print(df["label"].value_counts())

        print("\nMissing values:")
        print(df.isnull().sum())

    except Exception as e:
        print("Dataset error:", e)

if __name__ == "__main__":
    check_dataset()
