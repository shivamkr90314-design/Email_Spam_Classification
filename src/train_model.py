
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "database" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_model.joblib"

def train_model():

    print("Loading dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )

    df = df.dropna()

    X = df["message"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training model...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel training completed!")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Location:", MODEL_PATH)

if __name__ == "__main__":
    train_model()
