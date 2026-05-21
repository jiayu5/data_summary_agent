"""Standalone script to train and save the Titanic survival prediction model."""

import os

from engine.ml_engine import load_and_preprocess, train_model, save_model, DATA_DIR

CSV_PATH = os.path.join(DATA_DIR, "titanic_cleaned.csv")


def main():
    print("Loading and preprocessing data...")
    X, y, feature_names, preprocessor = load_and_preprocess(CSV_PATH)
    print(f"Features: {feature_names}")
    print(f"Samples: {len(X)}, Positive: {sum(y)}, Negative: {len(y) - sum(y)}")

    print("\nTraining RandomForest model...")
    model, metrics = train_model(X, y)

    print(f"\nAccuracy: {metrics['accuracy']}")
    print(f"AUC: {metrics['auc']}")
    print(f"\n{metrics['report']}")

    print("Saving model...")
    save_model(model, preprocessor, feature_names)
    print(f"Model saved to {os.path.join(DATA_DIR, 'models')}")


if __name__ == "__main__":
    main()
