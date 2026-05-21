import json
import os
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

from engine.query_engine import call_llm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")

DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]
NUMERIC_COLS = ["Pclass", "Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_COLS = ["Sex", "Embarked"]
TARGET_COL = "Survived"


def load_and_preprocess(csv_path: str):
    """Load CSV and preprocess features. Returns (X, y, feature_names, preprocessor)."""
    df = pd.read_csv(csv_path)

    y = df[TARGET_COL].values
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET_COL])

    # Store fill values for inference
    preprocessor = {
        "numeric_medians": {},
        "categorical_modes": {},
        "label_encoders": {},
    }

    # Numeric: fill with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            preprocessor["numeric_medians"][col] = median_val

    # Categorical: fill with mode + LabelEncoder
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            preprocessor["label_encoders"][col] = le
            preprocessor["categorical_modes"][col] = mode_val

    feature_names = list(df.columns)
    X = df.values.astype(float)

    return X, y, feature_names, preprocessor


def train_model(X, y):
    """Train RandomForest and return (model, metrics)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
        "report": classification_report(y_test, y_pred, target_names=["Died", "Survived"]),
    }

    return model, metrics


def save_model(model, preprocessor, feature_names, dir_path=None):
    """Save model and preprocessor to disk."""
    if dir_path is None:
        dir_path = MODEL_DIR
    os.makedirs(dir_path, exist_ok=True)

    joblib.dump(model, os.path.join(dir_path, "model.joblib"))

    # Convert LabelEncoder objects to serializable format
    serializable_le = {}
    for col, le in preprocessor["label_encoders"].items():
        serializable_le[col] = list(le.classes_)

    meta = {
        "preprocessor": {
            "numeric_medians": preprocessor["numeric_medians"],
            "categorical_modes": preprocessor["categorical_modes"],
            "label_encoders": serializable_le,
        },
        "feature_names": feature_names,
    }
    with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_model_and_preprocessor(dir_path=None):
    """Load model and preprocessor. Returns (model, preprocessor, feature_names)."""
    if dir_path is None:
        dir_path = MODEL_DIR

    model = joblib.load(os.path.join(dir_path, "model.joblib"))
    with open(os.path.join(dir_path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Reconstruct LabelEncoder objects from saved classes
    preprocessor = meta["preprocessor"]
    label_encoders = {}
    for col, classes in preprocessor["label_encoders"].items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        label_encoders[col] = le
    preprocessor["label_encoders"] = label_encoders

    return model, preprocessor, meta["feature_names"]


def extract_features_from_text(question: str) -> dict:
    """Use LLM to extract passenger features from natural language."""
    system = (
        "你是数据提取助手。从用户的自然语言描述中提取 Titanic 乘客特征。\n"
        "返回 JSON 格式，只包含用户明确提到的字段。\n\n"
        "可用字段：\n"
        "- pclass: int (1/2/3) 舱位等级\n"
        "- sex: str (male/female)\n"
        "- age: float 年龄\n"
        "- sibsp: int 同行兄弟姐妹/配偶数\n"
        "- parch: int 同行父母/子女人数\n"
        "- fare: float 票价\n"
        "- embarked: str (S/C/Q) 登船港口\n\n"
        "只返回 JSON，不要解释。未提到的字段不要包含。"
    )
    user = f"描述：{question}"

    raw = call_llm(system, user)

    # Extract JSON from response
    import re
    raw = re.sub(r"```(?:json)?\s*\n?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return {}


def predict_single(model, preprocessor, feature_names, passenger_dict: dict) -> dict:
    """Predict survival for a single passenger described as a dict."""
    # Normalize keys to match feature names (case-insensitive)
    key_map = {k.lower(): k for k in passenger_dict}

    row = {}
    for col in feature_names:
        col_lower = col.lower()
        if col in NUMERIC_COLS:
            if col_lower in key_map:
                row[col] = float(passenger_dict[key_map[col_lower]])
            else:
                row[col] = preprocessor["numeric_medians"][col]
        elif col in CATEGORICAL_COLS:
            if col_lower in key_map:
                le = preprocessor["label_encoders"][col]
                val = str(passenger_dict[key_map[col_lower]])
                if val in le.classes_:
                    row[col] = float(le.transform([val])[0])
                else:
                    row[col] = 0.0
            else:
                row[col] = 0.0

    X = np.array([[row[col] for col in feature_names]])
    prob = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])

    return {
        "survived": pred,
        "probability": round(float(prob[1]), 4),
        "label": "存活" if pred == 1 else "未存活",
    }
