"""
model.py
--------
Handles all ML model training, prediction, and comparison.
Trains Logistic Regression, Random Forest, and XGBoost.
Automatically picks the best model by F1 score.
"""

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score,
    confusion_matrix, accuracy_score,
    precision_score, recall_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@st.cache_resource
def train_all_models(df):
    """
    Trains TF-IDF vectorizer + 3 classifiers on the dataset.
    Returns tfidf, results dict, best model name.
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    X     = tfidf.fit_transform(df["filing_text"])
    y     = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )

    results = {}
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        results[name] = {
            "model":     m,
            "accuracy":  round(accuracy_score(y_test, y_pred),  3),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 3),
            "report":    classification_report(y_test, y_pred, zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred),
        }

    best_name = max(results, key=lambda k: results[k]["f1"])
    return tfidf, results, best_name


def predict(tfidf, model, text: str, fraud_terms: list) -> tuple:
    """
    Vectorizes cleaned text, runs prediction, applies keyword boost.
    Returns (fraud_prob, detected_terms).
    """
    from analysis import clean_text
    cleaned        = clean_text(text)
    vec            = tfidf.transform([cleaned])
    fraud_prob     = model.predict_proba(vec)[0][1]
    detected_terms = [t for t in fraud_terms if t in cleaned]

    if len(detected_terms) >= 2:
        fraud_prob = max(fraud_prob, 0.75)

    return round(float(fraud_prob), 3), detected_terms


def risk_label(fraud_prob: float) -> str:
    if fraud_prob > 0.45:
        return "HIGH"
    elif fraud_prob > 0.25:
        return "MEDIUM"
    return "LOW"