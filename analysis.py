"""
analysis.py
-----------
Text cleaning, fraud keyword detection, SHAP explainability chart,
highlighted text, confusion matrix, and model comparison chart.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------
# Fraud Terms
# ---------------------------------------------------------------
FRAUD_TERMS = [
    "lottery", "prize", "urgent", "winner", "claim",
    "refund", "verify", "bankruptcy", "account", "offer",
    "offshore", "unauthorized", "irregular", "undisclosed",
    "liability", "laundering", "ponzi", "embezzle", "scheme",
    "without approval", "conceal", "fictitious", "fabricat",
    "misappropriat", "falsif", "revenue recognition",
    "shell company", "kickback", "insider trading",
]

# ---------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------------------------------------------
# Highlighted Text
# ---------------------------------------------------------------
def highlight_text(original_text: str, detected_terms: list) -> str:
    """
    Returns HTML with detected fraud terms highlighted in red
    directly inside the original input text.
    """
    if not detected_terms:
        return f"<p style='font-size:15px;line-height:1.9;'>{original_text}</p>"

    highlighted = original_text
    for term in sorted(detected_terms, key=len, reverse=True):
        pattern     = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(
            f"<mark style='background-color:#c53030;color:white;"
            f"padding:2px 7px;border-radius:4px;font-weight:600;'>{term}</mark>",
            highlighted
        )
    return f"<p style='font-size:15px;line-height:1.9;'>{highlighted}</p>"

# ---------------------------------------------------------------
# SHAP Chart
# ---------------------------------------------------------------
def plot_shap(text: str, tfidf, model, df):
    """
    Generates a SHAP bar chart showing which words drove the
    fraud score up (red) or down (green).
    Automatically selects LinearExplainer or TreeExplainer
    based on the model type.
    """
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    X_bg    = tfidf.transform(df["filing_text"])

    # Pick explainer based on model type
    if isinstance(model, LogisticRegression):
        explainer   = shap.LinearExplainer(
            model, X_bg, feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(vec)
        shap_scores = shap_values[0]

    else:
        # Convert sparse to dense to avoid numpy dtype casting error
        vec_dense   = vec.toarray().astype(float)
        X_bg_dense  = X_bg.toarray().astype(float)

        explainer   = shap.TreeExplainer(model, X_bg_dense)
        shap_values = explainer.shap_values(vec_dense)

        # TreeExplainer returns list [class0, class1] for binary classifiers
        if isinstance(shap_values, list):
            shap_scores = np.array(shap_values[1]).flatten()
        else:
            shap_scores = np.array(shap_values).flatten()

    feature_names  = tfidf.get_feature_names_out()
    input_features = vec.toarray()[0]
    active_idx     = np.where(input_features > 0)[0]

    if len(active_idx) == 0:
        return None

    active_scores = [(feature_names[i], shap_scores[i]) for i in active_idx]
    active_scores = sorted(active_scores, key=lambda x: abs(float(x[1])), reverse=True)[:10]
    # Use distinct variable names to avoid overwriting shap_scores
    words        = [w for w, _ in active_scores]
    score_values = [float(s) for _, s in active_scores]
    colors       = ["#e53e3e" if s > 0 else "#38a169" for s in score_values]

    fig, ax = plt.subplots(figsize=(7, max(3, len(words) * 0.45)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.barh(words, score_values, color=colors)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on fraud score)", fontsize=10, color="#111111")
    ax.set_title("Word-level fraud impact", fontsize=11,
                 fontweight="bold", color="#111111")
    ax.tick_params(axis="y", labelsize=9, colors="#111111")
    ax.tick_params(axis="x", labelsize=9, colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    legend_elements = [
        mpatches.Patch(facecolor="#e53e3e", label="Increases fraud risk"),
        mpatches.Patch(facecolor="#38a169", label="Decreases fraud risk"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right",
              facecolor="#ffffff", edgecolor="#cccccc", labelcolor="#111111")

    plt.tight_layout()
    return fig

# ---------------------------------------------------------------
# Confusion Matrix Chart
# ---------------------------------------------------------------
def plot_confusion_matrix(cm, model_name: str):
    """Renders a confusion matrix heatmap for a given model."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.imshow(cm, interpolation="nearest", cmap="Reds")
    ax.set_title(f"Confusion matrix — {model_name}",
                 fontsize=10, fontweight="bold", color="#111111")

    classes    = ["Legitimate", "Fraud"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=9, color="#111111")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9, color="#111111")
    ax.set_ylabel("Actual",    fontsize=9, color="#111111")
    ax.set_xlabel("Predicted", fontsize=9, color="#111111")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="#111111")

    plt.tight_layout()
    return fig

# ---------------------------------------------------------------
# Model Comparison Bar Chart
# ---------------------------------------------------------------
def plot_model_comparison(model_results: dict):
    """Bar chart comparing Accuracy, F1, Precision, Recall across models."""
    names   = list(model_results.keys())
    metrics = ["accuracy", "f1", "precision", "recall"]
    labels  = ["Accuracy", "F1 Score", "Precision", "Recall"]
    colors  = ["#2b6cb0", "#c53030", "#276749", "#975a16"]

    x   = np.arange(len(names))
    w   = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [model_results[n][metric] for n in names]
        bars   = ax.bar(x + i * w, values, w, label=label, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7, color="#111111"
            )

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, fontsize=10, color="#111111")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=10, color="#111111")
    ax.set_title("Model comparison", fontsize=11,
                 fontweight="bold", color="#111111")
    ax.tick_params(colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.legend(fontsize=8, facecolor="#ffffff",
              edgecolor="#cccccc", labelcolor="#111111")

    plt.tight_layout()
    return fig