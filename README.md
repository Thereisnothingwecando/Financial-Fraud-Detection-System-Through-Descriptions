# 🛡️ FraudShield — NLP-Based Financial Fraud Detection

> **Final Year Project** | NLP & Machine Learning | Proactive Financial Fraud Detection from Transaction Narratives

---

## 📌 Overview

FraudShield is an AI-powered early warning system that analyses transaction narrations and SEC financial filing text in real time to detect potential fraud. It combines classical machine learning with large language model (LLM) explanations to provide explainable, actionable fraud alerts.

The system was built as a final year undergraduate project and demonstrates how NLP techniques can be applied to proactive financial fraud detection.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 **Single Analysis** | Analyse any transaction narration or SEC filing text instantly |
| 🔦 **Highlighted Text** | Suspicious words highlighted directly in the input |
| 🧠 **SHAP Explainability** | Word-level impact chart showing what drove the risk score |
| 🚩 **Keyword Detection** | 30+ fraud indicator terms from SEC and FinCEN patterns |
| 🤖 **AI Analysis** | Groq / Llama 3 explains the reasoning and recommends actions |
| 📁 **Batch CSV Upload** | Score hundreds of transactions at once |
| 📄 **PDF Report Export** | Professional downloadable report with flagged transactions |
| 📊 **Model Comparison** | Logistic Regression vs Random Forest vs XGBoost side-by-side |
| 📈 **Confusion Matrix** | Per-model performance visualisation |

---

## 🗂️ Project Structure

```
FraudShield/
├── app.py                    # Main Streamlit UI
├── model.py                  # ML model training & prediction
├── analysis.py               # SHAP, text highlighting, charts
├── llm.py                    # Groq LLM integration
├── pdf_report.py             # PDF report generation
├── prepare_dataset.py        # Dataset preparation (run once)
├── requirements.txt          # Python dependencies
├── Final_Dataset.csv         # Raw input dataset
├── final_labeled_fraud_dataset.csv  # Labeled dataset (generated)
└── .streamlit/
    ├── secrets.toml          # API keys (never commit this!)
    └── config.toml           # Streamlit theme config
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraudshield.git
cd fraudshield
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Get a **free** Groq API key at [console.groq.com](https://console.groq.com)

### 4. Prepare the dataset

Run this once to generate the labeled dataset from your raw CSV:

```bash
python prepare_dataset.py
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 How It Works

```
Input Text
    │
    ▼
Text Cleaning (lowercase, remove special chars)
    │
    ▼
TF-IDF Vectorisation (500 features)
    │
    ├──► ML Model (Best of LR / RF / XGBoost)
    │         │
    │         ▼
    │    Fraud Probability Score (0–1)
    │
    ├──► Keyword Detection (30+ fraud terms)
    │         │
    │         ▼
    │    Score Boost if 2+ signals detected
    │
    ├──► SHAP Explainability
    │         │
    │         ▼
    │    Word-level impact chart
    │
    └──► Groq / Llama 3 LLM
              │
              ▼
         Structured AI Analysis
         (Summary, Signals, Reasoning, Action)
```

### Risk Thresholds

| Score | Risk Level |
|-------|------------|
| > 45% | 🔴 HIGH — Immediate review |
| 25–45% | 🟡 MEDIUM — Manual verification |
| < 25% | 🟢 LOW — Appears safe |

---

## 🤖 ML Models

Three models are trained and compared automatically:

| Model | Notes |
|-------|-------|
| Logistic Regression | Fast, interpretable, good SHAP support |
| Random Forest | Ensemble, handles non-linear patterns |
| XGBoost | Gradient boosting, often best F1 score |

The model with the **highest F1 score** is automatically selected as the active model.

---

## 📊 Dataset Labeling

Since no pre-labeled fraud dataset was available, a **rule-based labeling strategy** was used inspired by:

- SEC enforcement action patterns
- FinCEN financial crime advisories
- Common financial scam language patterns

**Fraud keyword groups used for labeling:**
- Scam language (lottery, prize, winner, claim urgently)
- Urgency & pressure (urgent, act now, within 24 hours)
- Financial crime (offshore account, tax evasion, hidden assets)
- SEC violations (revenue recognition, channel stuffing, restatement)
- Account fraud (verify account, unauthorized transfer, without approval)

> ⚠️ **Note:** Because labels are rule-based and not from verified real-world fraud cases, model metrics reflect pattern-matching performance. Results should always be reviewed by a qualified compliance officer.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| LLM | Groq API (Llama 3.1 8B) |
| PDF Generation | ReportLab |
| Data | pandas, numpy |
| Visualisation | matplotlib |

---

## 📋 Sample CSV Format for Batch Upload

To use the batch analysis feature, upload a CSV with a single column named `text`:

```csv
text
Wire transfer of $47000 to offshore account outside business hours without approval.
Congratulations! You have been selected as a winner. Claim your $25000 prize urgently.
Monthly payroll disbursement to 42 employees. Standard ACH batch completed.
Company declared bankruptcy and failed to disclose material liabilities in SEC 10-K filing.
```

A sample test file `sample_batch_test.csv` is included in the repository.

---

## 🔒 Security Notes

- **Never commit** your `.streamlit/secrets.toml` file to GitHub
- Add `.streamlit/secrets.toml` to your `.gitignore`
- Rotate your API key immediately if accidentally exposed

---

## 📁 .gitignore Recommendation

Add the following to your `.gitignore` before pushing:

```
.streamlit/secrets.toml
__pycache__/
*.pyc
.env
*.csv
```

> You may want to keep `final_labeled_fraud_dataset.csv` and `Final_Dataset.csv` out of GitHub if they contain sensitive data.

---

## 👨‍💻 Author

**Aadish**
Final Year Project — NLP & Machine Learning
Proactive Financial Fraud Detection

---

## 📄 License

This project is submitted as an academic final year project. All rights reserved.

---

> Built with ❤️ using Streamlit, scikit-learn, SHAP, and Groq AI
