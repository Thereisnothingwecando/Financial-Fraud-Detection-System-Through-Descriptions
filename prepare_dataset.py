"""
prepare_dataset.py
------------------
Run this ONCE before starting app.py to convert your raw
Final_Dataset.csv into final_labeled_fraud_dataset.csv
using a real rule-based labeling strategy.

Usage:
    python prepare_dataset.py
"""

import pandas as pd
import numpy as np
import re

INPUT_FILE  = "Final_Dataset.csv"
OUTPUT_FILE = "final_labeled_fraud_dataset.csv"

# ---------------------------------------------------------------
# Load
# ---------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
print("Columns found:", list(df.columns))

# ---------------------------------------------------------------
# Clean
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

text_column   = "Fillings" if "Fillings" in df.columns else df.columns[0]
df["filing_text"] = df[text_column].apply(clean_text)

# ---------------------------------------------------------------
# Rule-based labeling
# ---------------------------------------------------------------
HIGH_CONFIDENCE = [
    "ponzi scheme", "pyramid scheme", "money laundering",
    "embezzlement", "misappropriation", "fictitious invoice",
    "fabricated revenue", "insider trading", "bribery",
    "shell company", "kickback", "falsified financial",
    "undisclosed related party", "material misstatement",
]

FRAUD_GROUPS = {
    "scam_language":    ["lottery", "prize", "winner", "congratulations",
                         "claim urgently", "selected", "reward", "gift"],
    "urgency_pressure": ["urgent", "immediately", "act now", "limited time",
                         "expires", "do not delay", "within 24 hours"],
    "financial_crime":  ["offshore account", "tax evasion", "unreported income",
                         "hidden assets", "undisclosed liability", "bankruptcy fraud",
                         "forged document", "altered record"],
    "sec_violations":   ["revenue recognition", "channel stuffing", "round tripping",
                         "improper disclosure", "material weakness", "restatement",
                         "audit failure", "going concern"],
    "account_fraud":    ["verify account", "confirm details", "provide credentials",
                         "unusual transaction", "unauthorized transfer",
                         "wire transfer outside", "without approval"],
}

def label_fraud(text: str) -> int:
    for signal in HIGH_CONFIDENCE:
        if signal in text:
            return 1
    for keywords in FRAUD_GROUPS.values():
        if sum(1 for kw in keywords if kw in text) >= 2:
            return 1
    return 0

df["fraud"] = df["filing_text"].apply(label_fraud)

# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------
fraud_count = df["fraud"].sum()
legit_count = len(df) - fraud_count
fraud_pct   = fraud_count / len(df) * 100

print(f"\n✅ Labeling complete")
print(f"   Total   : {len(df)}")
print(f"   Fraud   : {fraud_count}  ({fraud_pct:.1f}%)")
print(f"   Legit   : {legit_count}  ({100 - fraud_pct:.1f}%)")

if fraud_pct < 5:
    print("\n⚠️  Very few fraud samples — consider lowering threshold to hits >= 1")
elif fraud_pct > 60:
    print("\n⚠️  Dataset skews heavily toward fraud — review keyword groups")

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
df[["filing_text", "fraud"]].to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved: {OUTPUT_FILE}")