import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("final_labeled_fraud_dataset.csv")

X_text = df["filing_text"]
y = df["fraud"]


tfidf = TfidfVectorizer(stop_words="english", max_features=500)
X_text_vec = tfidf.fit_transform(X_text)


model = LogisticRegression(max_iter=1000)
model.fit(X_text_vec, y)


print("\n--- Real-Time SEC Filing Fraud Detection ---")
user_text = input("Enter transaction / filing text: ")

user_vec = tfidf.transform([user_text])
fraud_prob = model.predict_proba(user_vec)[0][1]


feature_names = tfidf.get_feature_names_out()
coeffs = model.coef_[0]

top_words = sorted(
    zip(feature_names, coeffs),
    key=lambda x: abs(x[1]),
    reverse=True
)[:5]

print("\nFraud Probability:", round(fraud_prob, 2))

if fraud_prob > 0.6:
    print("Risk Level: HIGH ⚠️")
elif fraud_prob > 0.3:
    print("Risk Level: MEDIUM")
else:
    print("Risk Level: LOW ✅")

print("\nSuspicious Keywords:")
for word, _ in top_words:
    print("-", word)
