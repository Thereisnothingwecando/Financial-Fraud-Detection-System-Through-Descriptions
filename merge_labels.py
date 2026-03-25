import pandas as pd


filings = pd.read_csv("clean_fraud_filings.csv")


labels = pd.read_csv("kaggle_fraud_labels.csv")



filings = filings.sample(len(labels)).reset_index(drop=True)
filings["fraud"] = labels["fraud"]

filings.to_csv("final_labeled_fraud_dataset.csv", index=False)

print("Final labeled dataset saved as final_labeled_fraud_dataset.csv")
