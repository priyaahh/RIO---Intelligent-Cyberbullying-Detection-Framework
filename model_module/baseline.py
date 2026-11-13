print("baseline.py started")

import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Ensure local imports work regardless of current working directory
sys.path.append(str(Path(__file__).resolve().parent))
from config import *

# Load datasets
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Train TF-IDF vectorizer and Logistic Regression baseline
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
Xtr = vec.fit_transform(train[TEXT_COL].fillna(""))
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
clf.fit(Xtr, train[LABEL_COL])

results = {}

for name, df in [("val", val), ("test", test)]:
    y_true = df[LABEL_COL]
    y_pred = clf.predict(vec.transform(df[TEXT_COL].fillna("")))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    rep = classification_report(y_true, y_pred, target_names=LABELS, digits=4, output_dict=True)

    print(f"\n===== {name.upper()} =====")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

    results[name] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "threat_f1": round(float(rep["Threat"]["f1-score"]), 4),
        "per_class": {
            label: {
                "precision": round(float(rep[label]["precision"]), 4),
                "recall": round(float(rep[label]["recall"]), 4),
                "f1": round(float(rep[label]["f1-score"]), 4),
                "support": int(rep[label]["support"])
            }
            for label in LABELS
        }
    }

# Save results to results/baseline_results.json
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
baseline_file = RESULTS_DIR / "baseline_results.json"
with open(baseline_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved baseline results to: {baseline_file}")

# Per-source check for the domain-shortcut risk
test["pred"] = clf.predict(vec.transform(test[TEXT_COL].fillna("")))
if "original_dataset" in test.columns:
    print("\nPer-source breakdown on test set:")
    for src, g in test.groupby("original_dataset"):
        print(src, "macro-F1:", round(f1_score(g[LABEL_COL], g["pred"], average="macro"), 3), "n =", len(g))
