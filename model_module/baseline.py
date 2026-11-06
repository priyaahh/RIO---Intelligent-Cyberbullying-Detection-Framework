print("baseline.py started")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from config import *

train = pd.read_csv(DATA_DIR / "train.csv")
val   = pd.read_csv(DATA_DIR / "val.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
Xtr = vec.fit_transform(train[TEXT_COL])
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(Xtr, train[LABEL_COL])

for name, df in [("VAL", val), ("TEST", test)]:
    pred = clf.predict(vec.transform(df[TEXT_COL]))
    print(f"\n===== {name} =====")
    print(classification_report(df[LABEL_COL], pred, target_names=LABELS, digits=3))
    print(confusion_matrix(df[LABEL_COL], pred))
    print("Macro-F1:", f1_score(df[LABEL_COL], pred, average="macro"))

# Per-source check for the domain-shortcut risk
test["pred"] = clf.predict(vec.transform(test[TEXT_COL]))
for src, g in test.groupby("original_dataset"):
    print(src, "macro-F1:", round(f1_score(g[LABEL_COL], g["pred"], average="macro"), 3), "n =", len(g))