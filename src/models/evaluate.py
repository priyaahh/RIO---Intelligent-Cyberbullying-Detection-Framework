
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "deberta_v3"
VALIDATION_FILE = ROOT / "data" / "processed" / "validation_processed.csv"
REPORT_DIR = ROOT / "reports"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    df = pd.read_csv(VALIDATION_FILE)

    if args.max_samples > 0:
        df = df.head(args.max_samples)

    texts = df["comment_text"].fillna("").astype(str).tolist()
    y_true = df[LABELS].to_numpy(dtype=np.int32)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), 4):
            batch_texts = texts[start:start + 4]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            logits = model(**inputs).logits
            probabilities = torch.sigmoid(logits)
            batch_predictions = (probabilities >= 0.5).int()

            predictions.extend(batch_predictions.cpu().numpy())

            print(
                f"Evaluated {min(start + 4, len(texts))}/{len(texts)} comments",
                end="\r",
            )

    y_pred = np.asarray(predictions, dtype=np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    subset_accuracy = accuracy_score(y_true, y_pred)
    overall_precision, overall_recall, overall_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
    )

    results = pd.DataFrame({
        "label": LABELS,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(REPORT_DIR / "label_metrics.csv", index=False)

    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    confusion_rows = []
    for label, matrix in zip(LABELS, confusion_matrices):
        tn, fp, fn, tp = matrix.ravel()
        confusion_rows.append({
            "label": label,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        })

    pd.DataFrame(confusion_rows).to_csv(
        REPORT_DIR / "confusion_matrix.csv",
        index=False,
    )

    with open(REPORT_DIR / "evaluation_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"Comments evaluated: {len(texts)}\n")
        file.write(f"Subset accuracy: {subset_accuracy:.4f}\n")
        file.write(f"Micro precision: {overall_precision:.4f}\n")
        file.write(f"Micro recall: {overall_recall:.4f}\n")
        file.write(f"Micro F1-score: {overall_f1:.4f}\n\n")
        file.write(results.to_string(index=False))

    print("\n\nEVALUATION COMPLETE")
    print(f"Comments evaluated: {len(texts)}")
    print(f"Subset accuracy: {subset_accuracy:.4f}")
    print(f"Micro precision: {overall_precision:.4f}")
    print(f"Micro recall: {overall_recall:.4f}")
    print(f"Micro F1-score: {overall_f1:.4f}")
    print("\nPer-label metrics:")
    print(results.to_string(index=False))
    print(f"\nReports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
