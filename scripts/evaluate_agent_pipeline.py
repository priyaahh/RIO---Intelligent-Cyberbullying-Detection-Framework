"""
scripts/evaluate_agent_pipeline.py

Evaluation script running the Multi-Agent pipeline over processed/test.csv.
Computes and prints:
  - Accuracy and Macro F1 for BERT
  - Accuracy and Macro F1 for DeBERTa-v3
  - Accuracy and Macro F1 for the Tri-Ensemble
Allows verification against the BERT team's benchmark (82.3% accuracy, 0.767 macro F1).
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.config import TEST_CSV, TEXT_COLUMN, LABEL_NAME_COLUMN, SHARED_LABELS
from agents.orchestrator import OrchestratorAgent


def evaluate(max_samples: int = 500, mock_mode: bool = True):
    print("=" * 70)
    print("RIO MULTI-AGENT PIPELINE EVALUATION ON PROCESSED/TEST.CSV")
    print("=" * 70)
    print(f"Dataset path: {TEST_CSV}")
    print(f"Max samples evaluated: {max_samples if max_samples > 0 else 'FULL DATASET'}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found at {TEST_CSV}")

    df = pd.read_csv(TEST_CSV)
    if max_samples > 0 and len(df) > max_samples:
        df = df.head(max_samples)

    print(f"Total evaluation samples: {len(df)}")
    print(f"Text column: '{TEXT_COLUMN}' | Label column: '{LABEL_NAME_COLUMN}'")

    y_true = df[LABEL_NAME_COLUMN].tolist()
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    # Initialize Orchestrator (save_to_db=False during evaluation)
    orchestrator = OrchestratorAgent(mock_mode_if_missing=mock_mode, save_to_db=False)

    print("\nModel Status:")
    print(f"  - BERT available:    {orchestrator.classification.bert_available}")
    print(f"  - DeBERTa available: {orchestrator.classification.deberta_available}")
    print(f"  - RoBERTa available: {orchestrator.classification.roberta_available}")
    print(f"  - Mock mode active:  {orchestrator.classification.mock_mode}")

    bert_preds = []
    deberta_preds = []
    roberta_preds = []
    ensemble_preds = []

    print("\nRunning Multi-Agent pipeline over samples...")
    for i, text in enumerate(texts):
        result = orchestrator.process(text)
        scores = result.per_model_scores

        # Extract predictions for each model path
        b_scores = scores.get("bert", {})
        d_scores = scores.get("deberta", {})
        r_scores = scores.get("roberta", {})
        e_scores = scores.get("ensemble", {})

        b_pred = max(b_scores, key=b_scores.get) if b_scores else "Not Cyberbullying"
        d_pred = max(d_scores, key=d_scores.get) if d_scores else "Not Cyberbullying"
        r_pred = max(r_scores, key=r_scores.get) if r_scores else "Not Cyberbullying"
        e_pred = max(e_scores, key=e_scores.get) if e_scores else "Not Cyberbullying"

        bert_preds.append(b_pred)
        deberta_preds.append(d_pred)
        roberta_preds.append(r_pred)
        ensemble_preds.append(e_pred)

        if (i + 1) % 100 == 0 or (i + 1) == len(texts):
            print(f"  Processed {i + 1}/{len(texts)} comments...")

    # Calculate metrics
    def calc_metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        return acc, macro_f1

    bert_acc, bert_f1 = calc_metrics(y_true, bert_preds)
    deberta_acc, deberta_f1 = calc_metrics(y_true, deberta_preds)
    roberta_acc, roberta_f1 = calc_metrics(y_true, roberta_preds)
    ensemble_acc, ensemble_f1 = calc_metrics(y_true, ensemble_preds)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model Path':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
    print("-" * 55)
    print(f"{'BERT (Teammate 1 & 2)':<25} | {bert_acc:<12.4f} | {bert_f1:<12.4f}")
    print(f"{'DeBERTa-v3 (Teammate 3 & 4)':<25} | {deberta_acc:<12.4f} | {deberta_f1:<12.4f}")
    print(f"{'RoBERTa (Your Model)':<25} | {roberta_acc:<12.4f} | {roberta_f1:<12.4f}")
    print(f"{'Tri-Ensemble (Agent)':<25} | {ensemble_acc:<12.4f} | {ensemble_f1:<12.4f}")
    print("=" * 70)

    print("\nBERT Team Benchmark Reference (Full Test Split with Trained Weights):")
    print("  Accuracy: 0.823 (82.3%) | Macro F1: 0.767")

    print("\nEnsemble Detailed Classification Report:")
    print(classification_report(y_true, ensemble_preds, target_names=SHARED_LABELS, zero_division=0))

    return {
        "bert": {"accuracy": bert_acc, "macro_f1": bert_f1},
        "deberta": {"accuracy": deberta_acc, "macro_f1": deberta_f1},
        "ensemble": {"accuracy": ensemble_acc, "macro_f1": ensemble_f1},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multi-Agent Pipeline")
    parser.add_argument("--max-samples", type=int, default=200, help="Number of test samples (default: 200, set 0 for all)")
    parser.add_argument("--mock", action="store_true", default=True, help="Enable mock fallback if weights missing")
    args = parser.parse_args()

    evaluate(max_samples=args.max_samples, mock_mode=args.mock)

