import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set paths
sys.path.append(str(Path(__file__).resolve().parent))
from config import ROOT, DATA_DIR, RESULTS_DIR, LABELS, TEXT_COL, LABEL_COL

MODEL_PATH = ROOT / "saved_model"
VAL_PROBS_PATH = RESULTS_DIR / "val_probs.npy"
TEST_PROBS_PATH = RESULTS_DIR / "test_probs.npy"
BERT_RESULTS_FILE = RESULTS_DIR / "bert_results.json"

BATCH_SIZE = 64
MAX_LENGTH = 128
THREAT_CLASS_IDX = 2  # Threat label_id is 2


def get_probabilities(df, probs_file, model=None, tokenizer=None, device="cpu"):
    """Load cached probabilities if available, otherwise compute and cache them."""
    if probs_file.exists():
        print(f"Loading cached probabilities from: {probs_file}")
        return np.load(probs_file)

    print(f"Computing probabilities for {len(df)} samples...")
    texts = df[TEXT_COL].fillna("").astype(str).tolist()

    # Sort indices by sequence length to minimize batch padding overhead on CPU
    sorted_idx = np.argsort([len(t.split()) for t in texts])
    sorted_texts = [texts[i] for i in sorted_idx]

    all_sorted_probs = []
    total_batches = (len(sorted_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(0, len(sorted_texts), BATCH_SIZE):
        batch_texts = sorted_texts[b_idx : b_idx + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_sorted_probs.append(probs)

        current_batch = b_idx // BATCH_SIZE + 1
        if current_batch % 25 == 0 or current_batch == total_batches:
            print(f"  Batch {current_batch}/{total_batches} completed")

    # Invert permutation back to original dataset order
    sorted_probs_arr = np.concatenate(all_sorted_probs, axis=0)
    inv_idx = np.empty_like(sorted_idx)
    inv_idx[sorted_idx] = np.arange(len(sorted_idx))
    probs_arr = sorted_probs_arr[inv_idx]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(probs_file, probs_arr)
    print(f"Saved probabilities to: {probs_file}")
    return probs_arr


def apply_threat_threshold(probs, threshold):
    """
    Predict Threat (index 2) only if its probability >= threshold,
    otherwise choose the highest among the remaining classes.
    """
    probs_other = probs.copy()
    probs_other[:, THREAT_CLASS_IDX] = -1.0
    best_other = probs_other.argmax(axis=1)

    threat_mask = probs[:, THREAT_CLASS_IDX] >= threshold
    return np.where(threat_mask, THREAT_CLASS_IDX, best_other)


def compute_metrics(y_true, y_pred):
    """Compute accuracy, macro F1, and per-class precision/recall/F1."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    rep = classification_report(y_true, y_pred, target_names=LABELS, digits=4, output_dict=True)

    per_class = {
        label: {
            "precision": round(float(rep[label]["precision"]), 4),
            "recall": round(float(rep[label]["recall"]), 4),
            "f1": round(float(rep[label]["f1-score"]), 4),
            "support": int(rep[label]["support"]),
        }
        for label in LABELS
    }
    threat_f1 = per_class["Threat"]["f1"]

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "threat_f1": round(threat_f1, 4),
        "per_class": per_class,
    }


def main():
    print("evaluate_bert.py started")
    torch.set_num_threads(8)

    # Load datasets
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    y_val = val_df[LABEL_COL].values
    y_test = test_df[LABEL_COL].values

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer only if probabilities are not already cached
    model, tokenizer = None, None
    if not VAL_PROBS_PATH.exists() or not TEST_PROBS_PATH.exists():
        print(f"Loading BERT model and tokenizer from: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

    val_probs = get_probabilities(val_df, VAL_PROBS_PATH, model, tokenizer, device)
    test_probs = get_probabilities(test_df, TEST_PROBS_PATH, model, tokenizer, device)

    # 1. Validation: Plain argmax evaluation
    val_preds_argmax = val_probs.argmax(axis=1)
    val_argmax_metrics = compute_metrics(y_val, val_preds_argmax)
    print("\n===== VAL (Plain Argmax) =====")
    print(f"Accuracy: {val_argmax_metrics['accuracy']:.4f} | Macro-F1: {val_argmax_metrics['macro_f1']:.4f} | Threat-F1: {val_argmax_metrics['threat_f1']:.4f}")

    # 2. Validation: Search Threat threshold from 0.10 to 0.95 (step 0.05)
    thresholds = [round(float(t), 2) for t in np.arange(0.10, 0.96, 0.05)]
    search_results = []
    best_t = 0.50
    best_val_macro_f1 = -1.0

    print("\n--- Tuning Threat Threshold on Validation Set ---")
    for t in thresholds:
        preds = apply_threat_threshold(val_probs, t)
        m_f1 = float(f1_score(y_val, preds, average="macro"))
        t_f1 = float(f1_score(y_val, preds, labels=[THREAT_CLASS_IDX], average="macro"))
        search_results.append({"threshold": t, "macro_f1": round(m_f1, 4), "threat_f1": round(t_f1, 4)})

        if m_f1 > best_val_macro_f1:
            best_val_macro_f1 = m_f1
            best_t = t

    print(f"Optimal Threat Threshold on Validation: t = {best_t:.2f} (Val Macro-F1: {best_val_macro_f1:.4f})")

    val_tuned_preds = apply_threat_threshold(val_probs, best_t)
    val_tuned_metrics = compute_metrics(y_val, val_tuned_preds)

    # 3. Test Evaluation ONCE with chosen best_t
    print("\n===== TEST EVALUATION =====")
    test_argmax_preds = test_probs.argmax(axis=1)
    test_argmax_metrics = compute_metrics(y_test, test_argmax_preds)

    test_tuned_preds = apply_threat_threshold(test_probs, best_t)
    test_tuned_metrics = compute_metrics(y_test, test_tuned_preds)

    print("\n--- TEST (Plain Argmax) ---")
    print(classification_report(y_test, test_argmax_preds, target_names=LABELS, digits=4))
    print(f"Accuracy: {test_argmax_metrics['accuracy']:.4f} | Macro-F1: {test_argmax_metrics['macro_f1']:.4f} | Threat-F1: {test_argmax_metrics['threat_f1']:.4f}")

    print(f"\n--- TEST (Tuned Threat Threshold t={best_t:.2f}) ---")
    print(classification_report(y_test, test_tuned_preds, target_names=LABELS, digits=4))
    print(f"Accuracy: {test_tuned_metrics['accuracy']:.4f} | Macro-F1: {test_tuned_metrics['macro_f1']:.4f} | Threat-F1: {test_tuned_metrics['threat_f1']:.4f}")

    # Save to results/bert_results.json
    bert_results = {
        "chosen_threat_threshold": best_t,
        "val": {
            "argmax": val_argmax_metrics,
            "tuned_threshold": val_tuned_metrics,
            "threshold_search": search_results,
        },
        "test": {
            "argmax": test_argmax_metrics,
            "tuned_threshold": test_tuned_metrics,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BERT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(bert_results, f, indent=2)
    print(f"\nSaved BERT evaluation results to: {BERT_RESULTS_FILE}")


if __name__ == "__main__":
    main()
