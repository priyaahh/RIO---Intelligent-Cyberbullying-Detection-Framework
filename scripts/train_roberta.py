"""
scripts/train_roberta.py

Fine-tunes RoBERTa-base (your model) on processed/train.csv for the 5 shared cyberbullying categories.
Saves model checkpoint to models/roberta for loading by the ClassificationAgent.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.config import (
    TRAIN_CSV,
    VAL_CSV,
    ROBERTA_MODEL_PATH,
    ROBERTA_MODEL_NAME,
    SHARED_LABELS,
    LABEL2ID,
    ID2LABEL,
    MAX_LENGTH,
)


def train(epochs: int = 3, batch_size: int = 16, lr: float = 2e-5, max_samples: int = None):
    print("=" * 70)
    print("TRAINING ROBERTA-BASE (YOUR MODEL) ON PROCESSED DATASET")
    print("=" * 70)

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as e:
        print(f"Error: Required libraries not found ({e}). Install via requirements.txt.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {ROBERTA_MODEL_NAME}")
    print(f"Output directory: {ROBERTA_MODEL_PATH}")

    # Load datasets
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples // 4)

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # Initialize tokenizer and model
    tok = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME,
        num_labels=len(SHARED_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # Prepare datasets
    def prep(df):
        ds = Dataset.from_pandas(df[["cleaned_text", "label_id"]].rename(columns={"label_id": "labels"}))
        ds = ds.map(lambda b: tok(b["cleaned_text"], truncation=True, max_length=MAX_LENGTH), batched=True)
        return ds.remove_columns(["cleaned_text"])

    print("Tokenizing datasets...")
    train_ds = prep(train_df)
    val_ds = prep(val_df)
    collator = DataCollatorWithPadding(tok)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=str(ROBERTA_MODEL_PATH / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting RoBERTa training...")
    trainer.train()

    # Save final model
    ROBERTA_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ROBERTA_MODEL_PATH))
    tok.save_pretrained(str(ROBERTA_MODEL_PATH))
    print(f"\n[DONE] RoBERTa model successfully saved to: {ROBERTA_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug subset")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_samples=args.max_samples)

