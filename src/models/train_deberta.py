
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "microsoft/deberta-v3-base"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "data" / "processed" / "train_processed.csv"
VAL_FILE = ROOT / "data" / "processed" / "validation_processed.csv"
OUTPUT_DIR = ROOT / "models" / "deberta_v3"


class CommentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.comments = dataframe["comment_text"].astype(str).tolist()
        self.labels = dataframe[LABEL_COLUMNS].to_numpy(dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.comments[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index]),
        }


def train(args):
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError(
            "Processed datasets not found. Run src/data/preprocess.py first."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    if args.max_train_samples:
        train_df = train_df.head(args.max_train_samples)

    if args.max_val_samples:
        val_df = val_df.head(args.max_val_samples)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )
    model.to(device)

    train_dataset = CommentDataset(train_df, tokenizer, args.max_length)
    val_dataset = CommentDataset(val_df, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # Calculate class weights to address label imbalance
    label_values = train_df[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    positive_counts = label_values.sum(axis=0)
    negative_counts = len(label_values) - positive_counts

    pos_weights = negative_counts / np.maximum(positive_counts, 1)
    pos_weights = np.clip(pos_weights, 1.0, 20.0)

    pos_weights = torch.tensor(
        pos_weights, dtype=torch.float32
    ).to(device)

    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    print("Positive class weights:")
    for label, weight in zip(LABEL_COLUMNS, pos_weights.cpu().numpy()):
        print(f"{label}: {weight:.2f}")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_number, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_function(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if batch_number % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = loss_function(outputs.logits, labels)
                total_val_loss += loss.item()

        print(f"\nEpoch {epoch + 1} completed")
        print(f"Average training loss: {total_train_loss / len(train_loader):.4f}")
        print(f"Average validation loss: {total_val_loss / len(val_loader):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel and tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3 for multi-label cyberbullying detection."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    train(parser.parse_args())
