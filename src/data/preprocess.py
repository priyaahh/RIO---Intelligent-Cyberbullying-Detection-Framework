
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "jigsaw-toxic-comment-classification-challenge"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def load_dataset():
    """Locate and load the original Jigsaw training CSV."""

    train_files = [
        path for path in RAW_DIR.rglob("train.csv")
        if path.is_file()
    ]

    if len(train_files) != 1:
        raise FileNotFoundError(
            f"Expected one train.csv file, found {len(train_files)}"
        )

    print(f"Loading dataset: {train_files[0]}")

    return pd.read_csv(train_files[0])


def preprocess_data(df):
    """Perform safe preprocessing and validate labels."""

    required_columns = ["id", "comment_text"] + LABELS

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    # Remove rows with missing comments
    df = df.dropna(subset=["comment_text"])

    # Preserve original language, punctuation and casing.
    df["comment_text"] = (
        df["comment_text"]
        .astype(str)
        .str.strip()
    )

    # Remove empty comments
    df = df[df["comment_text"] != ""]

    # Validate multilabel targets
    for label in LABELS:
        if not df[label].isin([0, 1]).all():
            raise ValueError(f"Invalid values in {label}")

    df[LABELS] = df[LABELS].astype("int8")

    return df.reset_index(drop=True)


def split_dataset(df):
    """Create multilabel-stratified train/validation splits."""

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    X = df["comment_text"].values
    y = df[LABELS].values

    train_indices, val_indices = next(
        splitter.split(X, y)
    )

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    return train_df, val_df


def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_dataset()

    print("\nOriginal dataset:", df.shape)

    # Preprocess
    df = preprocess_data(df)

    print("After preprocessing:", df.shape)

    # Split
    train_df, val_df = split_dataset(df)

    # Save
    train_path = PROCESSED_DIR / "train_processed.csv"
    val_path = PROCESSED_DIR / "validation_processed.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("\nPREPROCESSING COMPLETED")
    print("-----------------------")

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))

    print("\nTraining label distribution:")
    print(train_df[LABELS].sum())

    print("\nValidation label distribution:")
    print(val_df[LABELS].sum())

    print("\nSaved files:")
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
