from pathlib import Path

# Resolves correctly no matter where you run the script from
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml_preprocessing" / "datasets" / "processed"
TEXT_COL, LABEL_COL = "cleaned_text", "label_id"

LABELS = ["not_cyberbullying", "insult", "threat", "hate_speech", "harassment"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {l: i for i, l in ID2LABEL.items()}

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
SEED = 42