from pathlib import Path

# Resolves correctly no matter where you run the script from
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_5class"
RESULTS_DIR = ROOT / "results"
TEXT_COL, LABEL_COL = "cleaned_text", "label_id"

LABELS = ["Not Cyberbullying", "Insult", "Threat", "Hate Speech", "Harassment"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {l: i for i, l in ID2LABEL.items()}

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
SEED = 42
