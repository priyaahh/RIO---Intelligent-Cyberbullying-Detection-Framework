import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_model")
if not os.path.isdir(PATH):
    raise FileNotFoundError(f"saved_model folder not found at {os.path.abspath(PATH)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH).to(device).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}


def predict(text, return_all_scores=False):
    """Classify one string, or a list of strings. Returns a dict (or a list of dicts)."""
    single = isinstance(text, str)
    texts = [text] if single else list(text)

    # empty or blank input: skip the model
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Input text must be a non-empty string.")

    inputs = tok(texts, truncation=True, max_length=128,
                 padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).cpu()

    results = []
    for p in probs:
        i = int(p.argmax())
        r = {
            "label": id2label[i],
            "confidence": round(float(p[i]), 3),
            "is_cyberbullying": id2label[i] != "Not Cyberbullying",
        }
        if return_all_scores:
            r["scores"] = {id2label[j]: round(float(p[j]), 3) for j in range(len(p))}
        results.append(r)

    return results[0] if single else results


if __name__ == "__main__":
    samples = sys.argv[1:] or [
        "you are so stupid",
        "have a great day",
        "i will find you and hurt you",
    ]
    for s, r in zip(samples, predict(samples)):
        print(s, "->", r)