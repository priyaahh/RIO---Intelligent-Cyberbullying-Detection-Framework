
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from database import create_database, save_analysis, get_analysis_history

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "deberta_v3"

LABELS = [
    "Toxic",
    "Severe Toxic",
    "Obscene",
    "Threat",
    "Insult",
    "Identity Hate",
]

st.set_page_config(
    page_title="RIO | Cyberbullying Detection",
    page_icon="🛡️",
    layout="centered",
)

# Initialize SQLite database
create_database()

st.title("🛡️ RIO")
st.subheader("Intelligent Cyberbullying Detection Framework")
st.write("Enter a comment below to analyse its toxicity categories.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    return tokenizer, model


if not MODEL_PATH.exists():
    st.error("Trained model not found. Please train the model first.")
    st.stop()

with st.spinner("Loading RIO model..."):
    tokenizer, model = load_model()

comment = st.text_area(
    "Enter a comment",
    height=150,
    placeholder="Type or paste a comment here...",
)

if st.button("Analyse Comment", type="primary"):
    if not comment.strip():
        st.warning("Please enter a comment first.")
    else:
        with st.spinner("Analysing comment..."):
            inputs = tokenizer(
                comment,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.sigmoid(outputs.logits)[0].tolist()

        st.subheader("Analysis Results")

        detected = [
            LABELS[i]
            for i, probability in enumerate(probabilities)
            if probability >= 0.5
        ]

        if detected:
            prediction = ", ".join(detected)
            st.error("Potential harmful content detected.")
            st.write("Detected categories:", prediction)
            confidence = max(probabilities)
        else:
            prediction = "No categories detected"
            st.success("No categories detected at the current threshold.")
            confidence = max(probabilities)

        # Save analysis in SQLite
        save_analysis(
            comment,
            prediction,
            confidence
        )

        st.success("Analysis saved successfully in RIO database.")

        st.write("Category probabilities")
        for label, probability in zip(LABELS, probabilities):
            st.write(f"**{label}:** {probability:.2%}")
            st.progress(probability)

        st.caption(
            "RIO is a prototype. Predictions may be inaccurate "
            "and should be reviewed by a human."
        )

# Analysis History
st.divider()
st.subheader("📊 Analysis History")

history = get_analysis_history()

if history:
    st.write(f"Total analyses stored: {len(history)}")

    for record in history:
        with st.expander(f"Analysis #{record[0]} - {record[4]}"):
            st.write("**Comment:**", record[1])
            st.write("**Prediction:**", record[2])
            st.write("**Confidence:**", f"{record[3]:.2%}")
            st.write("**Date:**", record[4])
else:
    st.info("No analysis history available yet.")

