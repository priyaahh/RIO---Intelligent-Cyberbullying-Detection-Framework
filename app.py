
from pathlib import Path

import streamlit as st

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except (ImportError, OSError):
    torch = None

from database import create_database, save_analysis, get_analysis_history

# --- [AGENT LAYER EDIT] Import shared configuration and OrchestratorAgent ---
from agents.config import SHARED_LABELS, DEBERTA_MODEL_PATH
from agents.orchestrator import OrchestratorAgent

ROOT = Path(__file__).resolve().parent
MODEL_PATH = DEBERTA_MODEL_PATH
LABELS = SHARED_LABELS  # Single source of truth from agents.config
# ---------------------------------------------------------------------------

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

# --- [AGENT LAYER EDIT] Initialize OrchestratorAgent instead of hard stopping ---
@st.cache_resource
def get_orchestrator():
    return OrchestratorAgent(mock_mode_if_missing=True, save_to_db=True)

orchestrator = get_orchestrator()

# Status notification
if not MODEL_PATH.exists():
    st.info("ℹ️ Local DeBERTa/BERT weights not found in repo (they live on shared Drive). Multi-Agent pipeline is active in verification fallback mode.")
# ---------------------------------------------------------------------------------

comment = st.text_area(
    "Enter a comment",
    height=150,
    placeholder="Type or paste a comment here...",
)

if st.button("Analyse Comment", type="primary"):
    if not comment.strip():
        st.warning("Please enter a comment first.")
    else:
        with st.spinner("Multi-Agent pipeline analysing comment..."):
            # Execute full multi-agent pipeline: Ingestion -> Preprocessing -> Classification -> Reporting
            result = orchestrator.process(comment)

        st.subheader("Analysis Results")

        if result.is_cyberbullying:
            prediction = ", ".join(result.final_labels)
            st.error(f"Potential harmful content detected! (Severity: {result.severity})")
            st.write("**Detected categories:**", prediction)
        else:
            prediction = "Not Cyberbullying"
            st.success("No harmful categories detected. Content appears safe.")

        st.write(f"**Confidence:** {result.confidence:.2%}")
        st.write(f"**Pipeline Latency:** {result.latency_ms:.1f} ms")

        # Database logging is automatically performed by ReportingAgent inside orchestrator.process
        st.success("Analysis saved successfully in RIO database.")

        st.write("### Category probabilities (Ensemble)")
        ensemble_scores = result.per_model_scores.get("ensemble", {})
        for label in LABELS:
            prob = ensemble_scores.get(label, 0.0)
            st.write(f"**{label}:** {prob:.2%}")
            st.progress(min(1.0, max(0.0, prob)))

        # Show per-model breakdown
        with st.expander("🔍 View Per-Model Predictions (BERT vs DeBERTa vs RoBERTa)"):
            st.write("**BERT Scores (Teammate 1 & 2):**", result.per_model_scores.get("bert", {}))
            st.write("**DeBERTa-v3 Scores (Teammate 3 & 4 - Mapped):**", result.per_model_scores.get("deberta", {}))
            st.write("**RoBERTa Scores (Your Model):**", result.per_model_scores.get("roberta", {}))

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

