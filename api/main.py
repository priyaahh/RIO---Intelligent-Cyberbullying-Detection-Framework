"""
api/main.py

FastAPI microservice exposing the multi-agent cyberbullying detection pipeline.
Endpoints:
  - GET  /health  : System health, model availability status, and label schema.
  - POST /analyze : Submits text to the OrchestratorAgent, returning full agent results.
  - GET  /history : Retrieves recent SQLite analysis records.
"""

import sys
from pathlib import Path
from typing import List, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure RIO project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.config import SHARED_LABELS, BERT_MODEL_PATH, DEBERTA_MODEL_PATH, ROBERTA_MODEL_PATH
from agents.schemas import AgentMessage, SingleInput, BatchInput
from agents.orchestrator import OrchestratorAgent
from database import get_analysis_history

# Initialize FastAPI application
app = FastAPI(
    title="RIO Multi-Agent Cyberbullying Detection API",
    description="Multi-agent inference service integrating BERT, DeBERTa-v3, and Ensemble pipelines.",
    version="1.0.0"
)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Orchestrator instance
orchestrator = OrchestratorAgent(mock_mode_if_missing=True, save_to_db=True)


@app.get("/health", tags=["System"])
def health():
    """
    Returns service health and model availability flags.
    """
    return {
        "status": "healthy",
        "service": "RIO Multi-Agent Detection API",
        "bert_weights_found": BERT_MODEL_PATH.is_dir(),
        "deberta_weights_found": DEBERTA_MODEL_PATH.is_dir(),
        "roberta_weights_found": ROBERTA_MODEL_PATH.is_dir(),
        "bert_active": orchestrator.classification.bert_available,
        "deberta_active": orchestrator.classification.deberta_available,
        "roberta_active": orchestrator.classification.roberta_available,
        "mock_mode_active": orchestrator.classification.mock_mode,
        "shared_labels": SHARED_LABELS,
    }


@app.post("/analyze", response_model=AgentMessage, tags=["Analysis"])
def analyze_comment(payload: SingleInput):
    """
    Accepts a single comment and runs it through the full multi-agent pipeline:
    Ingestion -> Preprocessing -> Classification (BERT + DeBERTa + Ensemble) -> Reporting.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    result = orchestrator.process(payload.text)
    if result.status == "ERROR":
        raise HTTPException(status_code=500, detail=result.error or "Pipeline processing error.")

    return result


@app.post("/analyze/batch", response_model=List[AgentMessage], tags=["Analysis"])
def analyze_batch(payload: BatchInput):
    """
    Runs a list of comments through the multi-agent pipeline.
    """
    if not payload.texts:
        raise HTTPException(status_code=400, detail="List of texts cannot be empty.")

    return orchestrator.process_batch(payload.texts)


@app.get("/history", tags=["History"])
def get_history():
    """
    Fetches stored analysis history directly from SQLite.
    """
    try:
        records = get_analysis_history()
        return {"count": len(records), "records": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

