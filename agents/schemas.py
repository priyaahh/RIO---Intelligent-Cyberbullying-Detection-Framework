"""
agents/schemas.py

Pydantic v2 data models for inter-agent communication.
Every agent consumes and produces this typed message contract.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """
    Typed message passing through all stages of the multi-agent pipeline:
    Ingestion -> Preprocessing -> Classification -> Reporting
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    cleaned_text: Optional[str] = None

    # Scores per model: {"bert": {...}, "deberta": {...}, "ensemble": {...}}
    per_model_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Final outputs
    final_labels: List[str] = Field(default_factory=list)
    primary_label: str = "Not Cyberbullying"
    confidence: float = 0.0
    is_cyberbullying: bool = False
    severity: str = "LOW"

    # Execution telemetry and state
    status: str = "PENDING"
    model_flags: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    latency_ms: float = 0.0
    error: Optional[str] = None


class BatchInput(BaseModel):
    texts: List[str]


class SingleInput(BaseModel):
    text: str

