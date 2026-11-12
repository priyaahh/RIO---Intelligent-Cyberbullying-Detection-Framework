"""
agents/reporting.py

ReportingAgent:
- Synthesizes final classification outputs, per-model scores, and confidence metrics.
- Assigns severity ratings (LOW, MEDIUM, HIGH) based on detected category.
- Records analysis history directly into the SQLite database via database.py.
- Sets `status="REPORTED"`.
"""

import sys
from pathlib import Path
from datetime import datetime

from .config import SEVERITY_THRESHOLDS
from .schemas import AgentMessage

# Reference project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from database import save_analysis
except ImportError:
    save_analysis = None


class ReportingAgent:
    """
    Produces structured reports and persists analysis to SQLite.
    """

    def __init__(self, name: str = "ReportingAgent", save_to_db: bool = True):
        self.name = name
        self.save_to_db = save_to_db

    def run(self, message: AgentMessage) -> AgentMessage:
        """Processes and logs the completed message analysis."""
        if message.status == "ERROR":
            return message

        # 1. Determine severity
        message.severity = SEVERITY_THRESHOLDS.get(message.primary_label, "LOW")

        # 2. Update timestamp if missing
        if not message.timestamp:
            message.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3. Save to SQLite database
        if self.save_to_db and save_analysis is not None:
            try:
                prediction_str = ", ".join(message.final_labels) if message.final_labels else message.primary_label
                save_analysis(
                    comment=message.text,
                    prediction=prediction_str,
                    confidence=message.confidence
                )
            except Exception as exc:
                message.model_flags.append(f"db_save_warning: {exc}")

        message.status = "REPORTED"
        return message

