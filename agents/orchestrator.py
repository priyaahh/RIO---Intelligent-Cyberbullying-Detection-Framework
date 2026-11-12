"""
agents/orchestrator.py

OrchestratorAgent:
- Coordinates the end-to-end multi-agent detection pipeline.
- Automatically routes messages sequentially through:
    Stage 1: IngestionAgent (Validation & typing)
    Stage 2: PreprocessingAgent (Text cleaning & normalization)
    Stage 3: ClassificationAgent (BERT + DeBERTa + Tri-Ensemble scoring)
    Stage 4: ReportingAgent (Severity assignment & SQLite persistence)
- Provides per-stage error handling, latency monitoring, and graceful fallback
  when model checkpoints are not available locally.
"""

import time
import logging
from typing import Any, List, Optional, Union
from pathlib import Path

from .schemas import AgentMessage
from .ingestion import IngestionAgent
from .preprocessing import PreprocessingAgent
from .classification import ClassificationAgent
from .reporting import ReportingAgent

logger = logging.getLogger("OrchestratorAgent")


class OrchestratorAgent:
    """
    Central controller coordinating inter-agent data flow.
    """

    def __init__(
        self,
        name: str = "OrchestratorAgent",
        mock_mode_if_missing: bool = True,
        save_to_db: bool = True,
    ):
        self.name = name
        self.ingestion = IngestionAgent()
        self.preprocessing = PreprocessingAgent()
        self.classification = ClassificationAgent(mock_mode_if_missing=mock_mode_if_missing)
        self.reporting = ReportingAgent(save_to_db=save_to_db)

    def process(self, input_item: Union[str, AgentMessage, dict]) -> AgentMessage:
        """
        Executes the entire 4-stage multi-agent pipeline on a single item.
        """
        t0 = time.perf_counter()

        # Wrap into AgentMessage if needed
        if isinstance(input_item, AgentMessage):
            msg = input_item
        elif isinstance(input_item, dict):
            msg = AgentMessage(text=input_item.get("text", ""))
        else:
            msg = AgentMessage(text=str(input_item))

        try:
            # Stage 1: Ingestion & Validation
            msg = self.ingestion.run(msg)
            if msg.status == "ERROR":
                msg.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return msg

            # Stage 2: Preprocessing
            msg = self.preprocessing.run(msg)
            if msg.status == "ERROR":
                msg.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return msg

            # Stage 3: Multi-Model Classification & Ensemble
            msg = self.classification.run(msg)
            if msg.status == "ERROR":
                msg.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return msg

            # Stage 4: Reporting & Persistence
            msg = self.reporting.run(msg)

        except Exception as exc:
            logger.error(f"Pipeline error processing message {msg.id}: {exc}", exc_info=True)
            msg.status = "ERROR"
            msg.error = str(exc)

        msg.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return msg

    def process_batch(self, inputs: List[Union[str, AgentMessage]]) -> List[AgentMessage]:
        """Runs the pipeline over a list of messages."""
        return [self.process(item) for item in inputs]

    def run_csv(
        self,
        csv_path: Optional[Path] = None,
        max_rows: Optional[int] = None
    ) -> List[AgentMessage]:
        """
        Ingests a CSV dataset and streams it through the full multi-agent pipeline.
        """
        messages = self.ingestion._ingest_csv(csv_path, max_rows=max_rows)
        results = []
        for msg in messages:
            results.append(self.process(msg))
        return results

