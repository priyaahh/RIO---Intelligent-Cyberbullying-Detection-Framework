"""
agents/ingestion.py

IngestionAgent:
- Accepts a single comment, a list of comments, or a CSV path (default processed/test.csv).
- Validates text inputs (checks for empty strings, nulls, types).
- Wraps raw inputs into standardized AgentMessage objects.
"""

from pathlib import Path
from typing import Any, List, Union
import pandas as pd

from .config import TEST_CSV, TEXT_COLUMN, RAW_TEXT_FALLBACK
from .schemas import AgentMessage


class IngestionAgent:
    """
    Validates and ingests inputs into standard AgentMessage objects.
    """

    def __init__(self, name: str = "IngestionAgent"):
        self.name = name

    def run(self, message: AgentMessage) -> AgentMessage:
        """Pipeline stage processor for a single message."""
        if not message.text or not str(message.text).strip():
            message.status = "ERROR"
            message.error = "Input text cannot be empty."
            return message

        message.text = str(message.text).strip()
        message.status = "INGESTED"
        return message

    def ingest(
        self,
        input_data: Union[str, List[str], Path, pd.DataFrame, None] = None
    ) -> List[AgentMessage]:
        """
        Flexible ingestion helper:
        - None: loads processed/test.csv by default
        - str pointing to .csv file: loads that CSV
        - str (text comment): wraps as single AgentMessage
        - list of strings: wraps each as AgentMessage
        - DataFrame: extracts text column and converts to list of AgentMessages
        """
        # Default: processed/test.csv
        if input_data is None:
            return self._ingest_csv(TEST_CSV)

        # Path or CSV string
        if isinstance(input_data, (Path, str)):
            path_candidate = Path(str(input_data))
            if path_candidate.is_file() and path_candidate.suffix.lower() == ".csv":
                return self._ingest_csv(path_candidate)
            # Regular single string comment
            msg = AgentMessage(text=str(input_data).strip())
            return [self.run(msg)]

        # List of items
        if isinstance(input_data, list):
            messages = []
            for item in input_data:
                text = item.text if isinstance(item, AgentMessage) else str(item).strip()
                msg = AgentMessage(text=text)
                messages.append(self.run(msg))
            return messages

        # DataFrame
        if isinstance(input_data, pd.DataFrame):
            return self._ingest_dataframe(input_data)

        raise ValueError(f"Unsupported input type for IngestionAgent: {type(input_data)}")

    def _ingest_csv(self, csv_path: Path, max_rows: int = None) -> List[AgentMessage]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, nrows=max_rows)
        return self._ingest_dataframe(df)

    def _ingest_dataframe(self, df: pd.DataFrame) -> List[AgentMessage]:
        # Determine text column
        if TEXT_COLUMN in df.columns:
            text_col = TEXT_COLUMN
        elif RAW_TEXT_FALLBACK in df.columns:
            text_col = RAW_TEXT_FALLBACK
        elif "text" in df.columns:
            text_col = "text"
        else:
            raise KeyError(f"None of {[TEXT_COLUMN, RAW_TEXT_FALLBACK, 'text']} found in DataFrame columns: {df.columns.tolist()}")

        messages = []
        for idx, row in df.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            msg = AgentMessage(
                id=str(row.get("id", idx)),
                text=raw_text
            )
            messages.append(self.run(msg))
        return messages

