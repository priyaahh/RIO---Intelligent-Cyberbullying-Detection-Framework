"""
agents/preprocessing.py

PreprocessingAgent:
- Cleans and normalizes text inputs using the project's established preprocessing logic.
- Preserves casing, language, and punctuation as required by transformer tokenizers.
- Sets `cleaned_text` on the AgentMessage.
"""

import sys
from pathlib import Path
from .schemas import AgentMessage

# Reference project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PreprocessingAgent:
    """
    Cleans text matching the preprocessing behavior of the training pipeline.
    """

    def __init__(self, name: str = "PreprocessingAgent"):
        self.name = name

    def clean_text(self, text: str) -> str:
        """
        Applies standard text cleaning matching the training data preparation:
        - Ensures valid string type
        - Strips surrounding whitespace
        - Preserves punctuation, casing, and emojis needed for cyberbullying signals
        """
        if text is None:
            return ""
        cleaned = str(text).strip()
        return cleaned

    def run(self, message: AgentMessage) -> AgentMessage:
        """Executes text cleaning on a single AgentMessage."""
        if message.status == "ERROR":
            return message

        cleaned = self.clean_text(message.text)
        if not cleaned:
            message.status = "ERROR"
            message.error = "Text was empty after preprocessing."
            return message

        message.cleaned_text = cleaned
        message.status = "PREPROCESSED"
        return message

