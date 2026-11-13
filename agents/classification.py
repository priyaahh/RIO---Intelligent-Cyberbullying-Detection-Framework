"""
agents/classification.py

ClassificationAgent:
- Loads and runs the existing BERT model (via model_module/predict.py or saved_model checkpoint).
- Loads and runs the existing DeBERTa-v3 model (via models/deberta_v3 checkpoint).
- Implements an explicit mapping layer converting DeBERTa's 6 multi-label sigmoid outputs
  into the 5 shared canonical categories.
- Combines model outputs into an ensemble decision (average probability across shared categories).
- Gracefully handles missing model weights by flagging them in the message and providing
  a mock fallback mode so the entire pipeline, API, and tests run cleanly.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError) as exc:
    torch = None
    TORCH_AVAILABLE = False
    logging.getLogger("ClassificationAgent").warning(f"PyTorch could not be loaded ({exc}). Using mock mode.")

import numpy as np

from .config import (
    BERT_MODEL_PATH,
    DEBERTA_MODEL_PATH,
    ROBERTA_MODEL_PATH,
    ROBERTA_MODEL_NAME,
    SHARED_LABELS,
    DEBERTA_RAW_LABELS,
    MAX_LENGTH,
    CYBERBULLYING_THRESHOLD,
)
from .schemas import AgentMessage

logger = logging.getLogger("ClassificationAgent")


class ClassificationAgent:
    """
    Coordinates multi-model inference across BERT, DeBERTa-v3, and their ensemble.
    """

    def __init__(
        self,
        name: str = "ClassificationAgent",
        mock_mode_if_missing: bool = True,
        device: Optional[str] = None,
    ):
        self.name = name
        self.mock_mode = mock_mode_if_missing
        if device:
            self.device = device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Model containers
        self.bert_available = False
        self.deberta_available = False
        self.roberta_available = False
        self.bert_tok = None
        self.bert_model = None
        self.deberta_tok = None
        self.deberta_model = None
        self.roberta_tok = None
        self.roberta_model = None

        self._init_models()

    def _init_models(self):
        """Attempts to load BERT, DeBERTa, and RoBERTa from their respective project paths."""
        # ── 1. Initialize BERT ────────────────────────────────────────────────
        if BERT_MODEL_PATH.is_dir():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                logger.info(f"Loading BERT model from {BERT_MODEL_PATH}...")
                self.bert_tok = AutoTokenizer.from_pretrained(str(BERT_MODEL_PATH))
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    str(BERT_MODEL_PATH)
                ).to(self.device).eval()
                self.bert_available = True
                logger.info("BERT model loaded successfully.")
            except Exception as exc:
                logger.warning(f"Failed to load BERT model from {BERT_MODEL_PATH}: {exc}")
                self.bert_available = False
        else:
            logger.warning(f"BERT weights not found at {BERT_MODEL_PATH}. (Mock mode: {self.mock_mode})")

        # ── 2. Initialize DeBERTa-v3 ──────────────────────────────────────────
        if DEBERTA_MODEL_PATH.is_dir():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                logger.info(f"Loading DeBERTa model from {DEBERTA_MODEL_PATH}...")
                self.deberta_tok = AutoTokenizer.from_pretrained(str(DEBERTA_MODEL_PATH))
                self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
                    str(DEBERTA_MODEL_PATH)
                ).to(self.device).eval()
                self.deberta_available = True
                logger.info("DeBERTa-v3 model loaded successfully.")
            except Exception as exc:
                logger.warning(f"Failed to load DeBERTa model from {DEBERTA_MODEL_PATH}: {exc}")
                self.deberta_available = False
        else:
            logger.warning(f"DeBERTa weights not found at {DEBERTA_MODEL_PATH}. (Mock mode: {self.mock_mode})")

        # ── 3. Initialize RoBERTa (Your Model) ────────────────────────────────
        if ROBERTA_MODEL_PATH.is_dir():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                logger.info(f"Loading RoBERTa model from {ROBERTA_MODEL_PATH}...")
                self.roberta_tok = AutoTokenizer.from_pretrained(str(ROBERTA_MODEL_PATH))
                self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
                    str(ROBERTA_MODEL_PATH)
                ).to(self.device).eval()
                self.roberta_available = True
                logger.info("RoBERTa model loaded successfully.")
            except Exception as exc:
                logger.warning(f"Failed to load RoBERTa model from {ROBERTA_MODEL_PATH}: {exc}")
                self.roberta_available = False
        else:
            logger.warning(f"RoBERTa weights not found at {ROBERTA_MODEL_PATH}. (Mock mode: {self.mock_mode})")

    # ── Inference Methods ─────────────────────────────────────────────────────

    def predict_bert(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """
        Runs BERT inference. Returns calibrated probabilities across SHARED_LABELS.
        """
        flags = []
        if self.bert_available and self.bert_tok and self.bert_model:
            inputs = self.bert_tok(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.bert_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

            scores = {label: round(float(probs[i]), 4) for i, label in enumerate(SHARED_LABELS)}
            return scores, flags

        # Fallback / Mock mode when weights are not local
        flags.append("bert_weights_missing")
        scores = self._mock_predict(text, bias="bert")
        return scores, flags

    def predict_roberta(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """
        Runs RoBERTa inference (your model). Returns calibrated probabilities across SHARED_LABELS.
        """
        flags = []
        if self.roberta_available and self.roberta_tok and self.roberta_model:
            inputs = self.roberta_tok(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.roberta_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

            scores = {label: round(float(probs[i]), 4) for i, label in enumerate(SHARED_LABELS)}
            return scores, flags

        # Fallback / Mock mode when weights are not local
        flags.append("roberta_weights_missing")
        scores = self._mock_predict(text, bias="roberta")
        return scores, flags

    def predict_deberta(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """
        Runs DeBERTa-v3 inference, extracts 6-class multi-label sigmoid probabilities,
        and applies the explicit mapping layer to SHARED_LABELS.
        """
        flags = []
        if self.deberta_available and self.deberta_tok and self.deberta_model:
            inputs = self.deberta_tok(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.deberta_model(**inputs).logits
                raw_sigmoids = torch.sigmoid(logits)[0].cpu().tolist()

            raw_scores = {label: float(raw_sigmoids[i]) for i, label in enumerate(DEBERTA_RAW_LABELS)}
            shared_scores = self.map_deberta_to_shared(raw_scores)
            return shared_scores, flags

        # Fallback / Mock mode when weights are not local
        flags.append("deberta_weights_missing")
        raw_mock = self._mock_raw_deberta(text)
        shared_scores = self.map_deberta_to_shared(raw_mock)
        return shared_scores, flags

    @staticmethod
    def map_deberta_to_shared(deberta_raw: Dict[str, float]) -> Dict[str, float]:
        """
        Explicit mapping layer: converts DeBERTa's 6 multi-label sigmoid scores
        into normalized probabilities over the 5 canonical SHARED_LABELS:
          - Threat       <- threat
          - Insult       <- insult
          - Hate Speech  <- identity_hate
          - Harassment   <- max(toxic, severe_toxic, obscene)
          - Not Cyberbullying <- max(0.0, 1.0 - max(all toxic categories))
        """
        threat_score = deberta_raw.get("threat", 0.0)
        insult_score = deberta_raw.get("insult", 0.0)
        hate_score = deberta_raw.get("identity_hate", 0.0)
        harassment_score = max(
            deberta_raw.get("toxic", 0.0),
            deberta_raw.get("severe_toxic", 0.0),
            deberta_raw.get("obscene", 0.0)
        )

        max_toxic = max(threat_score, insult_score, hate_score, harassment_score)
        not_cb_score = max(0.05, 1.0 - max_toxic)

        raw_mapped = {
            "Not Cyberbullying": not_cb_score,
            "Insult": insult_score,
            "Threat": threat_score,
            "Hate Speech": hate_score,
            "Harassment": harassment_score,
        }

        # Normalize across the 5 classes so probabilities sum to 1.0
        total = sum(raw_mapped.values())
        if total <= 0:
            total = 1.0
        normalized = {k: round(v / total, 4) for k, v in raw_mapped.items()}
        return normalized

    def _mock_predict(self, text: str, bias: str = "bert") -> Dict[str, float]:
        """
        Deterministic mock predictor for pipeline testing when model weights are
        stored externally (Google Drive). Vocabulary derived from real dataset patterns
        in processed/train.csv across all 5 label classes.
        """
        t = text.lower()

        # ── Threat keywords (derived from train.csv Threat class) ─────────────
        threat_kw = [
            "kill", "murder", "shoot", "hang", "stab", "bomb", "attack",
            "die", "death", "dead", "gonna get you", "hunt you down",
            "hunt you", "destroy you", "make you pay", "you will pay",
            "watch your back", "i will find you", "cut off", "slice you",
            "blow up", "strangle", "poison", "slit", "corpse", "i will kill",
            "going to kill", "gonna kill", "i hope you die", "hope you die",
            "burn in hell", "burn alive", "get you", "hurt you", "i will hurt",
            "you must die", "you should die", "i shall", "i will make",
            "regret this", "you will regret", "kick your ass", "beat you",
            "bash you", "smash you", "slap you",
        ]

        # ── Hate Speech keywords (derived from Hate Speech class) ──────────────
        hate_kw = [
            "nigger", "niggers", "faggot", "faggots", "cunt",
            "slut", "whore", "terrorist", "beaner", "beaners",
            "feminazi", "subhuman", "inferior race", "white power",
            "go back to your country", "deport", "barbaric",
            "rape", "rapey", "statutory rape", "sexist", "sexism",
            "homophobe", "homophobic", "racist", "racism", "bigot",
            "nazi", "heil", "jihad", "islamophobe",
            "black people are", "white people are",
            "identity hate", "hate crime",
        ]

        # ── Insult keywords (derived from Insult class) ───────────────────────
        insult_kw = [
            "stupid", "idiot", "loser", "dumb", "dumbass", "ugly", "moron",
            "bitch", "bitches", "hoe", "hoes", "pussy", "dickhead", "asshole",
            "ass", "jerk", "clown", "freak", "weirdo", "creep", "retard",
            "retarded", "trash", "garbage", "pathetic", "worthless", "useless",
            "disgrace", "fool", "failure", "suck", "you suck", "piece of shit",
            "shut up", "shut your mouth", "you're so dumb", "you are dumb",
            "so stupid", "brain dead", "brainless", "dim wit", "dimwit",
            "inbred", "degenerate", "scum", "lowlife", "low life",
            "dirtbag", "dirt bag", "rat", "pig", "swine",
            "fat", "fat ass", "lard", "obese", "ugly face",
            "you're ugly", "you are ugly", "fugly",
            "fucking idiot", "fucking stupid", "fucking moron",
            "fuck you", "fuck off", "go fuck", "get lost",
        ]

        # ── Harassment keywords (derived from Harassment class) ───────────────
        harass_kw = [
            "bully", "bullied", "bullying", "bullies",
            "harass", "harassment", "stalking", "stalk",
            "nobody likes you", "no one likes you", "everyone hates you",
            "you have no friends", "go away", "get out", "leave me alone",
            "stop following", "won't leave you alone", "won't stop",
            "spreading rumors", "spread rumors", "rumors about you",
            "school bully", "high school", "middle school",
            "vandalize", "vandalism", "vandal",
            "reported you", "block you", "banned",
            "i hate you", "hate everything about you",
            "annoying", "pest", "pest you",
        ]

        # ── Scoring: check each category ─────────────────────────────────────
        scores = {label: 0.05 for label in SHARED_LABELS}

        # Multi-word phrase check (highest priority: check phrases first)
        threat_phrases = [p for p in threat_kw if " " in p]
        insult_phrases = [p for p in insult_kw if " " in p]
        harass_phrases = [p for p in harass_kw if " " in p]
        hate_phrases   = [p for p in hate_kw if " " in p]

        # Single-word signals (token boundary check to avoid partial matches)
        import re as _re
        def _has_word(word_list, text_lower):
            for item in word_list:
                if " " in item:
                    if item in text_lower:
                        return True
                else:
                    if _re.search(r'\b' + _re.escape(item) + r'\b', text_lower):
                        return True
            return False

        threat_hit   = _has_word(threat_kw, t)
        hate_hit     = _has_word(hate_kw, t)
        insult_hit   = _has_word(insult_kw, t)
        harass_hit   = _has_word(harass_kw, t)

        # Assign scores (allow multiple signals — pick the highest boost)
        if threat_hit:
            scores["Threat"] += 0.75
        if hate_hit:
            scores["Hate Speech"] += 0.70
        if insult_hit:
            scores["Insult"] += 0.60
        if harass_hit:
            scores["Harassment"] += 0.55

        # If nothing matched → lean toward Not Cyberbullying
        if not any([threat_hit, hate_hit, insult_hit, harass_hit]):
            scores["Not Cyberbullying"] += 0.80

        # Slight model-specific bias for diversity
        if bias == "bert":
            scores["Insult"] += 0.02
        elif bias == "roberta":
            scores["Threat"] += 0.01

        total = sum(scores.values())
        return {k: round(v / total, 4) for k, v in scores.items()}

    def _mock_raw_deberta(self, text: str) -> Dict[str, float]:
        """
        Generates realistic 6-class raw sigmoids for mock DeBERTa fallback.
        Vocabulary mirrors the same dataset-derived keywords as _mock_predict.
        """
        import re as _re
        t = text.lower()

        def _has_word(word_list, text_lower):
            for item in word_list:
                if " " in item:
                    if item in text_lower:
                        return True
                else:
                    if _re.search(r'\b' + _re.escape(item) + r'\b', text_lower):
                        return True
            return False

        raw = {label: 0.05 for label in DEBERTA_RAW_LABELS}

        # Threat signals
        threat_kw = [
            "kill", "murder", "shoot", "hang", "stab", "attack", "die", "death",
            "dead", "hunt you", "destroy you", "cut off", "strangle", "corpse",
            "i will kill", "going to kill", "gonna kill", "hope you die",
            "burn in hell", "hurt you", "i will hurt", "you must die",
            "you should die", "i shall", "regret this", "kick your ass",
            "beat you", "bash you", "smash you",
        ]
        # Hate speech signals
        hate_kw = [
            "nigger", "niggers", "faggot", "faggots", "slut", "whore",
            "terrorist", "beaner", "beaners", "feminazi", "subhuman",
            "white power", "deport", "barbaric", "rape", "rapey", "sexist",
            "homophobe", "homophobic", "racist", "racism", "bigot",
            "nazi", "heil", "jihad", "identity hate", "hate crime",
        ]
        # Insult signals
        insult_kw = [
            "stupid", "idiot", "loser", "dumb", "dumbass", "ugly", "moron",
            "bitch", "bitches", "hoe", "hoes", "pussy", "dickhead", "asshole",
            "ass", "jerk", "clown", "freak", "retard", "retarded", "trash",
            "pathetic", "worthless", "useless", "disgrace", "fool", "failure",
            "suck", "you suck", "piece of shit", "brain dead", "scum", "lowlife",
            "dirtbag", "fat ass", "fugly", "fucking idiot", "fuck you", "fuck off",
        ]
        # Harassment signals
        harass_kw = [
            "bully", "bullied", "bullying", "harass", "harassment", "stalking",
            "nobody likes you", "no one likes you", "everyone hates you",
            "go away", "get out", "leave me alone", "stop following",
            "spreading rumors", "spread rumors", "school bully",
            "i hate you", "hate everything about you", "annoying",
        ]

        threat_hit  = _has_word(threat_kw, t)
        hate_hit    = _has_word(hate_kw, t)
        insult_hit  = _has_word(insult_kw, t)
        harass_hit  = _has_word(harass_kw, t)

        if threat_hit:
            raw["threat"] = 0.88
            raw["toxic"] = 0.90
            raw["severe_toxic"] = 0.60
        if hate_hit:
            raw["identity_hate"] = 0.85
            raw["toxic"] = max(raw["toxic"], 0.88)
        if insult_hit:
            raw["insult"] = 0.82
            raw["obscene"] = 0.70
            raw["toxic"] = max(raw["toxic"], 0.75)
        if harass_hit:
            raw["toxic"] = max(raw["toxic"], 0.65)
            raw["obscene"] = max(raw["obscene"], 0.45)

        return raw

    # ── Pipeline Runner ───────────────────────────────────────────────────────

    def run(self, message: AgentMessage) -> AgentMessage:
        """
        Executes classification:
        1. Runs BERT (if available or mock fallback).
        2. Runs DeBERTa (if available or mock fallback).
        3. Computes ensemble average across shared categories.
        4. Selects primary label, confidence, and is_cyberbullying flag.
        """
        if message.status == "ERROR":
            return message

        text = message.cleaned_text or message.text
        if not text:
            message.status = "ERROR"
            message.error = "No text available for classification."
            return message

        bert_scores, bert_flags = self.predict_bert(text)
        deberta_scores, deberta_flags = self.predict_deberta(text)
        roberta_scores, roberta_flags = self.predict_roberta(text)

        message.model_flags.extend(bert_flags)
        message.model_flags.extend(deberta_flags)
        message.model_flags.extend(roberta_flags)

        # ── Tri-Model Ensemble Combination (BERT + DeBERTa + RoBERTa) ─────────
        ensemble_scores = {
            label: round((bert_scores[label] + deberta_scores[label] + roberta_scores[label]) / 3.0, 4)
            for label in SHARED_LABELS
        }

        message.per_model_scores = {
            "bert": bert_scores,
            "deberta": deberta_scores,
            "roberta": roberta_scores,
            "ensemble": ensemble_scores,
        }

        # Best prediction
        best_label = max(ensemble_scores, key=ensemble_scores.get)
        confidence = ensemble_scores[best_label]
        is_cb = (best_label != "Not Cyberbullying")

        message.primary_label = best_label
        message.confidence = confidence
        message.is_cyberbullying = is_cb
        message.final_labels = [best_label] if is_cb else ["Not Cyberbullying"]
        message.status = "CLASSIFIED"

        return message
