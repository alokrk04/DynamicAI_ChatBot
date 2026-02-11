"""
nlp_engine.py
─────────────
Offline NLP layer — Intent Recognition · Named Entity Recognition · Contextual Memory
Uses keyword-pattern matching + lightweight TF-IDF vectorisation so the chatbot
works instantly without downloading large model weights.
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════
# 1.  INTENT CATALOGUE  (patterns → intent label)
# ══════════════════════════════════════════════════════════════
INTENT_PATTERNS: dict[str, list[str]] = {
    "greeting": [
        r"\b(hi|hello|hey|howdy|good\s*(morning|afternoon|evening)|what'?s\s*up|sup|yo)\b"
    ],
    "farewell": [
        r"\b(bye|goodbye|see\s*you|later|take\s*care|have\s*a\s*(good|nice)\s*(day|night)|quit|exit)\b"
    ],
    "thanks": [
        r"\b(thank(s|you|s?\s*a\s*lot|s?\s*so\s*much)|appreciate|grateful|cheers)\b"
    ],
    "help": [
        r"\b(help|assist|support|what\s*can\s*you\s*do|how\s*do\s*i|usage|guide)\b"
    ],
    "weather": [
        r"\b(weather|temperature|forecast|rain|sunny|cloudy|wind)\b"
    ],
    "time_date": [
        r"\b(time|date|day|today|tomorrow|yesterday|clock|what\s*time)\b"
    ],
    "joke": [
        r"\b(joke|funny|laugh|humor|amuse|entertain)\b"
    ],
    "sentiment_query": [
        r"\b(how\s*(am\s*i|are\s*you|do\s*i\s*feel)|feel(ing)?|mood|emotion|sentiment)\b"
    ],
    "name_identity": [
        r"\b(your\s*name|who\s*are\s*you|what\s*are\s*you|introduce\s*yourself|are\s*you\s*(ai|bot|robot))\b"
    ],
    "capability": [
        r"\b(what\s*can|features|abilities|skills|trained|capable|do\s*you\s*support)\b"
    ],
    "general": []  # fallback
}


# ══════════════════════════════════════════════════════════════
# 2.  NAMED ENTITY PATTERNS
# ══════════════════════════════════════════════════════════════
NER_PATTERNS: dict[str, str] = {
    "EMAIL":    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "PHONE":    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,6}",
    "URL":      r"https?://[^\s<>\"']+",
    "DATE":     r"\b(?:(?:0?[1-9]|[12]\d|3[01])[-/](?:0?[1-9]|1[0-2])[-/]\d{2,4}|\d{4}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01]))\b",
    "TIME":     r"\b\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\b",
    "CURRENCY": r"(?:[\$€£₹]\s*\d+(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|INR|dollars?|euros?|pounds?|rupees?))",
    "PERSON":   r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
    "CITY":     r"\b(?:New\s*York|Los\s*Angeles|Chicago|Houston|Phoenix|San\s*Francisco|London|Paris|Tokyo|Mumbai|Delhi|Bangalore|Bengaluru|Chennai|Kolkata|Sydney|Berlin|Dubai|Singapore)\b",
}


# ══════════════════════════════════════════════════════════════
# 3.  INTENT RECOGNISER  (regex + TF-IDF fallback)
# ══════════════════════════════════════════════════════════════
class IntentRecogniser:
    """Two-stage intent classifier.

    Stage 1 – fast regex scan over INTENT_PATTERNS.
    Stage 2 – TF-IDF cosine-similarity fallback for queries that
              miss every regex (catches paraphrases).
    """

    # Representative sentence per intent – used for TF-IDF similarity
    _TFIDF_CORPUS: dict[str, str] = {
        "greeting":       "hello hi hey good morning",
        "farewell":       "bye goodbye see you later take care",
        "thanks":         "thank you thanks a lot appreciate",
        "help":           "help me how do I what can you do guide",
        "weather":        "what is the weather forecast temperature",
        "time_date":      "what time is it what is today's date",
        "joke":           "tell me a joke make me laugh funny",
        "sentiment_query":"how are you feeling mood emotion",
        "name_identity":  "what is your name who are you introduce yourself",
        "capability":     "what features do you have what can you do abilities",
    }

    def __init__(self) -> None:
        self._corpus_labels = list(self._TFIDF_CORPUS.keys())
        self._corpus_texts  = list(self._TFIDF_CORPUS.values())
        self._vectoriser    = TfidfVectorizer(stop_words="english")
        self._corpus_vecs   = self._vectoriser.fit_transform(self._corpus_texts)

    # ── public ────────────────────────────────────────────
    def predict(self, text: str) -> tuple[str, float]:
        """Return (intent_label, confidence 0-1)."""
        # Stage 1 – regex
        for label, pats in INTENT_PATTERNS.items():
            if label == "general":
                continue
            for pat in pats:
                if re.search(pat, text, re.IGNORECASE):
                    return label, 0.95

        # Stage 2 – TF-IDF fallback
        try:
            user_vec  = self._vectoriser.transform([text])
            sims      = cosine_similarity(user_vec, self._corpus_vecs)[0]
            best_idx  = int(np.argmax(sims))
            best_sim  = float(sims[best_idx])
            if best_sim > 0.12:                         # low bar – catch loose matches
                return self._corpus_labels[best_idx], round(best_sim, 3)
        except Exception:                               # pragma: no cover
            pass

        return "general", 0.50

    # ── multi-intent helper ───────────────────────────────
    def predict_multi(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top-k intents sorted by confidence (regex hits first)."""
        hits: list[tuple[str, float]] = []
        for label, pats in INTENT_PATTERNS.items():
            if label == "general":
                continue
            for pat in pats:
                if re.search(pat, text, re.IGNORECASE):
                    hits.append((label, 0.95))
                    break

        if not hits:
            try:
                user_vec = self._vectoriser.transform([text])
                sims     = cosine_similarity(user_vec, self._corpus_vecs)[0]
                ranked   = np.argsort(sims)[::-1][:top_k]
                for idx in ranked:
                    if sims[idx] > 0.05:
                        hits.append((self._corpus_labels[idx], round(float(sims[idx]), 3)))
            except Exception:
                pass

        if not hits:
            hits.append(("general", 0.50))

        # deduplicate, keep highest confidence per label
        seen: dict[str, float] = {}
        for label, conf in hits:
            if label not in seen or conf > seen[label]:
                seen[label] = conf
        return sorted(seen.items(), key=lambda x: -x[1])[:top_k]


# ══════════════════════════════════════════════════════════════
# 4.  NAMED ENTITY RECOGNISER
# ══════════════════════════════════════════════════════════════
class NERExtractor:
    """Regex-based NER – no model download required."""

    @staticmethod
    def extract(text: str) -> dict[str, list[str]]:
        entities: dict[str, list[str]] = {}
        for label, pattern in NER_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[label] = list(dict.fromkeys(matches))  # dedupe, keep order
        return entities


# ══════════════════════════════════════════════════════════════
# 5.  CONTEXTUAL MEMORY (sliding-window conversation store)
# ══════════════════════════════════════════════════════════════
class ContextMemory:
    """Keeps the last *window* turns so Gemini can stay on-topic."""

    def __init__(self, window: int = 20) -> None:
        self.window   = window
        self.turns: deque[dict[str, Any]] = deque(maxlen=window)
        self.entities_seen: dict[str, list[str]] = {}   # accumulated NER
        self.topic_history: list[str] = []               # intent trail

    # ── add / retrieve ────────────────────────────────────
    def add_turn(self, role: str, text: str, intent: str | None = None,
                 entities: dict[str, list[str]] | None = None) -> None:
        self.turns.append({
            "role":      role,
            "text":      text,
            "intent":    intent,
            "entities":  entities or {},
            "timestamp": time.time(),
        })
        if intent and role == "user":
            self.topic_history.append(intent)
        if entities:
            for k, v in entities.items():
                self.entities_seen.setdefault(k, []).extend(v)

    def get_history_for_gemini(self) -> list[dict[str, str]]:
        """Return turns as [{"role": ..., "parts": [{"text": ...}]}]."""
        return [
            {"role": t["role"], "parts": [{"text": t["text"]}]}
            for t in self.turns
        ]

    def get_summary(self) -> str:
        """One-line context summary for analytics."""
        n = len(self.turns)
        top_intent = max(set(self.topic_history), key=self.topic_history.count) if self.topic_history else "N/A"
        return f"{n} turns · dominant intent: {top_intent} · entities seen: {list(self.entities_seen.keys())}"

    def clear(self) -> None:
        self.turns.clear()
        self.entities_seen.clear()
        self.topic_history.clear()
