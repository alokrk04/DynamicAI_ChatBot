"""
analytics_store.py
──────────────────
In-memory (session-scoped) analytics log.
Every chatbot turn is recorded; the dashboard reads from this store.
Optionally persists to a local JSON file so data survives page refreshes.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pandas as pd

# ── persistence path ──────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "_analytics_log.json")


class AnalyticsStore:
    """Singleton-like class (one instance shared across the Streamlit session)."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._load()                                    # restore from disk if present

    # ── recording ─────────────────────────────────────────
    def record(
        self,
        user_text:       str,
        bot_text:        str,
        intent:          str,
        intent_conf:     float,
        entities:        dict[str, list[str]],
        sentiment:       dict[str, Any],
        response_time_ms: float,
    ) -> None:
        self.records.append({
            "timestamp":       time.time(),
            "user_text":       user_text,
            "bot_text":        bot_text,
            "intent":          intent,
            "intent_conf":     intent_conf,
            "entities":        entities,
            "sentiment_polarity": sentiment.get("polarity", "neutral"),
            "sentiment_conf":     sentiment.get("polarity_conf", 0.5),
            "emotion":            sentiment.get("emotion", "neutral"),
            "emotion_conf":       sentiment.get("emotion_conf", 0.5),
            "response_time_ms":   round(response_time_ms, 1),
        })
        self._save()

    # ── retrieval ─────────────────────────────────────────
    def as_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=[
                "timestamp", "user_text", "bot_text", "intent", "intent_conf",
                "entities", "sentiment_polarity", "sentiment_conf", "emotion",
                "emotion_conf", "response_time_ms",
            ])
        df = pd.DataFrame(self.records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    # ── aggregate helpers (used by the dashboard) ─────────
    def intent_counts(self) -> pd.Series:
        df = self.as_dataframe()
        return df["intent"].value_counts() if not df.empty else pd.Series(dtype="int64")

    def sentiment_counts(self) -> pd.Series:
        df = self.as_dataframe()
        return df["sentiment_polarity"].value_counts() if not df.empty else pd.Series(dtype="int64")

    def emotion_counts(self) -> pd.Series:
        df = self.as_dataframe()
        return df["emotion"].value_counts() if not df.empty else pd.Series(dtype="int64")

    def avg_response_time(self) -> float:
        df = self.as_dataframe()
        return float(df["response_time_ms"].mean()) if not df.empty else 0.0

    def total_interactions(self) -> int:
        return len(self.records)

    def entity_summary(self) -> dict[str, int]:
        """How many times each entity type was detected."""
        counts: dict[str, int] = {}
        for r in self.records:
            for etype in (r.get("entities") or {}):
                counts[etype] = counts.get(etype, 0) + 1
        return counts

    # ── persistence ───────────────────────────────────────
    def _save(self) -> None:
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(self.records, f)
        except Exception:
            pass          # non-fatal – analytics still live in memory

    def _load(self) -> None:
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE) as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []

    def clear(self) -> None:
        self.records = []
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
