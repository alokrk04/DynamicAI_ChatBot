"""
chatbot_core.py
───────────────
Central orchestrator.  Every user message flows through here:

    User text
        │
        ▼
    ┌─────────────────┐
    │  NLP Engine      │  intent + NER
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Sentiment Engine │  polarity + emotion
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  FAQ Engine      │  instant answer if confident
    └────────┬────────┘
             │  (miss)
             ▼
    ┌─────────────────┐
    │  Gemini Client   │  generative fallback
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Analytics Store  │  log everything
    └─────────────────┘
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

from nlp_engine import IntentRecogniser, NERExtractor, ContextMemory
from sentiment_engine import SentimentAnalyser
from faq_engine import FAQEngine
from analytics_store import AnalyticsStore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# RESPONSE ENVELOPE  (what the UI receives)
# ══════════════════════════════════════════════════════════════
@dataclass
class ChatResponse:
    text: str
    intent: str
    intent_conf: float
    multi_intents: list[tuple[str, float]]
    entities: dict[str, list[str]]
    sentiment: dict[str, Any]
    source: str  # "faq" | "gemini" | "fallback"
    response_time_ms: float
    context_summary: str = ""


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════
class ChatbotCore:
    def __init__(self) -> None:
        self.intent_rec = IntentRecogniser()
        self.ner = NERExtractor()
        self.sentiment = SentimentAnalyser()
        self.faq = FAQEngine()
        self.memory = ContextMemory(window=20)
        self.analytics = AnalyticsStore()
        self._gemini = None  # lazy-init (needs API key)

    # ── lazy Gemini init ──────────────────────────────────
    def _get_gemini(self):
        if self._gemini is None:
            from gemini_client import GeminiClient
            self._gemini = GeminiClient()
        return self._gemini

    # ── main entry point ──────────────────────────────────
    def process_message(self, user_text: str) -> ChatResponse:
        t0 = time.perf_counter()

        # ── 1. NLP ─────────────────────────────────────
        intent, intent_conf = self.intent_rec.predict(user_text)
        multi_intents = self.intent_rec.predict_multi(user_text)
        entities = self.ner.extract(user_text)

        # ── 2. Sentiment ───────────────────────────────
        sentiment = self.sentiment.analyse(user_text)

        # ── 3. Memory ──────────────────────────────────
        self.memory.add_turn("user", user_text, intent=intent, entities=entities)
        ctx_summary = self.memory.get_summary()

        # ── 4. FAQ check DISABLED - Always use Gemini ──
        # (FAQ engine still logs feedback for learning but doesn't return answers)

        # ── 5. Gemini (primary response source) ────────
        try:
            gemini = self._get_gemini()
            bot_text = gemini.respond(
                user_text,
                intent=intent,
                intent_conf=intent_conf,
                entities=entities,
                sentiment=sentiment,
                context_summary=ctx_summary,
            )
            source = "gemini"
        except Exception as exc:
            logger.error("Gemini unreachable: %s", exc)
            bot_text = _emergency_fallback(intent, sentiment)
            source = "fallback"

        # ── 6. Memory (bot turn) ───────────────────────
        self.memory.add_turn("model", bot_text)

        # ── 7. Analytics ───────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.analytics.record(
            user_text=user_text,
            bot_text=bot_text,
            intent=intent,
            intent_conf=intent_conf,
            entities=entities,
            sentiment=sentiment,
            response_time_ms=elapsed_ms,
        )

        return ChatResponse(
            text=bot_text,
            intent=intent,
            intent_conf=intent_conf,
            multi_intents=multi_intents,
            entities=entities,
            sentiment=sentiment,
            source=source,
            response_time_ms=round(elapsed_ms, 1),
            context_summary=ctx_summary,
        )

    # ── feedback loop (self-learning) ─────────────────────
    def handle_feedback(self, user_text: str, positive: bool) -> None:
        self.faq.feedback(user_text, positive)

    # ── reset ─────────────────────────────────────────────
    def clear_conversation(self) -> None:
        self.memory.clear()
        if self._gemini:
            self._gemini.reset_chat()

    def clear_analytics(self) -> None:
        self.analytics.clear()


# ══════════════════════════════════════════════════════════════
# EMERGENCY FALLBACK  (no Gemini, no FAQ)
# ══════════════════════════════════════════════════════════════
def _emergency_fallback(intent: str, sentiment: dict[str, Any]) -> str:
    prefix = ""
    if sentiment.get("polarity") == "negative":
        prefix = "I hear you – sorry you're going through that. "

    mapping = {
        "greeting": "Hey! 👋 I'm DynamiChat. How can I help?",
        "farewell": "Goodbye! 👋 See you next time!",
        "thanks": "You're welcome! 😊",
        "help": "I can answer questions, detect sentiment, extract entities and more. Try asking me something!",
        "joke": "Why did the scarecrow win an award? He was outstanding in his field! 🌾😄",
    }
    return prefix + mapping.get(intent, "I'm here to help! Could you rephrase that for me? 😊")