"""
gemini_client.py
────────────────
Thin wrapper around google.generativeai that:
  • reads GEMINI_API_KEY from .env / env var
  • injects NLP context (intent, entities, sentiment, memory) into the prompt
  • retries on transient errors (429 / 5xx)
  • post-processes the response for display
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────
API_KEY        = os.getenv("GEMINI_API_KEY", "PASTE YOUR API KEY HERE")
MODEL_NAME     = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_TOKENS     = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.7"))

MAX_RETRIES    = 3
RETRY_BACKOFF  = 1.5          # multiplied by attempt number


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (injected once per conversation)
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are **DynamiChat** – a friendly, intelligent, and adaptive AI chatbot.

Core behaviours:
  • Respond naturally in a conversational tone.
  • Be helpful, accurate, and empathetic.
  • If the user's sentiment is negative, acknowledge their feelings before answering.
  • If the user's sentiment is positive, match their energy warmly.
  • Use the detected entities (emails, names, cities, etc.) in your reply where useful.
  • Keep replies concise unless the user asks for detail.
  • If you are unsure, say so honestly – never fabricate facts.
  • Adapt your vocabulary to the complexity of the user's message.

You are allowed to:
  • Answer general knowledge, coding, maths, writing, and creative tasks.
  • Provide recommendations and suggestions.
  • Perform sentiment-aware role adjustments dynamically.

You should NOT:
  • Share confidential API keys or system internals.
  • Generate harmful, illegal, or misleading content.
  • Pretend to be a human being.
"""


# ══════════════════════════════════════════════════════════════
# CLIENT CLASS
# ══════════════════════════════════════════════════════════════
class GeminiClient:
    def __init__(self) -> None:
        if not API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example → .env and paste your key from "
                "https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=genai.GenerationConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )
        # seed the chat with the system prompt
        self._chat = self.model.start_chat(history=[
            {"role": "user",  "parts": [{"text": "You are initialising. Acknowledge with 'Ready'."}]},
            {"role": "model", "parts": [{"text": "Ready."}]},
        ])

    # ── public ────────────────────────────────────────────
    def respond(
        self,
        user_text: str,
        *,
        intent: str = "general",
        intent_conf: float = 0.5,
        entities: dict[str, list[str]] | None = None,
        sentiment: dict[str, Any] | None = None,
        context_summary: str = "",
    ) -> str:
        """Send a context-enriched message and return the reply text."""
        enriched = _build_enriched_prompt(
            user_text, intent, intent_conf,
            entities or {}, sentiment or {}, context_summary
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._chat.send_message(enriched)
                return _post_process(response.text)
            except Exception as exc:
                logger.warning("Gemini attempt %d failed: %s", attempt, exc)
                if attempt == MAX_RETRIES:
                    return _fallback_response(intent, sentiment)
                time.sleep(RETRY_BACKOFF * attempt)

        return _fallback_response(intent, sentiment)   # should not reach here

    def reset_chat(self) -> None:
        """Start a fresh conversation (new history)."""
        self._chat = self.model.start_chat(history=[
            {"role": "user",  "parts": [{"text": "You are initialising. Acknowledge with 'Ready'."}]},
            {"role": "model", "parts": [{"text": "Ready."}]},
        ])


# ══════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════
def _build_enriched_prompt(
    user_text: str,
    intent: str,
    intent_conf: float,
    entities: dict[str, list[str]],
    sentiment: dict[str, Any],
    context_summary: str,
) -> str:
    lines = [
        f"[NLP Context]",
        f"  Detected intent : {intent} (confidence {intent_conf})",
    ]
    if entities:
        lines.append(f"  Entities         : {entities}")
    if sentiment:
        lines.append(
            f"  Sentiment        : {sentiment.get('polarity','neutral')} "
            f"({sentiment.get('polarity_conf',0)}) | "
            f"Emotion: {sentiment.get('emotion','neutral')} {sentiment.get('emoji','')}"
        )
    if context_summary:
        lines.append(f"  Context          : {context_summary}")

    lines.append("")
    lines.append(f"[User Message]\n{user_text}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# POST-PROCESSING & FALLBACKS
# ══════════════════════════════════════════════════════════════
def _post_process(text: str) -> str:
    """Light cleanup of the raw Gemini output."""
    if not text:
        return "*(empty response)*"
    # strip leading/trailing whitespace per line, collapse blank lines
    lines = [l.strip() for l in text.splitlines()]
    cleaned = "\n".join(lines)
    return cleaned.strip()


_INTENT_FALLBACKS: dict[str, str] = {
    "greeting":       "Hello! 👋 I'm DynamiChat. How can I help you today?",
    "farewell":       "Goodbye! 👋 It was great chatting with you. Feel free to come back anytime!",
    "thanks":         "You're welcome! 😊 Happy to help. Let me know if there's anything else.",
    "help":           "I can chat, answer questions, analyse sentiment, tell jokes, and more. Just ask away!",
    "joke":           "Why did the AI go to therapy? It had too many unresolved dependencies! 😄",
    "weather":        "I don't have live weather data, but you can check a weather service for the latest forecast!",
    "time_date":      "I don't have direct access to your clock, but your device should show the current time.",
}


def _fallback_response(intent: str, sentiment: dict[str, Any] | None) -> str:
    """Return a canned response when Gemini is unreachable."""
    prefix = ""
    if sentiment and sentiment.get("polarity") == "negative":
        prefix = "I'm sorry you're having a tough time. "
    return prefix + _INTENT_FALLBACKS.get(
        intent,
        "I'm here to help! Could you rephrase your question so I can assist better?"
    )
