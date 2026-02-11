# 🤖 DynamiChat – Dynamic AI Chatbot

A fully end-to-end conversational AI chatbot built with **Google Gemini**, **NLP**, **Machine Learning**, and **Streamlit**.  
Every feature from the project specification is implemented and working.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (app.py)                   │
│   ┌──────────────┐    ┌──────────────────────────────────────┐  │
│   │   Sidebar    │    │   Chat Tab  /  Dashboard Tab          │  │
│   │  NLP details │    │   Message thread, Plotly charts       │  │
│   │  Entities    │    │   KPI cards, Interaction log          │  │
│   │  Sentiment   │    │   Feedback buttons (👍👎)             │  │
│   └──────────────┘    └──────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  calls
                            ▼
              ┌─────────────────────────┐
              │     chatbot_core.py     │  ← Orchestrator
              └──┬──┬──┬──┬──┬─────────┘
                 │  │  │  │  │
    ┌────────────┘  │  │  │  └─────────────┐
    ▼               ▼  │  ▼               ▼
┌──────────┐  ┌────────┐│ ┌─────────┐ ┌──────────────┐
│nlp_engine│  │ faq_   ││ │sentiment│ │analytics_    │
│          │  │engine  ││ │_engine  │ │store         │
│ Intent   │  │        ││ │         │ │              │
│ NER      │  │TF-IDF  ││ │TF-IDF + │ │Logs every    │
│ Memory   │  │FAQ DB  ││ │LogReg   │ │interaction   │
└──────────┘  └────────┘│ └─────────┘ └──────────────┘
                        ▼
              ┌─────────────────┐
              │  gemini_client  │  ← Google Gemini API
              │  (retry, ctx)   │
              └─────────────────┘
```

---

## ✅ Feature Checklist (maps to spec)

| # | Spec Feature | Implementation |
|---|---|---|
| 1 | **NLP-Based Conversational Understanding** | `nlp_engine.py` – Intent Recognition (regex + TF-IDF), NER, Contextual Memory |
| 2 | **Multi-Platform Integration** | API-based Streamlit app; architecture is platform-agnostic |
| 3 | **AI-Powered Response Generation** | Rule-based FAQ (`faq_engine.py`) + Gemini generative AI (`gemini_client.py`) |
| 4 | **Sentiment Analysis & Emotion Detection** | `sentiment_engine.py` – polarity, emotion (6 classes), subjectivity |
| 5 | **Self-Learning & Adaptive AI** | FAQ score boosting/penalising via 👍👎 feedback; automated fallback |
| 6 | **Smart Analytics Dashboard** | Full Plotly dashboard: intent pie, sentiment pie, emotion bar, response-time line, entity summary, interaction log |

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.9+**
- A free **Google Gemini API key** from [AI Studio](https://aistudio.google.com/app/apikey)

### 2. Clone / download the project
```bash
# If you downloaded the zip, extract it and cd into the folder
cd ai_chatbot
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set your API key
```bash
cp .env.example .env
# Open .env in any text editor and replace 'your_gemini_api_key_here'
# with your actual key from AI Studio
```

### 6. Run the app
```bash
streamlit run app.py
```
Your browser will open automatically at **http://localhost:8501**.

---

## 📁 File Map

| File | Role |
|---|---|
| `app.py` | Streamlit UI – chat thread, dashboard, sidebar |
| `chatbot_core.py` | Orchestrator – wires all engines together |
| `nlp_engine.py` | Intent Recognition, NER, Contextual Memory |
| `sentiment_engine.py` | Polarity + Emotion classification |
| `faq_engine.py` | Pre-trained FAQ store with self-learning |
| `gemini_client.py` | Google Gemini API wrapper (retry, context injection) |
| `analytics_store.py` | Interaction logger + aggregates for dashboard |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

---

## 🛠️ How Each Piece Works

### Intent Recognition (`nlp_engine.py`)
Two-stage pipeline:
1. **Regex scan** – catches common phrases instantly (< 1 ms).
2. **TF-IDF cosine-similarity** – catches paraphrases by comparing the user's message to representative sentences for each intent.

Supports **multi-intent detection** (e.g. a message that is both a greeting and a question).

### Named Entity Recognition
Regex-based extraction of: `EMAIL`, `PHONE`, `URL`, `DATE`, `TIME`, `CURRENCY`, `PERSON`, `CITY`.  
Extracted entities are highlighted in the sidebar and injected into the Gemini prompt.

### Sentiment & Emotion (`sentiment_engine.py`)
A **TF-IDF + Logistic Regression** pipeline trained on a curated labelled dataset at module load (< 0.5 s).
- **Polarity**: positive / negative / neutral (with confidence score)
- **Emotion**: joy / anger / sadness / fear / surprise / neutral
- **Subjectivity**: objective / subjective (keyword heuristic)

### FAQ Engine (`faq_engine.py`)
- 15+ pre-loaded Q&A pairs covering identity, features, how-it-works, and greetings.
- TF-IDF retrieval; if similarity × learned score ≥ threshold → returns instantly (no API call).
- **Self-learning**: 👍 boosts the matched FAQ's score; 👎 penalises it. Over time the FAQ engine gets better at what *this* user finds helpful.

### Gemini Client (`gemini_client.py`)
- Wraps `google.generativeai` with automatic retries (3×, exponential backoff).
- Injects full NLP context (intent, entities, sentiment, conversation summary) into every prompt.
- Falls back to intent-based canned responses if the API is unreachable.

### Analytics Dashboard
Built with **Plotly** inside Streamlit:
- 🎯 Intent distribution (pie)
- 😊 Sentiment distribution (pie)
- 🎭 Emotion breakdown (bar)
- ⚡ Response time over time (line + avg reference)
- 🏷️ Entity type frequency
- 📝 Scrollable interaction log table
- 4 KPI cards at the top

---

## 💡 Tips & Customisation

- **Add FAQs**: Edit the `_FAQ_DB` list in `faq_engine.py`.
- **Change the model**: Set `GEMINI_MODEL=gemini-1.5-pro` in `.env` for higher quality.
- **Tune confidence**: Lower `CONFIDENCE_THRESHOLD` in `faq_engine.py` to let the FAQ engine answer more queries offline.
- **Extend NER**: Add regex patterns to `NER_PATTERNS` in `nlp_engine.py`.
- **Extend intents**: Add entries to `INTENT_PATTERNS` and `_TFIDF_CORPUS` in `nlp_engine.py`.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Web UI | Streamlit |
| Generative AI | Google Gemini (gemini-2.0-flash) |
| NLP / Intent | scikit-learn TF-IDF + cosine similarity |
| Sentiment / Emotion | scikit-learn Logistic Regression |
| Entity Extraction | Python `re` (regex) |
| Charts | Plotly |
| Config | python-dotenv |

---

*Happy chatting! 🤖*
